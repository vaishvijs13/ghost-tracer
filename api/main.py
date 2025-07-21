import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional, Any, Callable
from functools import wraps
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import get_settings
from models.log_entry import LogEntry
from models.search import SearchQuery, SearchResult, SearchResultItem, SearchType
from models.analysis import RootCauseAnalysis, LogSummary
from ai.embeddings import EmbeddingService
from ai.llm import LLMService
from ai.anomaly_detector import AnomalyDetector, HealthMonitor
from database import DatabaseManager, get_db_session, LogRepository, SearchRepository

logger = structlog.get_logger(__name__)

embedding_service: Optional[EmbeddingService] = None
llm_service: Optional[LLMService] = None
db_manager: Optional[DatabaseManager] = None


def handle_api_errors(error_msg: str):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(error_msg, error=str(e))
                raise HTTPException(status_code=500, detail=error_msg)
        return wrapper
    return decorator


def format_logs_for_llm(logs_data: List[Any], limit: Optional[int] = None) -> str:
    logs_to_format = logs_data[:limit] if limit else logs_data
    return "\n".join([
        f"[{log.timestamp}] {log.level} {log.service_name}: {log.message}"
        for log in logs_to_format
    ])


def create_search_result_item(log_model: Any, relevance_score: float = 1.0, similarity_score: Optional[float] = None) -> SearchResultItem:
    return SearchResultItem(
        log_id=log_model.id,
        relevance_score=relevance_score,
        similarity_score=similarity_score or relevance_score,
        timestamp=log_model.timestamp,
        level=log_model.level,
        message=log_model.message,
        service_name=log_model.service_name,
        trace_id=log_model.trace_id
    )


def extract_search_filters(query: SearchQuery) -> tuple[Optional[datetime], Optional[datetime], Optional[List[str]]]:
    if not query.filters:
        return None, None, None
    return query.filters.time_range_start, query.filters.time_range_end, query.filters.services


async def fetch_logs_with_repo(
time_start: datetime,
    time_end: datetime,
    services: Optional[List[str]] = None,
    levels: Optional[List[str]] = None,
    limit: int = 1000
) -> List[Any]:
    async with get_db_session() as session:
        log_repo = LogRepository(session)
        return await log_repo.get_logs_by_time_range(
            start_time=time_start,
            end_time=time_end,
            services=services,
            levels=levels,
            limit=limit
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_service, llm_service, db_manager
    
    try:
        logger.info("Starting ghost_tracer API server")
        
        # database init
        db_manager = DatabaseManager()
        await db_manager.initialize()
        await db_manager.create_tables()
        
        embedding_service = EmbeddingService()
        await embedding_service.initialize()
        
        llm_service = LLMService()
        await llm_service.initialize()
        
        logger.info("All services initialized successfully")
        yield
        
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        raise
    finally:
        if db_manager:
            await db_manager.close()
        logger.info("Shutting down ghost_tracer API server")


app = FastAPI(
    title="ghost_tracer",
    description="AI-Powered Distributed Systems Debugging",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()


def get_embedding_service() -> EmbeddingService:
    if embedding_service is None:
        raise HTTPException(status_code=503, detail="Embedding service not initialized")
    return embedding_service


def get_llm_service() -> LLMService:
    if llm_service is None:
        raise HTTPException(status_code=503, detail="LLM service not initialized")
    return llm_service


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0"
    }


@app.post("/api/v1/logs/ingest")
@handle_api_errors("Failed to ingest log")
async def ingest_log(
    log_entry: LogEntry,
    background_tasks: BackgroundTasks,
    embedding_svc: EmbeddingService = Depends(get_embedding_service)
):
    async with get_db_session() as session:
        log_repo = LogRepository(session)
        await log_repo.create_log(log_entry)
    
    # generate embedding in background
    background_tasks.add_task(process_log_embedding, log_entry, embedding_svc)
    
    return {
        "status": "accepted",
        "log_id": str(log_entry.id),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/v1/logs/batch-ingest")
@handle_api_errors("Failed to batch ingest logs")
async def batch_ingest_logs(
    log_entries: List[LogEntry],
    background_tasks: BackgroundTasks,
    embedding_svc: EmbeddingService = Depends(get_embedding_service)
):
    if len(log_entries) > settings.max_log_batch_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size too large. Maximum: {settings.max_log_batch_size}"
        )
    
    # store logs in database
    async with get_db_session() as session:
        log_repo = LogRepository(session)
        await log_repo.create_logs_batch(log_entries)
    
    background_tasks.add_task(process_batch_embeddings, log_entries, embedding_svc)
    
    return {
        "status": "accepted",
        "count": len(log_entries),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/v1/search", response_model=SearchResult)
@handle_api_errors("Search failed")
async def search_logs(
    query: SearchQuery,
    embedding_svc: EmbeddingService = Depends(get_embedding_service)
):
    start_time = datetime.utcnow()
    
    async with get_db_session() as session:
        search_repo = SearchRepository(session)
        time_range_start, time_range_end, services = extract_search_filters(query)
        
        # perform search based on type
        if query.search_type == SearchType.SEMANTIC:
            query_embedding = await embedding_svc.embed_text(query.query_text)
            results = await search_repo.semantic_search(
                query_embedding=query_embedding,
                limit=query.limit,
                similarity_threshold=query.similarity_threshold,
                time_range_start=time_range_start,
                time_range_end=time_range_end,
                services=services
            )
            
            items = []
            for result in results:
                if hasattr(result, 'LogEntryModel'):
                    log_model = result.LogEntryModel
                    similarity_score = result.similarity_score
                else:
                    log_model = result
                    similarity_score = 0.0
                
                items.append(create_search_result_item(log_model, similarity_score, similarity_score))
            
        else:
            results = await search_repo.keyword_search(
                search_text=query.query_text,
                limit=query.limit,
                time_range_start=time_range_start,
                time_range_end=time_range_end,
                services=services
            )
            
            items = [create_search_result_item(log_model) for log_model in results]
        
        # get aggregations
        aggregations = await search_repo.get_search_aggregations(
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            services=services
        )
        
        search_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return SearchResult(
            query_id=query.id,
            query_text=query.query_text,
            items=items,
            total_count=aggregations.total_count,
            returned_count=len(items),
            has_more=len(items) == query.limit,
            next_offset=query.offset + len(items) if len(items) == query.limit else None,
            aggregations=aggregations,
            search_time_ms=search_time
        )


@app.post("/api/v1/analysis/root-cause", response_model=RootCauseAnalysis)
@handle_api_errors("Root cause analysis failed")
async def analyze_root_cause(
    query: str,
    time_start: datetime,
    time_end: datetime,
    services: Optional[List[str]] = None,
    llm_svc: LLMService = Depends(get_llm_service)
):
    # fetch error logs from database
    error_logs_data = await fetch_logs_with_repo(
        time_start=time_start,
        time_end=time_end,
        services=services,
        levels=["ERROR", "FATAL"],
        limit=500
    )
    
    context_logs_data = await fetch_logs_with_repo(
        time_start=time_start - timedelta(minutes=10),
        time_end=time_end + timedelta(minutes=10),
        services=services,
        limit=1000
    )
    
    error_logs = format_logs_for_llm(error_logs_data)
    context_logs = format_logs_for_llm(context_logs_data, limit=100)
    
    analysis_result = await llm_svc.generate_root_cause_analysis(
        error_logs=error_logs,
        context_logs=context_logs
    )
    
    return RootCauseAnalysis(
        query=query,
        time_range_start=time_start,
        time_range_end=time_end,
        summary=analysis_result.get("analysis", ""),
        llm_model_used=settings.llm_provider,
        analyzed_logs_count=len(error_logs_data) + len(context_logs_data),
        root_causes=analysis_result.get("root_causes", [])
    )


@app.post("/api/v1/analysis/summarize", response_model=LogSummary)
@handle_api_errors("Log summarization failed")
async def summarize_logs(
    time_start: datetime,
    time_end: datetime,
    services: Optional[List[str]] = None,
    llm_svc: LLMService = Depends(get_llm_service)
):
    logs_data = await fetch_logs_with_repo(
        time_start=time_start,
        time_end=time_end,
        services=services,
        limit=1000
    )
    
    logs_context = format_logs_for_llm(logs_data)
    
    summary_text = await llm_svc.summarize_logs(
        logs_context=logs_context,
        time_range=f"{time_start} to {time_end}"
    )
    
    # calculate statistics
    error_count = sum(1 for log in logs_data if log.level in ["ERROR", "FATAL"])
    warning_count = sum(1 for log in logs_data if log.level == "WARN")
    unique_services = set(log.service_name for log in logs_data)
    
    return LogSummary(
        time_range_start=time_start,
        time_range_end=time_end,
        service_names=services or list(unique_services),
        summary=summary_text,
        llm_model_used=settings.llm_provider,
        total_logs_processed=len(logs_data),
        error_count=error_count,
        warning_count=warning_count,
        unique_services_count=len(unique_services)
    )


@app.post("/api/v1/analysis/chat")
@handle_api_errors("Chat analysis failed")
async def chat_with_logs(
    query: str,
    context_time_minutes: int = 30,
    llm_svc: LLMService = Depends(get_llm_service)
):
    # fetch relevant logs from recent time window
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=context_time_minutes)
    
    logs_data = await fetch_logs_with_repo(
        time_start=start_time,
        time_end=end_time,
        limit=500
    )
    
    logs_context = format_logs_for_llm(logs_data)
    
    response = await llm_svc.analyze_logs(
        logs_context=logs_context,
        query=query,
        analysis_type="general"
    )
    
    return {
        "query": query,
        "response": response,
        "timestamp": datetime.utcnow().isoformat(),
        "context_logs_count": len(logs_data)
    }


@app.post("/api/v1/analysis/anomaly-detection")
@handle_api_errors("Anomaly detection failed")
async def detect_anomalies(
    time_window_minutes: int = Query(30, description="Time window to analyze in minutes"),
    services: Optional[List[str]] = Query(None, description="Services to analyze"),
    sensitivity: float = Query(0.7, description="Anomaly sensitivity (0.1-1.0)"),
    llm_svc: LLMService = Depends(get_llm_service)
):
    anomaly_detector = AnomalyDetector(llm_svc)
    return await anomaly_detector.detect_anomalies(
        time_window_minutes=time_window_minutes,
        services=services,
        sensitivity=sensitivity
    )


@app.get("/api/v1/health/detailed")
async def detailed_health_check():
    try:
        health_data = await HealthMonitor.get_detailed_health()
        
        service_status = {
            "database": "healthy",
            "embedding_service": "healthy" if embedding_service else "unavailable",
            "llm_service": "healthy" if llm_service else "unavailable"
        }
        
        health_data["services"] = service_status
        return health_data
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail="Health check failed")


async def process_log_embedding(log_entry: LogEntry, embedding_svc: EmbeddingService):
    try:
        embedding = await embedding_svc.embed_log_entry(log_entry)
        
        # update embedding in database
        async with get_db_session() as session:
            log_repo = LogRepository(session)
            await log_repo.update_log_embedding(str(log_entry.id), embedding)
        
        logger.info("Generated embedding for log", log_id=str(log_entry.id))
    
    except Exception as e:
        logger.error("Failed to process log embedding", log_id=str(log_entry.id), error=str(e))


async def process_batch_embeddings(log_entries: List[LogEntry], embedding_svc: EmbeddingService):
    try:
        embeddings = await embedding_svc.embed_log_entries(log_entries)
        
        # update embeddings in database
        async with get_db_session() as session:
            log_repo = LogRepository(session)
            for log_entry, embedding in zip(log_entries, embeddings):
                await log_repo.update_log_embedding(str(log_entry.id), embedding)
        
        logger.info("Generated embeddings for log batch", count=len(log_entries))
    
    except Exception as e:
        logger.error("Failed to process batch embeddings", count=len(log_entries), error=str(e))


def main():
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )


if __name__ == "__main__":
    main() 