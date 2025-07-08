from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, text, desc, asc, func
from sqlalchemy.dialects.postgresql import insert
import structlog

from models.log_entry import LogEntry, LogLevel
from models.search import SearchQuery, SearchResult, SearchResultItem, SearchAggregation
from .models import LogEntryModel, TraceSpanModel

logger = structlog.get_logger(__name__)


class LogRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_log(self, log_entry: LogEntry) -> LogEntryModel:
        """Store a single log entry"""
        try:
            db_log = LogEntryModel(
                id=log_entry.id,
                timestamp=log_entry.timestamp,
                level=log_entry.level.value,
                message=log_entry.message,
                service_name=log_entry.service_name,
                trace_id=log_entry.trace_id,
                span_id=log_entry.span_id,
                labels=log_entry.labels,
                fields=log_entry.fields,
                host=log_entry.host,
                pod=log_entry.pod,
                namespace=log_entry.namespace,
                container=log_entry.container,
                error=log_entry.error,
                stack_trace=log_entry.stack_trace,
                ingested_at=log_entry.ingested_at,
                processed_at=log_entry.processed_at,
                embedding_vector=log_entry.embedding_vector
            )
            
            self.session.add(db_log)
            await self.session.flush()
            return db_log
            
        except Exception as e:
            logger.error("Failed to create log entry", error=str(e))
            raise
    
    async def create_logs_batch(self, log_entries: List[LogEntry]) -> None:
        """Store multiple log entries efficiently"""
        try:
            db_logs = []
            for log_entry in log_entries:
                db_log = LogEntryModel(
                    id=log_entry.id,
                    timestamp=log_entry.timestamp,
                    level=log_entry.level.value,
                    message=log_entry.message,
                    service_name=log_entry.service_name,
                    trace_id=log_entry.trace_id,
                    span_id=log_entry.span_id,
                    labels=log_entry.labels,
                    fields=log_entry.fields,
                    host=log_entry.host,
                    pod=log_entry.pod,
                    namespace=log_entry.namespace,
                    container=log_entry.container,
                    error=log_entry.error,
                    stack_trace=log_entry.stack_trace,
                    ingested_at=log_entry.ingested_at,
                    processed_at=log_entry.processed_at,
                    embedding_vector=log_entry.embedding_vector
                )
                db_logs.append(db_log)
            
            self.session.add_all(db_logs)
            await self.session.flush()
            
        except Exception as e:
            logger.error("Failed to create log batch", error=str(e))
            raise
    
    async def update_log_embedding(self, log_id: str, embedding: List[float]) -> None:
        """Update embedding vector for a log entry"""
        try:
            result = await self.session.execute(
                select(LogEntryModel).where(LogEntryModel.id == log_id)
            )
            log_entry = result.scalar_one_or_none()
            
            if log_entry:
                log_entry.embedding_vector = embedding
                log_entry.processed_at = datetime.utcnow()
                await self.session.flush()
                
        except Exception as e:
            logger.error("Failed to update log embedding", log_id=log_id, error=str(e))
            raise
    
    async def get_logs_by_time_range(
        self, 
        start_time: datetime, 
        end_time: datetime,
        services: Optional[List[str]] = None,
        levels: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[LogEntryModel]:
        """Fetch logs within a time range with optional filters"""
        try:
            query = select(LogEntryModel).where(
                and_(
                    LogEntryModel.timestamp >= start_time,
                    LogEntryModel.timestamp <= end_time
                )
            )
            
            if services:
                query = query.where(LogEntryModel.service_name.in_(services))
            
            if levels:
                query = query.where(LogEntryModel.level.in_(levels))
            
            query = query.order_by(desc(LogEntryModel.timestamp)).limit(limit)
            
            result = await self.session.execute(query)
            return result.scalars().all()
            
        except Exception as e:
            logger.error("Failed to fetch logs by time range", error=str(e))
            raise
    
    async def get_logs_by_trace_id(self, trace_id: str) -> List[LogEntryModel]:
        """Fetch all logs for a specific trace"""
        try:
            query = select(LogEntryModel).where(
                LogEntryModel.trace_id == trace_id
            ).order_by(LogEntryModel.timestamp)
            
            result = await self.session.execute(query)
            return result.scalars().all()
            
        except Exception as e:
            logger.error("Failed to fetch logs by trace ID", trace_id=trace_id, error=str(e))
            raise


class SearchRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def semantic_search(
        self,
        query_embedding: List[float],
        limit: int = 50,
        similarity_threshold: float = 0.7,
        time_range_start: Optional[datetime] = None,
        time_range_end: Optional[datetime] = None,
        services: Optional[List[str]] = None
    ) -> List[LogEntryModel]:
        """Perform semantic search using cosine similarity"""
        try:
            # Build the similarity query using PostgreSQL vector operations
            similarity_expr = text("""
                1 - (embedding_vector <=> :query_vector) as similarity_score
            """)
            
            query = select(
                LogEntryModel,
                similarity_expr.label('similarity_score')
            ).where(
                and_(
                    LogEntryModel.embedding_vector.isnot(None),
                    text("1 - (embedding_vector <=> :query_vector) >= :threshold")
                )
            )
            
            # Add filters
            if time_range_start:
                query = query.where(LogEntryModel.timestamp >= time_range_start)
            if time_range_end:
                query = query.where(LogEntryModel.timestamp <= time_range_end)
            if services:
                query = query.where(LogEntryModel.service_name.in_(services))
            
            # Order by similarity and limit
            query = query.order_by(text("similarity_score DESC")).limit(limit)
            
            result = await self.session.execute(
                query,
                {
                    "query_vector": query_embedding,
                    "threshold": similarity_threshold
                }
            )
            
            return result.all()
            
        except Exception as e:
            logger.error("Failed to perform semantic search", error=str(e))
            raise
    
    async def keyword_search(
        self,
        search_text: str,
        limit: int = 50,
        time_range_start: Optional[datetime] = None,
        time_range_end: Optional[datetime] = None,
        services: Optional[List[str]] = None
    ) -> List[LogEntryModel]:
        """Perform keyword search using PostgreSQL full-text search"""
        try:
            # Use PostgreSQL full-text search
            query = select(LogEntryModel).where(
                or_(
                    LogEntryModel.message.ilike(f"%{search_text}%"),
                    LogEntryModel.error.ilike(f"%{search_text}%") if search_text else False
                )
            )
            
            # Add filters
            if time_range_start:
                query = query.where(LogEntryModel.timestamp >= time_range_start)
            if time_range_end:
                query = query.where(LogEntryModel.timestamp <= time_range_end)
            if services:
                query = query.where(LogEntryModel.service_name.in_(services))
            
            query = query.order_by(desc(LogEntryModel.timestamp)).limit(limit)
            
            result = await self.session.execute(query)
            return result.scalars().all()
            
        except Exception as e:
            logger.error("Failed to perform keyword search", error=str(e))
            raise
    
    async def get_search_aggregations(
        self,
        time_range_start: Optional[datetime] = None,
        time_range_end: Optional[datetime] = None,
        services: Optional[List[str]] = None
    ) -> SearchAggregation:
        """Get aggregated data for search results"""
        try:
            base_query = select(LogEntryModel)
            
            # Add filters
            if time_range_start:
                base_query = base_query.where(LogEntryModel.timestamp >= time_range_start)
            if time_range_end:
                base_query = base_query.where(LogEntryModel.timestamp <= time_range_end)
            if services:
                base_query = base_query.where(LogEntryModel.service_name.in_(services))
            
            # Total count
            count_query = select(func.count()).select_from(base_query.subquery())
            total_result = await self.session.execute(count_query)
            total_count = total_result.scalar()
            
            # Service counts
            service_query = select(
                LogEntryModel.service_name,
                func.count().label('count')
            ).group_by(LogEntryModel.service_name)
            
            if time_range_start:
                service_query = service_query.where(LogEntryModel.timestamp >= time_range_start)
            if time_range_end:
                service_query = service_query.where(LogEntryModel.timestamp <= time_range_end)
            
            service_result = await self.session.execute(service_query)
            services_dict = {row.service_name: row.count for row in service_result}
            
            # Level counts
            level_query = select(
                LogEntryModel.level,
                func.count().label('count')
            ).group_by(LogEntryModel.level)
            
            if time_range_start:
                level_query = level_query.where(LogEntryModel.timestamp >= time_range_start)
            if time_range_end:
                level_query = level_query.where(LogEntryModel.timestamp <= time_range_end)
            
            level_result = await self.session.execute(level_query)
            levels_dict = {row.level: row.count for row in level_result}
            
            # Error count
            error_count = levels_dict.get('ERROR', 0) + levels_dict.get('FATAL', 0)
            
            # Trace count
            trace_query = select(func.count(func.distinct(LogEntryModel.trace_id))).where(
                LogEntryModel.trace_id.isnot(None)
            )
            
            if time_range_start:
                trace_query = trace_query.where(LogEntryModel.timestamp >= time_range_start)
            if time_range_end:
                trace_query = trace_query.where(LogEntryModel.timestamp <= time_range_end)
            
            trace_result = await self.session.execute(trace_query)
            trace_count = trace_result.scalar() or 0
            
            return SearchAggregation(
                total_count=total_count,
                services=services_dict,
                log_levels=levels_dict,
                error_count=error_count,
                trace_count=trace_count
            )
            
        except Exception as e:
            logger.error("Failed to get search aggregations", error=str(e))
            raise 