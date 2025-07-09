from datetime import datetime
from typing import List, Optional, Dict, Any, Callable
from functools import wraps
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, text, desc, asc, func
from sqlalchemy.dialects.postgresql import insert
import structlog

from models.log_entry import LogEntry, LogLevel
from models.search import SearchQuery, SearchResult, SearchResultItem, SearchAggregation
from .models import LogEntryModel, TraceSpanModel

logger = structlog.get_logger(__name__)


def handle_repository_errors(operation_name: str):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Repository operation failed: {operation_name}", error=str(e), **kwargs)
                raise
        return wrapper
    return decorator


def log_entry_to_model(log_entry: LogEntry) -> LogEntryModel:
    return LogEntryModel(
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


def apply_common_filters(query, time_range_start=None, time_range_end=None, services=None, levels=None):
    if time_range_start:
        query = query.where(LogEntryModel.timestamp >= time_range_start)
    if time_range_end:
        query = query.where(LogEntryModel.timestamp <= time_range_end)
    if services:
        query = query.where(LogEntryModel.service_name.in_(services))
    if levels:
        query = query.where(LogEntryModel.level.in_(levels))
    return query


class LogRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    @handle_repository_errors("create_log")
    async def create_log(self, log_entry: LogEntry) -> LogEntryModel:
        db_log = log_entry_to_model(log_entry)
        self.session.add(db_log)
        await self.session.flush()
        return db_log
    
    @handle_repository_errors("create_logs_batch")
    async def create_logs_batch(self, log_entries: List[LogEntry]) -> None:
        db_logs = [log_entry_to_model(log_entry) for log_entry in log_entries]
        self.session.add_all(db_logs)
        await self.session.flush()
    
    @handle_repository_errors("update_log_embedding")
    async def update_log_embedding(self, log_id: str, embedding: List[float]) -> None:
        result = await self.session.execute(
            select(LogEntryModel).where(LogEntryModel.id == log_id)
        )
        log_entry = result.scalar_one_or_none()
        
        if log_entry:
            log_entry.embedding_vector = embedding
            log_entry.processed_at = datetime.utcnow()
            await self.session.flush()
    
    @handle_repository_errors("get_logs_by_time_range")
    async def get_logs_by_time_range(
        self, 
        start_time: datetime, 
        end_time: datetime,
        services: Optional[List[str]] = None,
        levels: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[LogEntryModel]:
        query = select(LogEntryModel).where(
            and_(
                LogEntryModel.timestamp >= start_time,
                LogEntryModel.timestamp <= end_time
            )
        )
        
        query = apply_common_filters(query, services=services, levels=levels)
        query = query.order_by(desc(LogEntryModel.timestamp)).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    @handle_repository_errors("get_logs_by_trace_id")
    async def get_logs_by_trace_id(self, trace_id: str) -> List[LogEntryModel]:
        query = select(LogEntryModel).where(
            LogEntryModel.trace_id == trace_id
        ).order_by(LogEntryModel.timestamp)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    @handle_repository_errors("get_log_statistics")
    async def get_log_statistics(
        self,
        start_time: datetime,
        end_time: datetime,
        services: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        base_query = select(LogEntryModel).where(
            and_(
                LogEntryModel.timestamp >= start_time,
                LogEntryModel.timestamp <= end_time
            )
        )
        
        if services:
            base_query = base_query.where(LogEntryModel.service_name.in_(services))
        
        # Total count
        count_query = select(func.count()).select_from(base_query.subquery())
        total_result = await self.session.execute(count_query)
        total_count = total_result.scalar()
        
        # Counts by level
        level_query = select(
            LogEntryModel.level,
            func.count().label('count')
        ).where(
            and_(
                LogEntryModel.timestamp >= start_time,
                LogEntryModel.timestamp <= end_time
            )
        ).group_by(LogEntryModel.level)
        
        if services:
            level_query = level_query.where(LogEntryModel.service_name.in_(services))
        
        level_result = await self.session.execute(level_query)
        level_counts = {row.level: row.count for row in level_result}
        
        # Service counts
        service_query = select(
            LogEntryModel.service_name,
            func.count().label('count')
        ).where(
            and_(
                LogEntryModel.timestamp >= start_time,
                LogEntryModel.timestamp <= end_time
            )
        ).group_by(LogEntryModel.service_name)
        
        service_result = await self.session.execute(service_query)
        service_counts = {row.service_name: row.count for row in service_result}
        
        return {
            "total_count": total_count,
            "level_counts": level_counts,
            "service_counts": service_counts,
            "error_count": level_counts.get('ERROR', 0) + level_counts.get('FATAL', 0),
            "warning_count": level_counts.get('WARN', 0)
        }


class SearchRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    @handle_repository_errors("semantic_search")
    async def semantic_search(
        self,
        query_embedding: List[float],
        limit: int = 50,
        similarity_threshold: float = 0.7,
        time_range_start: Optional[datetime] = None,
        time_range_end: Optional[datetime] = None,
        services: Optional[List[str]] = None
    ) -> List[LogEntryModel]:
        # build the similarity query using PostgreSQL vector operations
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
        
        query = apply_common_filters(query, time_range_start, time_range_end, services)
        query = query.order_by(text("similarity_score DESC")).limit(limit)
        
        result = await self.session.execute(
            query,
            {
                "query_vector": query_embedding,
                "threshold": similarity_threshold
            }
        )
        
        return result.all()
    
    @handle_repository_errors("keyword_search")
    async def keyword_search(
        self,
        search_text: str,
        limit: int = 50,
        time_range_start: Optional[datetime] = None,
        time_range_end: Optional[datetime] = None,
        services: Optional[List[str]] = None
    ) -> List[LogEntryModel]:
        query = select(LogEntryModel).where(
            or_(
                LogEntryModel.message.ilike(f"%{search_text}%"),
                LogEntryModel.error.ilike(f"%{search_text}%") if search_text else False
            )
        )
        
        query = apply_common_filters(query, time_range_start, time_range_end, services)
        query = query.order_by(desc(LogEntryModel.timestamp)).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    @handle_repository_errors("get_search_aggregations")
    async def get_search_aggregations(
        self,
        time_range_start: Optional[datetime] = None,
        time_range_end: Optional[datetime] = None,
        services: Optional[List[str]] = None
    ) -> SearchAggregation:
        base_query = select(LogEntryModel)
        base_query = apply_common_filters(base_query, time_range_start, time_range_end, services)
        
        count_query = select(func.count()).select_from(base_query.subquery())
        total_result = await self.session.execute(count_query)
        total_count = total_result.scalar()
        
        # service counts
        service_query = select(
            LogEntryModel.service_name,
            func.count().label('count')
        ).group_by(LogEntryModel.service_name)
        service_query = apply_common_filters(service_query, time_range_start, time_range_end)
        
        service_result = await self.session.execute(service_query)
        services_dict = {row.service_name: row.count for row in service_result}
        
        # level counts
        level_query = select(
            LogEntryModel.level,
            func.count().label('count')
        ).group_by(LogEntryModel.level)
        level_query = apply_common_filters(level_query, time_range_start, time_range_end)
        
        level_result = await self.session.execute(level_query)
        levels_dict = {row.level: row.count for row in level_result}
        
        error_count = levels_dict.get('ERROR', 0) + levels_dict.get('FATAL', 0)
        
        # trace count
        trace_query = select(func.count(func.distinct(LogEntryModel.trace_id))).where(
            LogEntryModel.trace_id.isnot(None)
        )
        trace_query = apply_common_filters(trace_query, time_range_start, time_range_end)
        
        trace_result = await self.session.execute(trace_query)
        trace_count = trace_result.scalar() or 0
        
        return SearchAggregation(
            total_count=total_count,
            services=services_dict,
            log_levels=levels_dict,
            error_count=error_count,
            trace_count=trace_count
        ) 