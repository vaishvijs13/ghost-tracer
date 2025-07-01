from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class SearchType(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    TRACE = "trace"
    TIME_RANGE = "time_range"
    HYBRID = "hybrid"


class SortOrder(str, Enum):
    RELEVANCE = "relevance"
    TIMESTAMP_ASC = "timestamp_asc"
    TIMESTAMP_DESC = "timestamp_desc"
    CONFIDENCE = "confidence"


class SearchFilter(BaseModel):
    services: Optional[List[str]] = Field(None, description="Filter by service names")
    log_levels: Optional[List[str]] = Field(None, description="Filter by log levels")
    time_start: Optional[datetime] = Field(None, description="Start time filter")
    time_end: Optional[datetime] = Field(None, description="End time filter")
    trace_ids: Optional[List[str]] = Field(None, description="Filter by trace IDs")
    hosts: Optional[List[str]] = Field(None, description="Filter by host names")
    namespaces: Optional[List[str]] = Field(None, description="Filter by Kubernetes namespaces")
    has_errors: Optional[bool] = Field(None, description="Filter for logs with errors")
    custom_fields: Optional[Dict[str, Any]] = Field(None, description="Custom field filters")


class SearchQuery(BaseModel):
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this query")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the query was created")
    
    # query content
    query_text: str = Field(..., description="The search query text")
    search_type: SearchType = Field(default=SearchType.SEMANTIC, description="Type of search to perform")
    
    limit: int = Field(default=50, ge=1, le=1000, description="Maximum number of results to return")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")
    sort_by: SortOrder = Field(default=SortOrder.RELEVANCE, description="How to sort results")
    filters: Optional[SearchFilter] = Field(None, description="Additional filter criteria")
    
    # semantic params
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity threshold for semantic search")
    include_embeddings: bool = Field(default=False, description="Whether to include embedding vectors in results")
    
    use_ai_expansion: bool = Field(default=True, description="Whether to use AI to expand/refine the query")
    context_window_minutes: int = Field(default=30, description="Time window for contextual results")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class SearchResultItem(BaseModel):
    log_id: UUID = Field(..., description="ID of the matching log entry")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score for this result")
    similarity_score: Optional[float] = Field(None, description="Semantic similarity score")
    
    # log content
    timestamp: datetime = Field(..., description="Log timestamp")
    level: str = Field(..., description="Log level")
    message: str = Field(..., description="Log message")
    service_name: str = Field(..., description="Service that generated the log")
    trace_id: Optional[str] = Field(None, description="Associated trace ID")
    
    # context
    context_before: Optional[List[str]] = Field(None, description="Log messages before this one for context")
    context_after: Optional[List[str]] = Field(None, description="Log messages after this one for context")
    
    # highlighting
    highlighted_text: Optional[str] = Field(None, description="Text with search terms highlighted")
    matched_fields: List[str] = Field(default_factory=list, description="Fields that matched the search")
    
    # metadata
    embedding_vector: Optional[List[float]] = Field(None, description="Embedding vector if requested")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class SearchAggregation(BaseModel):
    total_count: int = Field(..., description="Total number of matching results")
    services: Dict[str, int] = Field(default_factory=dict, description="Count by service")
    log_levels: Dict[str, int] = Field(default_factory=dict, description="Count by log level")
    time_distribution: Dict[str, int] = Field(default_factory=dict, description="Count by time bucket")
    trace_count: int = Field(default=0, description="Number of unique traces")
    error_count: int = Field(default=0, description="Number of error-level logs")
    
    # pattern analysis
    common_patterns: List[str] = Field(default_factory=list, description="Common patterns in results")
    related_terms: List[str] = Field(default_factory=list, description="Related terms found")


class SearchResult(BaseModel):
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this result set")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the search was performed")
    
    # query reference
    query_id: UUID = Field(..., description="ID of the original query")
    query_text: str = Field(..., description="Original query text")
    
    # results
    items: List[SearchResultItem] = Field(default_factory=list, description="Individual search results")
    total_count: int = Field(..., description="Total number of matching results")
    returned_count: int = Field(..., description="Number of results returned in this response")
    
    # pagination
    has_more: bool = Field(..., description="Whether there are more results available")
    next_offset: Optional[int] = Field(None, description="Offset for next page of results")
    
    # aggregations
    aggregations: SearchAggregation = Field(default_factory=SearchAggregation, description="Aggregated result information")
    
    # ai-gen
    ai_summary: Optional[str] = Field(None, description="AI-generated summary of search results")
    suggested_queries: List[str] = Field(default_factory=list, description="AI-suggested related queries")
    patterns_detected: List[str] = Field(default_factory=list, description="Patterns detected in results")
    
    search_time_ms: float = Field(..., description="Time taken to perform the search")
    ai_processing_time_ms: Optional[float] = Field(None, description="Time taken for AI processing")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        } 