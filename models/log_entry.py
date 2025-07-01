from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class LogLevel(str, Enum):
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"


class TraceSpan(BaseModel):
    span_id: str = Field(..., description="Unique identifier for this span")
    trace_id: str = Field(..., description="Trace ID that groups related spans")
    parent_span_id: Optional[str] = Field(None, description="Parent span ID for hierarchical traces")
    operation_name: str = Field(..., description="Name of the operation this span represents")
    service_name: str = Field(..., description="Name of the service that created this span")
    start_time: datetime = Field(..., description="When the span started")
    end_time: Optional[datetime] = Field(None, description="When the span ended")
    duration_ms: Optional[float] = Field(None, description="Duration in milliseconds")
    tags: Dict[str, Any] = Field(default_factory=dict, description="Key-value tags for the span")
    logs: List[Dict[str, Any]] = Field(default_factory=list, description="Structured logs within the span")
    status: str = Field(default="OK", description="Span status (OK, ERROR, TIMEOUT, etc.)")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LogEntry(BaseModel):
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this log entry")
    timestamp: datetime = Field(..., description="When the log was created")
    level: LogLevel = Field(..., description="Log severity level")
    message: str = Field(..., description="The log message content")
    service_name: str = Field(..., description="Name of the service that created this log")
    
    # trace info
    trace_id: Optional[str] = Field(None, description="Associated trace ID")
    span_id: Optional[str] = Field(None, description="Associated span ID")
    
    # structured data
    labels: Dict[str, str] = Field(default_factory=dict, description="Key-value labels")
    fields: Dict[str, Any] = Field(default_factory=dict, description="Structured fields")
    
    # context info
    host: Optional[str] = Field(None, description="Host that generated the log")
    pod: Optional[str] = Field(None, description="Kubernetes pod name")
    namespace: Optional[str] = Field(None, description="Kubernetes namespace")
    container: Optional[str] = Field(None, description="Container name")
    
    # error context
    error: Optional[str] = Field(None, description="Error message if this is an error log")
    stack_trace: Optional[str] = Field(None, description="Stack trace for errors")
    
    # processing metadata
    ingested_at: datetime = Field(default_factory=datetime.utcnow, description="When this log was ingested")
    processed_at: Optional[datetime] = Field(None, description="When this log was processed by AI")
    embedding_vector: Optional[List[float]] = Field(None, description="Vector embedding for semantic search")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    
    @property
    def full_message(self) -> str:
        parts = [self.message]
        
        if self.error:
            parts.append(f"Error: {self.error}")
        
        if self.fields:
            field_strs = [f"{k}={v}" for k, v in self.fields.items()]
            parts.append(f"Fields: {', '.join(field_strs)}")
        
        if self.labels:
            label_strs = [f"{k}={v}" for k, v in self.labels.items()]
            parts.append(f"Labels: {', '.join(label_strs)}")
        
        return " | ".join(parts)
    
    @property
    def is_error(self) -> bool:
        return self.level in [LogLevel.ERROR, LogLevel.FATAL]
    
    @property
    def context_key(self) -> str:
        parts = [self.service_name]
        if self.trace_id:
            parts.append(f"trace:{self.trace_id}")
        if self.pod:
            parts.append(f"pod:{self.pod}")
        return ":".join(parts) 