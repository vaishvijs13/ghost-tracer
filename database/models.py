from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, String, DateTime, Integer, Text, JSON, Index, Float
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.sql import func
import uuid

from .connection import Base


class LogEntryModel(Base):
    __tablename__ = "log_entries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    level = Column(String(10), nullable=False, index=True)
    message = Column(Text, nullable=False)
    service_name = Column(String(255), nullable=False, index=True)
    
    # Trace information
    trace_id = Column(String(255), nullable=True, index=True)
    span_id = Column(String(255), nullable=True, index=True)
    
    # Structured data - stored as JSON
    labels = Column(JSON, nullable=True)
    fields = Column(JSON, nullable=True)
    
    # Context information
    host = Column(String(255), nullable=True)
    pod = Column(String(255), nullable=True)
    namespace = Column(String(255), nullable=True)
    container = Column(String(255), nullable=True)
    
    # Error context
    error = Column(Text, nullable=True)
    stack_trace = Column(Text, nullable=True)
    
    # Processing metadata
    ingested_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Vector embedding - stored as array of floats
    embedding_vector = Column(ARRAY(Float), nullable=True)
    
    # Create indexes for common queries
    __table_args__ = (
        Index('ix_log_entries_timestamp_service', 'timestamp', 'service_name'),
        Index('ix_log_entries_level_timestamp', 'level', 'timestamp'),
        Index('ix_log_entries_trace_timestamp', 'trace_id', 'timestamp'),
        Index('ix_log_entries_service_level', 'service_name', 'level'),
    )


class TraceSpanModel(Base):
    __tablename__ = "trace_spans"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    span_id = Column(String(255), nullable=False, unique=True, index=True)
    trace_id = Column(String(255), nullable=False, index=True)
    parent_span_id = Column(String(255), nullable=True, index=True)
    
    operation_name = Column(String(255), nullable=False)
    service_name = Column(String(255), nullable=False, index=True)
    
    start_time = Column(DateTime(timezone=True), nullable=False, index=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Float, nullable=True)
    
    # Structured data
    tags = Column(JSON, nullable=True)
    logs = Column(JSON, nullable=True)
    status = Column(String(50), default="OK", nullable=False)
    
    # Processing metadata
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('ix_trace_spans_trace_start', 'trace_id', 'start_time'),
        Index('ix_trace_spans_service_operation', 'service_name', 'operation_name'),
        Index('ix_trace_spans_parent_trace', 'parent_span_id', 'trace_id'),
    ) 