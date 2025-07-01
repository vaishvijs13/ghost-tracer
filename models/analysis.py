from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class AnalysisType(str, Enum):
    ROOT_CAUSE = "root_cause"
    ANOMALY_DETECTION = "anomaly_detection"
    LOG_SUMMARIZATION = "log_summarization"
    PATTERN_DETECTION = "pattern_detection"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RootCause(BaseModel):
    description: str = Field(..., description="Human-readable description of the potential cause")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    confidence_level: ConfidenceLevel = Field(..., description="Categorical confidence level")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence for this hypothesis")
    affected_services: List[str] = Field(default_factory=list, description="Services potentially affected")
    suggested_actions: List[str] = Field(default_factory=list, description="Recommended remediation steps")
    trace_ids: List[str] = Field(default_factory=list, description="Related trace IDs")
    log_patterns: List[str] = Field(default_factory=list, description="Log patterns that support this cause")


class RootCauseAnalysis(BaseModel):
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this analysis")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the analysis was performed")
    analysis_type: AnalysisType = Field(default=AnalysisType.ROOT_CAUSE, description="Type of analysis")
    
    # Query context
    query: str = Field(..., description="Original query that triggered this analysis")
    time_range_start: datetime = Field(..., description="Start of analyzed time range")
    time_range_end: datetime = Field(..., description="End of analyzed time range")
    
    # Analysis results
    root_causes: List[RootCause] = Field(default_factory=list, description="Potential root causes ranked by confidence")
    summary: str = Field(..., description="High-level summary of findings")
    
    # Metadata
    analyzed_logs_count: int = Field(default=0, description="Number of logs analyzed")
    analyzed_traces_count: int = Field(default=0, description="Number of traces analyzed")
    processing_time_ms: float = Field(default=0.0, description="Time taken to perform analysis")
    llm_model_used: str = Field(..., description="LLM model used for analysis")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    
    @property
    def top_root_cause(self) -> Optional[RootCause]:
        if not self.root_causes:
            return None
        return max(self.root_causes, key=lambda x: x.confidence)


class AnomalyScore(BaseModel):
    metric_name: str = Field(..., description="Name of the metric or pattern")
    score: float = Field(..., ge=0.0, le=1.0, description="Anomaly score between 0 (normal) and 1 (highly anomalous)")
    threshold: float = Field(..., description="Threshold used for anomaly detection")
    is_anomalous: bool = Field(..., description="Whether this is considered anomalous")
    baseline_value: Optional[float] = Field(None, description="Expected baseline value")
    actual_value: Optional[float] = Field(None, description="Actual observed value")
    deviation_percentage: Optional[float] = Field(None, description="Percentage deviation from baseline")


class AnomalyDetection(BaseModel):
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this detection")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the detection was performed")
    analysis_type: AnalysisType = Field(default=AnalysisType.ANOMALY_DETECTION, description="Type of analysis")
    
    # detection context
    time_window_start: datetime = Field(..., description="Start of detection window")
    time_window_end: datetime = Field(..., description="End of detection window")
    service_name: Optional[str] = Field(None, description="Specific service analyzed")
    
    # detection results
    anomaly_scores: List[AnomalyScore] = Field(default_factory=list, description="Anomaly scores for different metrics")
    overall_anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Overall anomaly score")
    is_anomalous: bool = Field(..., description="Whether an anomaly was detected")
    
    # context
    description: str = Field(..., description="Human-readable description of the anomaly")
    potential_causes: List[str] = Field(default_factory=list, description="Potential causes of the anomaly")
    affected_components: List[str] = Field(default_factory=list, description="Components showing anomalous behavior")
    
    # metadata
    detection_method: str = Field(..., description="Method used for anomaly detection")
    model_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the detection")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class LogSummary(BaseModel):
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this summary")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the summary was generated")
    analysis_type: AnalysisType = Field(default=AnalysisType.LOG_SUMMARIZATION, description="Type of analysis")
    
    time_range_start: datetime = Field(..., description="Start of summarized time range")
    time_range_end: datetime = Field(..., description="End of summarized time range")
    service_names: List[str] = Field(default_factory=list, description="Services included in summary")
    trace_ids: List[str] = Field(default_factory=list, description="Trace IDs included in summary")
    
    # summary content
    summary: str = Field(..., description="Human-readable summary of log activity")
    key_events: List[str] = Field(default_factory=list, description="Important events identified")
    error_patterns: List[str] = Field(default_factory=list, description="Common error patterns found")
    performance_insights: List[str] = Field(default_factory=list, description="Performance-related observations")
    
    total_logs_processed: int = Field(default=0, description="Total number of logs processed")
    error_count: int = Field(default=0, description="Number of error-level logs")
    warning_count: int = Field(default=0, description="Number of warning-level logs")
    unique_services_count: int = Field(default=0, description="Number of unique services")
    
    # metadata
    llm_model_used: str = Field(..., description="LLM model used for summarization")
    processing_time_ms: float = Field(default=0.0, description="Time taken to generate summary")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        } 