from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import structlog

from ai.llm import LLMService
from database import get_db_session, LogRepository

logger = structlog.get_logger(__name__)


class AnomalyDetector:  
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def detect_anomalies(
        self,
        time_window_minutes: int = 30,
        services: Optional[List[str]] = None,
        sensitivity: float = 0.7
    ) -> Dict[str, Any]:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=time_window_minutes)
        current_stats = await self._get_period_stats(start_time, end_time, services)
        baseline_start = start_time - timedelta(days=1)
        baseline_end = end_time - timedelta(days=1)
        baseline_stats = await self._get_period_stats(baseline_start, baseline_end, services)
        
        logs_data = await self._fetch_logs_for_analysis(start_time, end_time, services)
        
        statistical_anomalies = self._detect_statistical_anomalies(
            current_stats, baseline_stats, sensitivity
        )
        
        ai_anomalies = await self._detect_ai_anomalies(logs_data)
        
        # combine and rank anomalies
        all_anomalies = statistical_anomalies + ai_anomalies
        all_anomalies.sort(key=lambda x: x.get('severity_score', 0), reverse=True)
        
        return {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "time_window": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_minutes": time_window_minutes
            },
            "anomalies_detected": len(all_anomalies),
            "high_severity_count": len([a for a in all_anomalies if a.get('severity_score', 0) > 0.8]),
            "anomalies": all_anomalies[:10],
            "system_health_score": self._calculate_health_score(current_stats, all_anomalies),
            "baseline_comparison": {
                "current_period": current_stats,
                "baseline_period": baseline_stats
            },
            "recommendations": self._generate_recommendations(all_anomalies)
        }
    
    async def _get_period_stats(
        self, 
        start_time: datetime, 
        end_time: datetime, 
        services: Optional[List[str]]
    ) -> Dict[str, Any]:
        async with get_db_session() as session:
            log_repo = LogRepository(session)
            return await log_repo.get_log_statistics(start_time, end_time, services)
    
    async def _fetch_logs_for_analysis(
        self,
        start_time: datetime,
        end_time: datetime, 
        services: Optional[List[str]]
    ) -> List[Any]:
        async with get_db_session() as session:
            log_repo = LogRepository(session)
            return await log_repo.get_logs_by_time_range(
                start_time=start_time,
                end_time=end_time,
                services=services,
                limit=1000
            )
    
    def _detect_statistical_anomalies(
        self, 
        current_stats: Dict, 
        baseline_stats: Dict, 
        sensitivity: float
    ) -> List[Dict[str, Any]]:
        anomalies = []
        threshold = 1.0 + (1.0 - sensitivity)
        
        current_error_rate = current_stats['error_count'] / max(current_stats['total_count'], 1)
        baseline_error_rate = baseline_stats['error_count'] / max(baseline_stats['total_count'], 1)
        
        if current_error_rate > baseline_error_rate * threshold:
            severity = min(1.0, (current_error_rate / max(baseline_error_rate, 0.001)) - 1.0)
            anomalies.append({
                "type": "error_rate_spike",
                "title": "Elevated Error Rate Detected",
                "description": f"Error rate increased from {baseline_error_rate:.2%} to {current_error_rate:.2%}",
                "severity_score": severity,
                "current_value": current_error_rate,
                "baseline_value": baseline_error_rate,
                "change_percentage": ((current_error_rate - baseline_error_rate) / max(baseline_error_rate, 0.001)) * 100
            })
        
        volume_ratio = current_stats['total_count'] / max(baseline_stats['total_count'], 1)
        if volume_ratio > threshold or volume_ratio < (1.0 / threshold):
            severity = abs(1.0 - volume_ratio)
            anomalies.append({
                "type": "log_volume_anomaly",
                "title": "Unusual Log Volume Detected",
                "description": f"Log volume changed from {baseline_stats['total_count']} to {current_stats['total_count']}",
                "severity_score": min(1.0, severity),
                "current_value": current_stats['total_count'],
                "baseline_value": baseline_stats['total_count'],
                "change_percentage": ((current_stats['total_count'] - baseline_stats['total_count']) / max(baseline_stats['total_count'], 1)) * 100
            })
        
        for service, current_count in current_stats['service_counts'].items():
            baseline_count = baseline_stats['service_counts'].get(service, 0)
            if baseline_count > 0:
                service_ratio = current_count / baseline_count
                if service_ratio > threshold * 2 or service_ratio < (1.0 / (threshold * 2)):
                    severity = min(1.0, abs(1.0 - service_ratio) / 2.0)
                    anomalies.append({
                        "type": "service_anomaly",
                        "title": f"Service Activity Anomaly: {service}",
                        "description": f"Service {service} log activity changed significantly",
                        "severity_score": severity,
                        "service": service,
                        "current_value": current_count,
                        "baseline_value": baseline_count,
                        "change_percentage": ((current_count - baseline_count) / baseline_count) * 100
                    })
        
        return anomalies
    
    async def _detect_ai_anomalies(self, logs_data: List[Any]) -> List[Dict[str, Any]]:
        if not logs_data:
            return []
        
        logs_context = "\n".join([
            f"[{log.timestamp}] {log.level} {log.service_name}: {log.message}"
            for log in logs_data[:200]
        ])
        
        system_message = """You are an expert system reliability engineer analyzing logs for anomalies.
        Identify unusual patterns, error sequences, or concerning trends that might indicate system issues.
        Focus on actionable insights."""
        
        prompt = f"""Analyze the following recent system logs for anomalies and unusual patterns:

{logs_context}

Identify any concerning patterns such as:
- Unusual error sequences or cascading failures  
- New types of errors not typically seen
- Performance degradation indicators
- Suspicious activity patterns
- Resource exhaustion signs
- Integration/dependency issues

For each anomaly found, provide:
1. Type of anomaly
2. Severity level (1-10)
3. Brief description
4. Potential impact
5. Recommended action

Format your response as a clear, structured analysis."""
        
        try:
            ai_response = await self.llm_service.analyze_logs(logs_context, prompt, "anomaly_detection")
            
            ai_anomalies = []
            if "anomaly" in ai_response.lower() or "unusual" in ai_response.lower():
                ai_anomalies.append({
                    "type": "ai_pattern_anomaly",
                    "title": "AI-Detected Pattern Anomaly",
                    "description": ai_response[:200] + "..." if len(ai_response) > 200 else ai_response,
                    "severity_score": 0.6,  # Default moderate severity
                    "source": "llm_analysis",
                    "full_analysis": ai_response
                })
            
            return ai_anomalies
            
        except Exception as e:
            logger.error("AI anomaly detection failed", error=str(e))
            return []
    
    def _calculate_health_score(self, current_stats: Dict, anomalies: List[Dict]) -> float:
        base_score = 1.0
        
        error_rate = current_stats['error_count'] / max(current_stats['total_count'], 1)
        base_score -= min(0.5, error_rate * 2)
        
        for anomaly in anomalies:
            severity_impact = anomaly.get('severity_score', 0) * 0.1
            base_score -= severity_impact
        
        return max(0.0, base_score)
    
    def _generate_recommendations(self, anomalies: List[Dict]) -> List[str]:
        recommendations = []
        
        error_anomalies = [a for a in anomalies if a.get('type') == 'error_rate_spike']
        volume_anomalies = [a for a in anomalies if a.get('type') == 'log_volume_anomaly']
        service_anomalies = [a for a in anomalies if a.get('type') == 'service_anomaly']
        
        if error_anomalies:
            recommendations.append("Investigate recent deployments or configuration changes that may have introduced errors")
            recommendations.append("Check error logs for specific error patterns and affected endpoints")
        
        if volume_anomalies:
            recommendations.append("Monitor system resources (CPU, memory, disk) for capacity issues")
            recommendations.append("Review application logic for logging loops or verbose logging configurations")
        
        if service_anomalies:
            services = [a.get('service') for a in service_anomalies]
            recommendations.append(f"Focus monitoring on services with anomalies: {', '.join(services)}")
            recommendations.append("Check inter-service dependencies and network connectivity")
        
        if not recommendations:
            recommendations.append("No immediate action required - continue monitoring")
        
        return recommendations


class HealthMonitor:
    @staticmethod
    async def get_detailed_health() -> Dict[str, Any]:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)
        
        try:
            async with get_db_session() as session:
                log_repo = LogRepository(session)
                stats = await log_repo.get_log_statistics(start_time, end_time, None)
            
            error_rate = stats['error_count'] / max(stats['total_count'], 1)
            
            return {
                "status": "healthy" if error_rate < 0.1 else "degraded",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "0.1.0",
                "metrics": {
                    "recent_logs_count": stats['total_count'],
                    "recent_error_rate": f"{error_rate:.2%}",
                    "recent_error_count": stats['error_count'],
                    "active_services": len(stats['service_counts'])
                },
                "uptime_seconds": (datetime.utcnow() - datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
            }
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            } 