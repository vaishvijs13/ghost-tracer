import asyncio
import aiohttp
import json
from typing import Optional, Dict, Any, List
import structlog
from abc import ABC, abstractmethod

from ..config import get_settings, LLMProvider

logger = structlog.get_logger(__name__)


class BaseLLMProvider(ABC):    
    @abstractmethod
    async def generate_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system_message: Optional[str] = None
    ) -> str:
        pass
    
    @abstractmethod
    async def initialize(self):
        pass


class OpenAIProvider(BaseLLMProvider):    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"
    
    async def initialize(self):
        logger.info("Initialized OpenAI provider", model=self.model)
    
    async def generate_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system_message: Optional[str] = None
    ) -> str:
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"OpenAI API error: {error_text}")
                    
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        
        except Exception as e:
            logger.error("OpenAI API error", error=str(e))
            raise


class OllamaProvider(BaseLLMProvider):    
    def __init__(self, host: str = "localhost", port: int = 11434, model: str = "mistral:7b"):
        self.host = host
        self.port = port
        self.model = model
        self.base_url = f"http://{host}:{port}"
    
    async def initialize(self):
        try:
            # test connection
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        logger.info("Initialized Ollama provider", model=self.model)
                    else:
                        raise Exception("Ollama server not accessible")
        except Exception as e:
            logger.error("Failed to initialize Ollama provider", error=str(e))
            raise
    
    async def generate_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system_message: Optional[str] = None
    ) -> str:
        try:
            full_prompt = prompt
            if system_message:
                full_prompt = f"System: {system_message}\n\nUser: {prompt}"
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error: {error_text}")
                    
                    result = await response.json()
                    return result["response"]
        
        except Exception as e:
            logger.error("Ollama API error", error=str(e))
            raise


class HuggingFaceProvider(BaseLLMProvider):    
    def __init__(self, model_name: str, token: Optional[str] = None):
        self.model_name = model_name
        self.token = token
        self.pipeline = None
    
    async def initialize(self):
        try:
            from transformers import pipeline
            
            loop = asyncio.get_event_loop()
            self.pipeline = await loop.run_in_executor(
                None, self._load_pipeline
            )
        
        except Exception as e:
            logger.error("Failed to initialize HuggingFace provider", error=str(e))
            raise
    
    def _load_pipeline(self):
        from transformers import pipeline
        return pipeline("text-generation", model=self.model_name, use_auth_token=self.token)
    
    async def generate_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system_message: Optional[str] = None
    ) -> str:
        try:
            full_prompt = prompt
            if system_message:
                full_prompt = f"{system_message}\n\n{prompt}"
            
            # generate in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._generate, full_prompt, max_tokens, temperature
            )
            
            return result
        
        except Exception as e:
            logger.error("HuggingFace generation error", error=str(e))
            raise
    
    def _generate(self, prompt: str, max_tokens: Optional[int], temperature: float) -> str:
        kwargs = {
            "temperature": temperature,
            "do_sample": True
        }
        
        if max_tokens:
            kwargs["max_new_tokens"] = max_tokens
        
        result = self.pipeline(prompt, **kwargs)
        return result[0]["generated_text"][len(prompt):].strip()


class LLMService:    
    def __init__(self):
        self.settings = get_settings()
        self.provider: Optional[BaseLLMProvider] = None
    
    async def initialize(self):
        try:
            if self.settings.llm_provider == LLMProvider.OPENAI:
                if not self.settings.openai_api_key:
                    raise ValueError("OpenAI API key not configured")
                self.provider = OpenAIProvider(
                    api_key=self.settings.openai_api_key,
                    model=self.settings.openai_model
                )
            
            elif self.settings.llm_provider == LLMProvider.OLLAMA:
                self.provider = OllamaProvider(
                    host=self.settings.ollama_host,
                    port=self.settings.ollama_port,
                    model=self.settings.ollama_model
                )
            
            elif self.settings.llm_provider == LLMProvider.HUGGINGFACE:
                self.provider = HuggingFaceProvider(
                    model_name=self.settings.hf_model,
                    token=self.settings.hf_token
                )
            
            else:
                raise ValueError(f"Unsupported LLM provider: {self.settings.llm_provider}")
            
            await self.provider.initialize()
            logger.info("LLM service initialized", provider=self.settings.llm_provider)
        
        except Exception as e:
            logger.error("Failed to initialize LLM service", error=str(e))
            raise
    
    async def analyze_logs(
        self,
        logs_context: str,
        query: str,
        analysis_type: str = "general"
    ) -> str:
        if not self.provider:
            raise RuntimeError("LLM service not initialized")
        
        system_message = self._get_system_message(analysis_type)
        prompt = self._build_analysis_prompt(logs_context, query, analysis_type)
        
        try:
            response = await self.provider.generate_response(
                prompt=prompt,
                system_message=system_message,
                max_tokens=self.settings.max_context_length,
                temperature=0.3
            )
            
            return response.strip()
        
        except Exception as e:
            logger.error("LLM analysis failed", error=str(e))
            raise
    
    async def summarize_logs(self, logs_context: str, time_range: str = "") -> str:
        if not self.provider:
            raise RuntimeError("LLM service not initialized")
        
        system_message = """You are an expert system administrator analyzing log data. 
        Provide clear, actionable summaries of system behavior. Focus on:
        1. Key events and patterns
        2. Error conditions and their frequency
        3. Performance insights
        4. Potential issues requiring attention"""
        
        prompt = f"""Analyze the following logs{f' from {time_range}' if time_range else ''}:

{logs_context}

Provide a comprehensive summary including:
- Overall system health assessment
- Key events and patterns identified
- Error analysis and trends
- Performance observations
- Recommended actions if any issues are found

Keep the summary concise but thorough."""
        
        try:
            response = await self.provider.generate_response(
                prompt=prompt,
                system_message=system_message,
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.strip()
        
        except Exception as e:
            logger.error("Log summarization failed", error=str(e))
            raise
    
    async def generate_root_cause_analysis(
        self,
        error_logs: str,
        context_logs: str,
        system_state: str = ""
    ) -> Dict[str, Any]:
        if not self.provider:
            raise RuntimeError("LLM service not initialized")
        
        system_message = """You are an expert systems engineer performing root cause analysis.
        Analyze the provided error logs and context to identify potential root causes.
        Always provide confidence scores and supporting evidence."""
        
        prompt = f"""Perform root cause analysis for the following system failure:

ERROR LOGS:
{error_logs}

CONTEXT LOGS:
{context_logs}

{f'SYSTEM STATE: {system_state}' if system_state else ''}

Please provide a structured analysis in the following format:

ROOT CAUSE ANALYSIS:
1. **Primary Hypothesis** (Confidence: X%)
   - Description: [Clear explanation]
   - Evidence: [Supporting log patterns/data]
   - Affected Services: [List of services]
   - Recommended Actions: [Specific remediation steps]

2. **Secondary Hypothesis** (Confidence: X%)
   - [Same format as above]

SUMMARY:
[Brief overall assessment]

Focus on actionable insights and concrete evidence from the logs."""
        
        try:
            response = await self.provider.generate_response(
                prompt=prompt,
                system_message=system_message,
                max_tokens=1500,
                temperature=0.2
            )
            
            return self._parse_root_cause_response(response)
        
        except Exception as e:
            logger.error("Root cause analysis failed", error=str(e))
            raise
    
    def _get_system_message(self, analysis_type: str) -> str:
        messages = {
            "general": "You are an expert system administrator analyzing logs for insights.",
            "error": "You are troubleshooting system errors. Focus on identifying root causes.",
            "performance": "You are analyzing system performance. Focus on bottlenecks and optimization opportunities.",
            "security": "You are analyzing logs for security issues. Focus on potential threats and anomalies."
        }
        return messages.get(analysis_type, messages["general"])
    
    def _build_analysis_prompt(self, logs_context: str, query: str, analysis_type: str) -> str:
        return f"""Analyze the following logs to answer this question: "{query}"

LOGS:
{logs_context}

Please provide a detailed analysis focusing on {analysis_type} aspects. 
Include specific evidence from the logs and actionable recommendations."""
    
    def _parse_root_cause_response(self, response: str) -> Dict[str, Any]:
        return {
            "analysis": response,
            "confidence": "medium",
            "recommendations": [],
            "affected_services": []
        } 