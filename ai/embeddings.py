import asyncio
import numpy as np
from typing import List, Optional, Dict, Any
import structlog
from sentence_transformers import SentenceTransformer
import torch

from ..config import get_settings
from ..models.log_entry import LogEntry

logger = structlog.get_logger(__name__)


class EmbeddingService:    
    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.embedding_model
        self.model = None
        self.dimension = self.settings.embedding_dimension
        self._cache: Dict[str, List[float]] = {}
        
    async def initialize(self):
        try:
            logger.info("Loading embedding model", model=self.model_name)
            
            # load model in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, self._load_model
            )
            
            test_embedding = await self.embed_text("test")
            self.dimension = len(test_embedding)
            
            logger.info(
                "Embedding model loaded successfully",
                model=self.model_name,
                dimension=self.dimension
            )
            
        except Exception as e:
            logger.error("Failed to load embedding model", error=str(e))
            raise
    
    def _load_model(self) -> SentenceTransformer:
        return SentenceTransformer(self.model_name)
    
    async def embed_text(self, text: str) -> List[float]:
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        if text in self._cache:
            return self._cache[text]
        
        try:
            # generate embedding in executor to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, self._generate_embedding, text
            )
            
            self._cache[text] = embedding
            return embedding
            
        except Exception as e:
            logger.error("Failed to generate embedding", text=text[:100], error=str(e))
            raise
    
    def _generate_embedding(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        # check cache first
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text in self._cache:
                embeddings.append(self._cache[text])
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # generate embeddings for uncached texts
        if uncached_texts:
            try:
                loop = asyncio.get_event_loop()
                new_embeddings = await loop.run_in_executor(
                    None, self._generate_embeddings_batch, uncached_texts
                )
                
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    embeddings[idx] = embedding
                    self._cache[texts[idx]] = embedding
                    
            except Exception as e:
                logger.error("Failed to generate batch embeddings", error=str(e))
                raise
        
        return embeddings
    
    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return [emb.tolist() for emb in embeddings]
    
    async def embed_log_entry(self, log_entry: LogEntry) -> List[float]:
        text_parts = [log_entry.message]
        if log_entry.error:
            text_parts.append(f"Error: {log_entry.error}")
        
        if log_entry.fields:
            field_text = " ".join([f"{k}:{v}" for k, v in log_entry.fields.items()])
            text_parts.append(field_text)
        
        if log_entry.labels:
            label_text = " ".join([f"{k}:{v}" for k, v in log_entry.labels.items()])
            text_parts.append(label_text)
        
        text_parts.extend([
            f"service:{log_entry.service_name}",
            f"level:{log_entry.level.value}"
        ])
        
        full_text = " | ".join(text_parts)
        
        return await self.embed_text(full_text)
    
    async def embed_log_entries(self, log_entries: List[LogEntry]) -> List[List[float]]:
        texts = []
        for log_entry in log_entries:
            text_parts = [log_entry.message]
            
            if log_entry.error:
                text_parts.append(f"Error: {log_entry.error}")
            
            if log_entry.fields:
                field_text = " ".join([f"{k}:{v}" for k, v in log_entry.fields.items()])
                text_parts.append(field_text)
            
            text_parts.extend([
                f"service:{log_entry.service_name}",
                f"level:{log_entry.level.value}"
            ])
            
            texts.append(" | ".join(text_parts))
        
        return await self.embed_texts(texts)
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def find_most_similar(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[List[float]],
        threshold: float = 0.7
    ) -> List[tuple]:
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, candidate)
            if similarity >= threshold:
                similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def clear_cache(self):
        self._cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self._cache),
            "model_name": self.model_name,
            "dimension": self.dimension
        } 