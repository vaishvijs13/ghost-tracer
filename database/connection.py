import asyncio
from typing import Optional, AsyncGenerator
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from contextlib import asynccontextmanager

from config import get_settings

logger = structlog.get_logger(__name__)


class Base(DeclarativeBase):
    pass


class DatabaseManager:
    def __init__(self):
        self.settings = get_settings()
        self.engine = None
        self.async_session = None
        
    async def initialize(self):
        try:
            # Create async engine
            database_url = self.settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
            
            self.engine = create_async_engine(
                database_url,
                echo=self.settings.debug,
                pool_size=20,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("Database connection initialized")
            
        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise
    
    async def create_tables(self):
        try:
            from .models import LogEntryModel, TraceSpanModel
            
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database tables created")
            
        except Exception as e:
            logger.error("Failed to create tables", error=str(e))
            raise
    
    async def get_session(self) -> AsyncSession:
        if not self.async_session:
            raise RuntimeError("Database not initialized")
        return self.async_session()
    
    async def close(self):
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection closed")


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


async def get_db_manager() -> DatabaseManager:
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
        await db_manager.initialize()
    return db_manager


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    manager = await get_db_manager()
    session = await manager.get_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close() 