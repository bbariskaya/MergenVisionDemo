from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

engine: AsyncEngine | None = None
AsyncSessionLocal: async_sessionmaker[AsyncSession] | None = None


def configure_engine() -> None:
    global engine, AsyncSessionLocal
    if engine is None:
        engine = create_async_engine(settings.database_url, echo=False, future=True)
        AsyncSessionLocal = async_sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )


async def dispose_engine() -> None:
    global engine
    if engine is not None:
        await engine.dispose()
        engine = None


async def get_db() -> AsyncSession:
    if AsyncSessionLocal is None:
        raise RuntimeError("Database engine not configured")
    async with AsyncSessionLocal() as session:
        yield session
