from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg


@asynccontextmanager
async def get_connection(
    pool: asyncpg.Pool,
) -> AsyncIterator[asyncpg.Connection]:
    async with pool.acquire() as conn:
        yield conn
