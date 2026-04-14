from contextlib import asynccontextmanager
from functools import wraps
from typing import AsyncGenerator, ParamSpec, TypeVar, Callable

import asyncpg
import orjson


async def _init_connection(conn: asyncpg.Connection):
    """
    Initializes a database connection with custom JSONB type codec.

    The function sets up a custom codec for the JSONB type to enable
    asynchronous serialization and deserialization of JSON data using
    orjson, a high-performance JSON parser. Meant to be used as an
    "init" function for asyncpg.pool.Pool.

    Args:
        conn (asyncpg.Connection): The database connection instance.

    Returns:
        None
    """
    await conn.set_type_codec(
        "jsonb",
        encoder=lambda x: orjson.dumps(x).decode("utf-8"),
        decoder=lambda x: orjson.loads(x.encode()),
        schema="pg_catalog",
    )


P = ParamSpec("P")
R = TypeVar("R")


def _inject_init(fn: Callable[P, R]) -> Callable[P, R]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        kwargs.setdefault("init", _init_connection)
        return fn(*args, **kwargs)

    return wrapper


create_pool = _inject_init(asyncpg.create_pool)


@asynccontextmanager
async def create_pool_context(dsn: str, **kwargs) -> AsyncGenerator[asyncpg.Pool, None]:
    pool = await create_pool(dsn, **kwargs)
    try:
        yield pool
    finally:
        await pool.close()
