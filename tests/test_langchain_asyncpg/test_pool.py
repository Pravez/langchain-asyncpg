from langchain_asyncpg.pool import create_pool


async def test__can_create_pool(setup_postgres: str):
    async with create_pool(setup_postgres) as pool:
        result = await pool.fetchval("SELECT 1;")
        assert result == 1
