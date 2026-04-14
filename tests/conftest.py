import logging
import os
from typing import Generator

import pytest
from testcontainers.postgres import PostgresContainer

postgres_container = PostgresContainer("postgres:18-alpine")


@pytest.fixture(scope="session")
def setup_postgres(request: pytest.FixtureRequest):
    postgres_container.start()

    def remove_container():
        postgres_container.stop()

    request.addfinalizer(remove_container)
    os.environ["DATABASE_CONNECTION_STRING"] = postgres_container.get_connection_url(
        driver="asyncpg"
    )
