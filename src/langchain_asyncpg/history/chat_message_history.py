import re
import uuid
import orjson
from typing import Sequence, List

from langchain_core.chat_history import BaseChatMessageHistory
from asyncpg import Connection
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

GET_MESSAGES = """
               SELECT message
               FROM $1
               WHERE session_id = $2
               ORDER BY id; \
               """

DELETE_BY_SESSION_ID = """
                       DELETE
                       FROM $1
                       WHERE session_id = $2; \
                       """

DELETE_BY_TABLE = """
                  DROP TABLE IF EXISTS $1; \
                  """

INSERT_MESSAGE = """
                 INSERT INTO $1 (session_id, message)
                 VALUES ($2, $3::jsonb); \
                 """


class AsyncPostgresChatMessageHistory(BaseChatMessageHistory):
    _connection: Connection
    _table_name: str
    _session_id: str

    def __init__(
        self,
        table_name: str,
        session_id: str | None = None,
        /,
        *,
        connection: Connection,
    ) -> None:
        self._table_name = table_name
        self._connection = connection

        try:
            uuid.UUID(session_id)
        except ValueError:
            raise ValueError(
                f"Invalid session id. Session id must be a valid UUID. Got {session_id}"
            )

        self._session_id = session_id

        if not re.match(r"^\w+$", table_name):
            raise ValueError(
                "Invalid table name. Table name must contain only alphanumeric "
                "characters and underscores."
            )
        self._table_name = table_name

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        values = [
            (self._session_id, orjson.dumps(message_to_dict(message)))
            for message in messages
        ]

        async with self._connection.transaction():
            await self._connection.executemany(INSERT_MESSAGE, values)

    async def aget_messages(self) -> List[BaseMessage]:
        """Retrieve messages from the chat message history."""

        async with self._connection.transaction():
            items = await self._connection.fetch(
                GET_MESSAGES, self._table_name, self._session_id
            )

        messages = messages_from_dict(items)
        return messages

    async def aclear(self) -> None:
        """Clear the chat message history for the GIVEN session."""
        async with self._connection.transaction():
            await self._connection.execute(
                DELETE_BY_SESSION_ID, self._table_name, self._session_id
            )

    def clear(self) -> None:
        raise NotImplementedError(
            "AsyncPostgresChatMessageHistory does not support sync clear, call aclear instead."
        )
