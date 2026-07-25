from __future__ import annotations

import asyncio
import json
from typing import Any

from hallucide.core_types.exceptions import RetrievalError
from hallucide.analysis.trust import ensure_system_trust_store


class McpToolClient:
    """Synchronous facade over an MCP streamable-HTTP server.

    The rest of the codebase is synchronous (see GeminiModelProvider,
    MistralModelProvider); this wraps the async `mcp` SDK the same way,
    opening one session per call rather than holding a long-lived loop.

    The `mcp` SDK uses `httpx`, which validates TLS against certifi's
    bundle and rejects connections when an antivirus/proxy performs HTTPS
    inspection; ensure_system_trust_store() switches to the OS trust store
    so those connections succeed (§17.3 souverain, robustesse réseau).
    """

    def __init__(self, url: str) -> None:
        ensure_system_trust_store()
        self.url = url

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            return asyncio.run(self._call_tool_async(name, arguments))
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"MCP call to '{name}' failed: {exc}") from exc

    async def _call_tool_async(self, name: str, arguments: dict[str, Any]) -> Any:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(self.url) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)

                if result.isError:
                    message = "; ".join(
                        block.text for block in result.content if hasattr(block, "text")
                    )
                    raise RetrievalError(f"MCP tool '{name}' returned an error: {message}")

                texts = [block.text for block in result.content if hasattr(block, "text")]
                if not texts:
                    raise RetrievalError(f"MCP tool '{name}' returned no content.")

                payload = "\n".join(texts)
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    return payload
