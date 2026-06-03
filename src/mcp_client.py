import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

SERVER = StdioServerParameters(
    command="uv", args=["run", "python", "-m", "src.mcp_server"]
)


async def main() -> None:
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()  # handshake
            tools = await session.list_tools()  # discover tools
            print("Tools on server:", [t.name for t in tools.tools])

            result = await session.call_tool("corpus_overview", {})  # invoke
            block = result.content[0]
            text = block.text if isinstance(block, TextContent) else str(block)
            print("\nResult:\n" + text)


if __name__ == "__main__":
    asyncio.run(main())
