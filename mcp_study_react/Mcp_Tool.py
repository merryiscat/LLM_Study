import base64
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import os
import json
from dotenv import load_dotenv

load_dotenv()

# MCP 접속 정보 설정
config = {
    "kakaoMapApiKey": os.getenv("KAKAO_MAP_API_KEY")
}
smithery_api_key = os.getenv("SMITHERY_API_KEY")

def get_mcp_url():
    config_b64 = base64.b64encode(json.dumps(config).encode()).decode()
    return f"https://server.smithery.ai/@cgoinglove/mcp-server-kakao-map/mcp?config={config_b64}&api_key={smithery_api_key}"

# 실제 툴 호출 함수
async def recommend_place(query: str):
    url = get_mcp_url()
    async with streamablehttp_client(url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool("kakao_map_place_recommender", {"query": query})
            return result.content
