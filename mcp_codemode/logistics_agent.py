import os
import sys
import logging

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from fastmcp import FastMCP, settings
from fastmcp.experimental.transforms.code_mode import CodeMode

settings.host = "0.0.0.0"
settings.debug = False
settings.show_server_banner = False
settings.log_enabled = False

logger = logging.getLogger("mcp_server")
load_dotenv()
mcp = FastMCP(name="Logistics Agent", version="1.0.0", transforms=[CodeMode()])
model = ChatOllama(client_kwargs={"headers":{"Authorization": f"Bearer {os.getenv("OLLAMA_API_KEY")}"}}, model="glm-5:cloud", temperature=0.2, top_k=50, top_p=0.5)
logistics_agent = create_agent(
    model=model,
    system_prompt="""You are a travel logistics expert. You handle practical travel planning:
    - Calculate distances between locations and travel times
    - Estimate costs for transportation, accommodation, and activities
    - Optimize routes and suggest efficient itineraries
    - Consider time zones, weather, and practical constraints
    Always provide short, clear, practical logistics information."""
)

@mcp.tool(name="plan_logistics_agent", description="Plan travel logistics including distances, times, costs, and routes", tags=["logistics"], meta={"version": "1.0", "author": "Umberto Andrisani"})
def plan_logistics_agent(trip_request: str) -> str:
    """
    Plan travel logistics including distances, times, costs, and routes.
    Use this to calculate practical travel information and optimize itineraries.
    
    Args:
        trip_request: Trip details (e.g., "3 days in Paris, budget $1500, from London")
    
    Returns:
        Logistics information: distances, travel times, costs, and route suggestions
    """
    logger.info(f"Planning logistics for trip request: {trip_request}")
    response = logistics_agent.invoke({"messages": [HumanMessage(f"Plan logistics for this trip: {trip_request}")]})
    return response["messages"][-1].content

if __name__ == "__main__":
    # Initialize and run the server
    logger.info("Starting MCP server...")
    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)