import os
import sys
import logging

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from fastmcp import FastMCP, settings

settings.host = "0.0.0.0"
settings.debug = False
settings.show_server_banner = False
settings.log_enabled = False

logger = logging.getLogger("mcp_server")
load_dotenv()
mcp = FastMCP(name="Recommendations Agent", version="1.0.0")
model = ChatOllama(client_kwargs={"headers":{"Authorization": f"Bearer {os.getenv("OLLAMA_API_KEY")}"}}, model="glm-5:cloud", temperature=0.2, top_k=50, top_p=0.5)
recommendations_agent = create_agent(
    model=model,
    system_prompt="""You are a travel recommendations specialist. You suggest experiences and activities:
    - Recommend top attractions, landmarks, and must-see places
    - Suggest restaurants, local cuisine, and dining experiences
    - Recommend cultural activities, events, and local experiences
    - Provide insights about local customs, best times to visit, and hidden gems
    Always provide brief, engaging, personalized recommendations."""
)

@mcp.tool(name="get_recommendations_agent", description="Get travel recommendations for attractions, restaurants, and activities", tags=["recommendations"], meta={"version": "1.0", "author": "Umberto Andrisani"})
def get_recommendations_agent(trip_details: str) -> str:
    """
    Get travel recommendations for attractions, restaurants, and activities.
    Use this to suggest what to see, do, and eat at the destination.
    
    Args:
        trip_details: Destination and trip information (e.g., "3 days in Paris, interested in art and food")
    
    Returns:
        Recommendations for attractions, restaurants, activities, and cultural insights
    """
    logger.info(f"Getting recommendations for trip details: {trip_details}")
    response = recommendations_agent.invoke({"messages": [HumanMessage(f"Provide recommendations for: {trip_details}")]})
    return response["messages"][-1].content

if __name__ == "__main__":
    # Initialize and run the server
    logger = logging.getLogger("mcp_server")
    logger.info("Starting MCP server...")
    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)