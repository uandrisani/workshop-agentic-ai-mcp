import logging
import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp_client")

async def main():
    model = ChatOllama(client_kwargs={"headers":{"Authorization": f"Bearer {os.getenv("OLLAMA_API_KEY")}"}}, model="glm-5:cloud", temperature=0.2, top_k=50, top_p=0.5)

    mcp_client_logistics = MultiServerMCPClient({
        "default": {
            "transport": "stdio",
            "command": "fastmcp",
            "args": ["run", "./mcp_codemode/logistics_agent.py:mcp", "--transport", "stdio"]
        }
    })

    mcp_tools_logistics = await mcp_client_logistics.get_tools()

    mcp_client_recommendations = MultiServerMCPClient({
        "default": {
            "transport": "stdio",
            "command": "fastmcp",
            "args": ["run", "./mcp_codemode/recommendations_agent.py:mcp", "--transport", "stdio"]
        }
    })

    mcp_tools_recommendations = await mcp_client_recommendations.get_tools()
    
    orchestrator = create_agent(
        model=model,
        system_prompt="""You are a travel planning coordinator. 
        When planning trips, use both specialists:
        1. Use plan_logistics_agent to calculate practical details: distances, times, costs, and routes
        2. Use get_recommendations_agent to suggest attractions, restaurants, and activities
        Pay attention that the MCP server works with "Code Mode", so it won't respond with the previous tools in the list but you will have to:
        * search for the tool by calling `search` with query name of the tool you need to execute
        * get the schema of the found tool by calling the `get_schema` tool passing it the found tool as tools
        * call the tool by calling `execute` with the tool name and the request object as arguments
        Always combine both the practical logistics and exciting recommendations in your final response.""",
        tools=mcp_tools_logistics + mcp_tools_recommendations
    )

    # Test 1: City trip planning - uses both agents
    logger.info("Starting test: City trip planning")
    response = await orchestrator.ainvoke({"messages": [HumanMessage("Plan a 3-day trip to Rome. I'm coming from London with a budget of $2000. Calculate travel costs and time, and suggest must-see attractions and restaurants.")]})
    logger.info("Test completed: City trip planning")
    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())