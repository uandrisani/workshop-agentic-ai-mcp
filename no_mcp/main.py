import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp_client")

def main():
    model = ChatOllama(client_kwargs={"headers":{"Authorization": f"Bearer {os.getenv("OLLAMA_API_KEY")}"}}, model="glm-5:cloud", temperature=0.2, top_k=50, top_p=0.5)

    # Create specialized logistics planning agent
    logistics_agent = create_agent(
        model=model,
        system_prompt="""You are a travel logistics expert. You handle practical travel planning:
        - Calculate distances between locations and travel times
        - Estimate costs for transportation, accommodation, and activities
        - Optimize routes and suggest efficient itineraries
        - Consider time zones, weather, and practical constraints
        Always provide short, clear, practical logistics information."""
    )

    # Create specialized recommendations agent
    recommendations_agent = create_agent(
        model=model,
        system_prompt="""You are a travel recommendations specialist. You suggest experiences and activities:
        - Recommend top attractions, landmarks, and must-see places
        - Suggest restaurants, local cuisine, and dining experiences
        - Recommend cultural activities, events, and local experiences
        - Provide insights about local customs, best times to visit, and hidden gems
        Always provide brief, engaging, personalized recommendations."""
    )

    @tool
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

    @tool
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
    
    orchestrator = create_agent(
        model=model,
        system_prompt="""You are a travel planning coordinator. 
        When planning trips, use both specialists:
        1. Use plan_logistics_agent to calculate practical details: distances, times, costs, and routes
        2. Use get_recommendations_agent to suggest attractions, restaurants, and activities
        Always combine both the practical logistics and exciting recommendations in your final response.""",
        tools=[plan_logistics_agent, get_recommendations_agent]
    )

    # Test 1: City trip planning - uses both agents
    logger.info("Starting test: City trip planning")
    response = orchestrator.invoke({"messages": [HumanMessage("Plan a 3-day trip to Rome. I'm coming from London with a budget of $2000. Calculate travel costs and time, and suggest must-see attractions and restaurants.")]})
    logger.info("Test completed: City trip planning")
    print(response["messages"][-1].content)

if __name__ == "__main__":
    main()