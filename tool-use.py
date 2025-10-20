import os
import asyncio
import nest_asyncio
import logging
from typing import List
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

llm = ChatDeepSeek(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-chat",
    temperature=0
)

@tool
def search_information(query: str) -> str:
    """
    Provides information on a given topic by simulating a search operation.
    """
    simulated_results = {
        "weather in london": "The current weather in London is cloudy with a temperature of 15°C.",
        "capital of france": "The capital of France is Paris.",
        "population of earth": "The current population of Earth is approximately 8 billion people.",
        "tallest mountain": "The tallest mountain in the world is Mount Everest, standing at 8,848 meters (29,029 feet) above sea level.",
    }
    return simulated_results.get(query.lower(), f"No information found for query: {query}.")

tools = [search_information]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant. Use the provided tools to answer user queries accurately. Note that you MUST use the tool provided regardless of the query.",
)

async def run_agent_with_tool(query: str):
    try:
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        messages = response.get("messages", [])
        if messages:
            last_message = messages[-1]
            print(f"\n🧠 Query: {query}\n💬 Response: {last_message.content}")
        else:
            print("No response messages returned.")
    except Exception as e:
        logging.error(f"Error during agent execution: {e}")


async def main():
    tasks = [
        run_agent_with_tool("What is the weather in London?"),
        run_agent_with_tool("What is the capital of France?"),
        run_agent_with_tool("What's my name?")
    ]
    await asyncio.gather(*tasks)

nest_asyncio.apply()
asyncio.run(main())
