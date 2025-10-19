import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
"""
Used when an agent needs to decide between multiple workflows, tools or subagents based on an input
Important for anything that needs to triage or classify different types of tasks
"""
load_dotenv()

try:
    llm = ChatDeepSeek(api_key=os.getenv('DEEPSEEK_API_KEY'), model="deepseek-chat", temperature=0)
except Exception as e:
    print(f"Error initializing LLM: {e}")
    raise e

def booking_handler(request:str) -> str:
    print(f"Handling booking request: {request}")
    return f"Booking confirmed for: {request}. Result: Simulated booking action."

def info_handler(request:str) -> str:
    print(f"Handling information request: {request}")
    return f"Information provided for: {request}. Result: Simulated information retrieval."

def unclear_handler(request:str) -> str:
    print(f"Handling unclear request: {request}")
    return f"Could not understand the request: {request}. Result: Simulated clarification needed."

coordination_router_prompt = ChatPromptTemplate.from_messages([
    ("system", """Analyse the user's request and determine which handler to use."
        - If the request is for booking, route to 'booking_handler'.
        - If the request is for information, route to 'info_handler'.
        - If the request is unclear, route to 'unclear_handler'.
     ONLY output one word: 'booker', 'info', or 'unclear'."""),
     ("user", "{request}")
])

if llm:
    coordination_router_chain = coordination_router_prompt | llm | StrOutputParser()

branches = {
    "booker": RunnablePassthrough.assign(output = lambda x: booking_handler(x['request']['request'])),
    "info": RunnablePassthrough.assign(output = lambda x: info_handler(x['request']['request'])),
    "unclear": RunnablePassthrough.assign(output = lambda x: unclear_handler(x['request']['request']))
}

delegation_branch = RunnableBranch(
    (lambda x: x['decision'].strip() == 'booker', branches['booker']),
    (lambda x: x['decision'].strip() == 'info', branches['info']),
    branches["unclear"]
)

coordinator_agent = {
    "decision": coordination_router_chain,
    "request": RunnablePassthrough()
} | delegation_branch | (lambda x: x['output']) 

def main():
    if not llm:
        print("LLM is not initialized. Exiting.")
        return
    print('---Running Coordinator Agent---')
    request_a = "Book me a flight to Hanoi"
    result_a = coordinator_agent.invoke({"request": request_a})
    print(f"Result A: {result_a}\n")

    request_b = "What is 500^2"
    result_b = coordinator_agent.invoke({"request": request_b})
    print(f"Result B: {result_b}\n")

if __name__ == "__main__":
    main()