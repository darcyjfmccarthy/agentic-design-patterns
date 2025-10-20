import os
import asyncio
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
"""
Normal async stuff
"""

load_dotenv()

llm = ChatDeepSeek(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-chat",
    temperature=0.7
)

summarise_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Summarise the following text in a concise manner."),
        ("user", "{text}")
    ])
    | llm
    | StrOutputParser()
)

questions_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Generate three insightful questions based on the following text."),
        ("user", "{text}")
    ])
    | llm
    | StrOutputParser()
)

terms_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Identify 5-10 key terms from the following topic, separated by commas."),
        ("user", "{text}")
    ])
    | llm
    | StrOutputParser()
)

# Defining a block of tasks to run in parallel
map_chain = RunnableParallel(
    {
        "summary": summarise_chain,
        "questions": questions_chain,
        "key_terms": terms_chain,
        "text": RunnablePassthrough()
    }
)

# Define final syntehsis prompt to tie the results together
synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", """Using the following components:
    - Summary {summary}
    - Questions {questions}
    - Key Terms {key_terms}
    Synthesize a comprehensive overview of the text. At the end, provide the key terms as a bulleted list."""),
    ("user", "Original tetxt: {text}")
])

full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()

async def run_parallel_example(text: str) -> None:
    try:
        reponse = await full_parallel_chain.ainvoke({"text": text})
        print("Final Synthesis Output:")
        print(reponse)
    except Exception as e:
        print(f"Error during parallel execution: {e}")

if __name__ == "__main__":
    test_topic = "The history of GPUs"
    asyncio.run(run_parallel_example(test_topic))