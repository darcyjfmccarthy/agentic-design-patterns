import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
"""
This pattern is used for breaking down complex tasks into smaller, manageable sub-tasks.
Here, we first extract technical specifications from a text and then transform those specifications into a structured JSON format.
"""

load_dotenv()

llm = ChatDeepSeek(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0,
    model="deepseek-chat"
)

prompt_extract = ChatPromptTemplate.from_template(
    "Extract the technical specifications from the following text:\n\n{input_text}"
)

prompt_transform = ChatPromptTemplate.from_template(
    "Transform the following specifications into a JSON object with 'cpu', 'memory' and 'storage' fields:\n\n{specifications}"
)

extraction_chain = prompt_extract | llm | StrOutputParser()

full_chain = (
    {"specifications":extraction_chain}
    | prompt_transform
    | llm
    | StrOutputParser()
)

input_text = "The new laptop features an Intel i7 processor, 16GB of RAM, and a 512GB SSD."

final_result = full_chain.invoke({"input_text": input_text})
print(final_result)