import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatDeepSeek(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-chat",
    temperature=0.1
)

def run_reflection_loop():
    task_prompt = """
    Your task is to create a Python function named `calculate_factorial`.
    This function should do the following:
    1. Take a single integer input `n`.
    2. Return the factorial of `n` (i.e., n! = n * (n-1) * ... * 1).
    3. Include a clear docstring explaining the function's purpose and parameters.
    4. Handle edge cases: The factorial of 0 is 1
    5. Ensure the function raises a ValueError for negative inputs.
    """

    # Reflection loop
    max_iterations = 3
    current_code = ""
    message_history = [HumanMessage(content=task_prompt)]

    for i in range(max_iterations):
        print(f"--- Iteration {i+1} ---")

        # Generate / refine stage
        # First iteration is generation, then it gets refined each time after that
        if i == 0:
            response = llm.invoke(message_history)
            current_code = response.content
        else:
            message_history.append(HumanMessage(content="Please refine the code using the critiques provided."))
            response = llm.invoke(message_history)
            current_code = response.content

        print("Generated Code:\n", current_code)
        message_history.append(response)

        print("Now reflqecting on the generated code...")

        reflector_prompt = [
            SystemMessage(content="""
                          You are a senior software engineer and an expert in Python.
                          Your role is to perform a meticulous code review.
                          Critically evaluate the provided Python code based on the original task requirements.
                          Look for bugs, style issues, edge cases, and adherence to best practices.
                          If the code is perfect and meets all requirements, respond only with 'APPROVED'.
                          Otherwise, provide a bulleted list of specific critiques and suggestions for improvement.
                          """),
            HumanMessage(content=f"Here is the code to review:\n{current_code}\n\nTask Requirements:\n{task_prompt}")
        ]

        critique_response = llm.invoke(reflector_prompt)
        critique = critique_response.content

        if 'APPROVED' in critique:
            print("Code approved!")
            break

        print("Critique:\n", critique)
        message_history.append(HumanMessage(content=f"Critique of the previous code:\n{critique}"))
    
    print("Final Code after reflection:")
    print(current_code)

if __name__ == "__main__":
    run_reflection_loop()