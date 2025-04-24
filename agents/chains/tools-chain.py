from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from typing import Literal

# define myself, or use community tools
@tool
def simple_calculator(operation: Literal["add", "subtract", "multiply", "divide"], x: float, y: float) -> float:
    '''Perform basic arithmetic operations.
    
    Args:
        operation: The arithmetic operation to perform (add, subtract, multiply, divide)
        x: First number
        y: Second number
        
    Returns:
        float: The result of the operation
    '''
    if operation == "add":
        return x + y
    elif operation == "subtract":
        return x - y
    elif operation == "multiply":
        return x * y
    elif operation == "divide":
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y
    else:
        raise ValueError(f"Unsupported operation: {operation}")

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
).bind_tools([simple_calculator])

# Create a chain that will handle the tool calls
chain = llm

# Invoke the chain and get the response
response = chain.invoke("你知道一千万乘二是多少吗？")

# Print the tool calls and their results
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool call: {tool_call['name']}")
        print(f"Arguments: {tool_call['args']}")
        # Execute the tool call using invoke
        result = simple_calculator.invoke(tool_call['args'])
        print(f"Result: {result}")
else:
    print("No tool calls were made")