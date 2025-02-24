from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun

def search_web(query):
    # Initialize the DuckDuckGo search tool
    search_tool = DuckDuckGoSearchRun()

    # Perform the search
    search_results = search_tool.invoke(query)
    
    print('search_results: ' + search_results + "\n")
    # Ensure search_results is a list of dictionaries
    if isinstance(search_results, str):
        try:
            search_results = eval(search_results)
        except SyntaxError:
          return search_results

    # Format the search results
    summary = "\n".join(
        [f"{result['title']}: {result['snippet']}" for result in search_results[:3]]
    )

    return summary

if __name__ == "__main__":
    question_input = "如何做西红柿炒鸡蛋?"
    print('question_input: ' + question_input + "\n")

    template = """Question: {question}

    Search Results: {search_results}

    Answer: Let's think step by step.
    please use {language} to answer the question.
    """

    prompt = ChatPromptTemplate.from_template(template)

    model = OllamaLLM(model="nezahatkorkmaz/deepseek-v3")

    chain = prompt | model

    # Get search results
    search_results = search_web(question_input)

    # Pass search results to the chain
    result = chain.invoke(
        {"question": question_input, "search_results": search_results, "language": "中文"}
    )
    print(result)