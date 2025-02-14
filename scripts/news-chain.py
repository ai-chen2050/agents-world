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
    question_input = "今日全球军事和政治经济有什么重大新闻?"
    print('question_input: ' + question_input + "\n")

    template = """Question: {question}

    网络资料: {search_results}
    
    Answer: 我希望你能充当一名微信公众号运营专员。我将为你提供一个文章的核心词汇，你的任务是根据关键词，生成与之相关的文章。
    你还应该利用你毕生所学的知识和写作技巧的经验，编写出完善的公众号文章, 使文章更具有专业性。
    同时你可以参考网络资料中的信息整理出一篇适合社交传播的网红文章，标题和内容都需要有创意。格式请用 Markdown, 标题请使用 H2 标题，内容请使用正文格式。
    请用 {language} 回答
    """

    prompt = ChatPromptTemplate.from_template(template)

    model = OllamaLLM(model="deepseek-r1")

    chain = prompt | model

    # Get search results
    search_results = search_web(question_input)

    # Pass search results to the chain
    result = chain.invoke(
        {"question": question_input, "search_results": search_results, "language": "中文"}
    )
    print(result)