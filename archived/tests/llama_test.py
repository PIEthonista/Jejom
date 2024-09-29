import os
# from llama_index.llms.upstage import Upstage
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
from llama_index.tools.tavily_research.base import TavilyToolSpec
# from llama_index.core.tools.tool_spec import TavilyToolSpec
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import ReActAgent
from tavily_test import TavilyClient
from api_keys import TAVILY_API_KEY, UPSTAGE_API_KEY, GROQ_API_KEY
from system_prompts import HOTEL_SP, CAFE_SP, SPOTS_SP, MASTER_PLANNER_SP

# set api keys
os.environ["UPSTAGE_API_KEY"] = UPSTAGE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
VERBOSE = False

# llm = Upstage()
# response = llm.chat(messages=[
#   ChatMessage(role="system", content="You are a helpful assistant."),
#   ChatMessage(role="user", content="Hi, how are you?")
# ])
# print(response)


# tavily_tool = TavilyToolSpec(
#     api_key=TAVILY_API_KEY,
# )
def tavily_browser_tool(search_query: str) -> str:
    """
    browses the internet for relevant information about the search_query and returns it.
    """
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    info = tavily.search(query=search_query, search_depth="advanced")
    return str(info)

tavily = FunctionTool.from_defaults(fn=tavily_browser_tool)

# tavily_tool_list = tavily_tool.to_tool_list()
# for tool in tavily_tool_list:
#     print(tool.metadata.name)

# llm1 = Upstage(model='solar-1-mini-chat')
llm = Groq(model='llama3-groq-70b-8192-tool-use-preview', api_key=GROQ_API_KEY)



agent1 = ReActAgent.from_tools(tools=[tavily], llm=llm, verbose=VERBOSE)
agent2 = ReActAgent.from_tools(tools=[tavily], llm=llm, verbose=VERBOSE)
agent3 = ReActAgent.from_tools(tools=[tavily], llm=llm, verbose=VERBOSE)
agent4 = ReActAgent.from_tools(tools=[tavily], llm=llm, verbose=VERBOSE)



# response = agent.chat("How is the weather in Alor Setar today?")
# response = agent.chat("What is the latest deal in starbucks malaysia?")

USER_PROMPT = "I wish to visit Jeju island this upcoming 1-3 September. Can you plan a detailed itinerary for me? Your itinerary should include details such as flight, transport, acccomodation, visiting spots, schedule, and entrance fees."

try:
    hotel_response = agent1.chat(HOTEL_SP + USER_PROMPT)
    print("===== HOTEL")
    print(hotel_response)
except:
    pass

try:
    cafe_response = agent2.chat(CAFE_SP + USER_PROMPT)
    print("===== CAFE")
    print(cafe_response)
except:
    pass

try:
    spot_response = agent3.chat(SPOTS_SP + USER_PROMPT)
    print("===== SPOT")
    print(spot_response)
except:
    pass


# plan_response = agent4.chat(MASTER_PLANNER_SP + USER_PROMPT + \
#     "Available information about accodomation are as follows: " + hotel_response + \
#     "Available information about cafes and eateries are as follows: " + cafe_response + \
#     "Available information about tourist spots are as follows: " + spot_response
#     )

# print(str(plan_response))