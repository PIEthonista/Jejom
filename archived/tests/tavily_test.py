from langchain_upstage import ChatUpstage
from langchain.tools import tool
from api_keys import TAVILY_API_KEY, UPSTAGE_API_KEY
from tavily_test import TavilyClient
import json
 
 
# # Example dummy function hard coded to return the same weather
# # In production, this could be your backend API or an external API
# @tool
# def get_current_weather(location, unit="fahrenheit"):
#     """
#     return location's weather information
#     """
#     weather_data = {
#         "San Francisco": {"celsius": "15°C", "fahrenheit": "59°F"},
#         "Seoul": {"celsius": "16°C", "fahrenheit": "61°F"},
#         "Paris": {"celsius": "11°C", "fahrenheit": "52°F"},
#     }
#     return f"The weather in {location} is {weather_data[location][unit]}."
 
tavily = TavilyClient(api_key="YOUR_API_KEY")
 
@tool
def tavily_browser_tool(search_query: str):
    """
    browses the internet for relevant information about the search_query and returns it.
    """
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    info = tavily.search(query=search_query, search_depth="advanced")
    return info
 

# available_functions = {"get_current_weather": get_current_weather}
available_functions = {"tavily_browser_tool": tavily_browser_tool}
 
llm = ChatUpstage(api_key=UPSTAGE_API_KEY)
 
# tools = [get_current_weather]
tools = [tavily_browser_tool]
llm_with_tools = llm.bind_tools(tools)
 
 
# Step 1: send the conversation and available functions to the model
messages = [{"role": "user", "content": "How is the weather in San Francisco today?"}]
response = llm_with_tools.invoke(messages)
 
# Step 2: check if the model wanted to call a function
if response.tool_calls:
    tool_call = response.tool_calls[0]
 
    # Step 3: call the function
    function_name = tool_call["name"]
    function_to_call = available_functions[function_name]
    function_args = tool_call["args"]
    # Step 4: send the info for each function call and function response to the model
    function_response = function_to_call.invoke(function_args)
 
    print(function_response)