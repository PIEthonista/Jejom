import os
import json
import math
import random
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from datetime import datetime, timedelta

# https://github.com/mangiucugna/json_repair
from json_repair import repair_json  # LLM JSON output fixing if necessary

from tavily import TavilyClient
from llama_index.core import PromptTemplate, Settings
# from llama_index.llms.openai import OpenAI
# from llama_index.llms.nvidia import NVIDIA
from llama_index.llms.groq import Groq
# from llama_index.llms.upstage import Upstage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.upstage import UpstageEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.agent.openai import OpenAIAgent

load_dotenv()
VERBOSE = False
CLUSTER_DESTINATION_DISTANCE = 15  # km



def tavily_browser_tool(input: str) -> str:
    tavily = TavilyClient()
    info = tavily.search(query=input, search_depth="advanced")
    print(">>> tavily", str(info)) if VERBOSE else None
    return str(info)

def tavily_browser_tool_longlat(input: str) -> str:
    tavily = TavilyClient()
    info = tavily.search(query=f"Geographical longtitude and latitude of {input}", search_depth="advanced")
    print(">>> tavily", str(info)) if VERBOSE else None
    return str(info)

def tavily_browser_tool_address(input: str) -> str:
    tavily = TavilyClient()
    info = tavily.search(query=f"Location and Address of {input}", search_depth="advanced")
    print(">>> tavily", str(info)) if VERBOSE else None
    return str(info)

def tavily_browser_tool_rating(input: str) -> str:
    tavily = TavilyClient()
    info = tavily.search(query=f"Rating of {input}", search_depth="advanced")
    print(">>> tavily", str(info)) if VERBOSE else None
    return str(info)

def tavily_browser_tool_provider(input: str) -> str:
    tavily = TavilyClient()
    info = tavily.search(query=f"Provider of {input}", search_depth="advanced")
    print(">>> tavily", str(info)) if VERBOSE else None
    return str(info)


accomodation_detail_top_orchestrator_agent = OpenAIAgent.from_tools(
    system_prompt="""
        You are an assistant model that returns the details of a tourist accomodation given its name.
        Your response MUST be in the form of a JSON object, and MUST contain only the following attributes for the JSON object: 'Name', 'Address', 'Price', 'Rating', 'Latitude', 'Longtitude', 'Provider'.
        Your will be given tools that can browse the web for information about specific aspects. You must use them should you require additional information about those aspects.
        If the web browser tool does not return relevant information about the aspects that you are looking for, just set None for that aspect.
        Do not make up any information.

        Example Input:
        "Lotte Hotel Jeju"
        
        Corresponding Output:
        {
            "Name": "Lotte Hotel Jeju",
            "Address": "35, Jungmungwangwang-ro 72beon-gil, Seogwipo-si, Jeju-do, South Korea",
            "Price": "KRW 300,000 - 600,000 per night",
            "Rating": "4.5 out of 5",
            "Latitude": "33.247204",
            "Longitude": "126.412185",
            "Provider": "Lotte Hotels & Resorts"
        }
    """,
    llm=Settings.llm,
    verbose=VERBOSE,
    tools=[
        QueryEngineTool(
            query_engine=OpenAIAgent.from_tools(
                tools=[
                    FunctionTool.from_defaults(
                        fn=tavily_browser_tool,
                        tool_metadata=ToolMetadata(
                            name='general web browser tool',
                            description=f"Useful for browsing general information."
                        )
                    )
                ],
                llm=Settings.llm,
                verbose=VERBOSE,
                system_prompt=f"""
                    You are an assistant that browses the web for general information.
                    You must use the web browser tool provided to you to search for information required, and answer based on the findings returned.
                    Do not make up any information or reply on prior knowledge.
                    If no related information is found, just respond with the word 'None'.
                """
            ),
            metadata=ToolMetadata(
                name='general web browser tool',
                description=f"Useful for browsing general information."
            )
        ),
        QueryEngineTool(
            query_engine=OpenAIAgent.from_tools(
                tools=[
                    FunctionTool.from_defaults(
                        fn=tavily_browser_tool_longlat,
                        tool_metadata=ToolMetadata(
                            name='location longtitude and latitude web browser tool',
                            description=f"Useful for browsing geographic longtitue and latitude of a location."
                        )
                    )
                ],
                llm=Settings.llm,
                verbose=VERBOSE,
                system_prompt=f"""
                    You are an assistant that browses the web for longtitude and latitude of a location.
                    You must use the web browser tool provided to you to search for information required, and answer based on the findings returned.
                    Do not make up any information or reply on prior knowledge.
                    If no related information is found, just respond with the word 'None'.
                """
            ),
            metadata=ToolMetadata(
                name='location longtitude and latitude web browser tool',
                description=f"Useful for browsing geographic longtitue and latitude of a location."
            )
        ),
        QueryEngineTool(
            query_engine=OpenAIAgent.from_tools(
                tools=[
                    FunctionTool.from_defaults(
                        fn=tavily_browser_tool_address,
                        tool_metadata=ToolMetadata(
                            name='location address web browser tool',
                            description=f"Useful for browsing address of location."
                        )
                    )
                ],
                llm=Settings.llm,
                verbose=VERBOSE,
                system_prompt=f"""
                    You are an assistant that browses the web for the address of a location.
                    You must use the web browser tool provided to you to search for information required, and answer based on the findings returned.
                    Do not make up any information or reply on prior knowledge.
                    If no related information is found, just respond with the word 'None'.
                """
            ),
            metadata=ToolMetadata(
                name='location address web browser tool',
                description=f"Useful for browsing address of location."
            )
        ),
        QueryEngineTool(
            query_engine=OpenAIAgent.from_tools(
                tools=[
                    FunctionTool.from_defaults(
                        fn=tavily_browser_tool_rating,
                        tool_metadata=ToolMetadata(
                            name='location rating web browser tool',
                            description=f"Useful for browsing rating of location."
                        )
                    )
                ],
                llm=Settings.llm,
                verbose=VERBOSE,
                system_prompt=f"""
                    You are an assistant that browses the web for the rating of a location.
                    You must use the web browser tool provided to you to search for information required, and answer based on the findings returned.
                    Do not make up any information or reply on prior knowledge.
                    If no related information is found, just respond with the word 'None'.
                """
            ),
            metadata=ToolMetadata(
                name='location rating web browser tool',
                description=f"Useful for browsing rating of location."
            )
        ),
        QueryEngineTool(
            query_engine=OpenAIAgent.from_tools(
                tools=[
                    FunctionTool.from_defaults(
                        fn=tavily_browser_tool_provider,
                        tool_metadata=ToolMetadata(
                            name='location provider web browser tool',
                            description=f"Useful for browsing provider of location."
                        )
                    )
                ],
                llm=Settings.llm,
                verbose=VERBOSE,
                system_prompt=f"""
                    You are an assistant that browses the web for the provider of a location.
                    You must use the web browser tool provided to you to search for information required, and answer based on the findings returned.
                    Do not make up any information or reply on prior knowledge.
                    If no related information is found, just respond with the word 'None'.
                """
            ),
            metadata=ToolMetadata(
                name='location provider web browser tool',
                description=f"Useful for browsing provider of location."
            )
        )
    ]
)

destination_detail_top_orchestrator_agent = OpenAIAgent.from_tools(
    system_prompt="""
        You are an assistant model that returns the details of a tourist destination given its name.
        Your response MUST be in the form of a JSON object, and MUST contain only the following attributes for the JSON object: 'Name', 'Description', 'Price', 'Address', 'Latitude', 'Longtitude'.
        Your will be given tools that can browse the web for information about specific aspects. You must use them should you require additional information about those aspects.
        If the web browser tool does not return relevant information about the aspects that you are looking for, just set None for that aspect.
        Do not make up any information.

        Example Input:
        'Seongsan Ilchulbong'
        
        Corresponding Output:
        {
            "Name": "Seongsan Ilchulbong",
            "Description": "Seongsan Ilchulbong, also known as 'Sunrise Peak,' is a UNESCO World Heritage site formed by a volcanic eruption over 5,000 years ago. It offers a breathtaking view of the sunrise from its peak, making it a popular destination for early morning hikers.",
            "Price": "KRW 2,000",
            "Address": "Seongsan-eup, Seogwipo-si, Jeju-do, South Korea",
            "Latitude": "33.461111",
            "Longitude": "126.940556"
        }
    """,
    llm=Settings.llm,
    verbose=VERBOSE,
    tools=[
        QueryEngineTool(
            query_engine=OpenAIAgent.from_tools(
                tools=[
                    FunctionTool.from_defaults(
                        fn=tavily_browser_tool,
                        tool_metadata=ToolMetadata(
                            name='general web browser tool',
                            description=f"Useful for browsing general information."
                        )
                    )
                ],
                llm=Settings.llm,
                verbose=VERBOSE,
                system_prompt=f"""
                    You are an assistant that browses the web for general information.
                    You must use the web browser tool provided to you to search for information required, and answer based on the findings returned.
                    Do not make up any information or reply on prior knowledge.
                    If no related information is found, just respond with the word 'None'.
                """
            ),
            metadata=ToolMetadata(
                name='general web browser tool',
                description=f"Useful for browsing general information."
            )
        ),
        QueryEngineTool(
            query_engine=OpenAIAgent.from_tools(
                tools=[
                    FunctionTool.from_defaults(
                        fn=tavily_browser_tool_longlat,
                        tool_metadata=ToolMetadata(
                            name='location longtitude and latitude web browser tool',
                            description=f"Useful for browsing geographic longtitue and latitude of a location."
                        )
                    )
                ],
                llm=Settings.llm,
                verbose=VERBOSE,
                system_prompt=f"""
                    You are an assistant that browses the web for longtitude and latitude of a location.
                    You must use the web browser tool provided to you to search for information required, and answer based on the findings returned.
                    Do not make up any information or reply on prior knowledge.
                    If no related information is found, just respond with the word 'None'.
                """
            ),
            metadata=ToolMetadata(
                name='location longtitude and latitude web browser tool',
                description=f"Useful for browsing geographic longtitue and latitude of a location."
            )
        ),
        QueryEngineTool(
            query_engine=OpenAIAgent.from_tools(
                tools=[
                    FunctionTool.from_defaults(
                        fn=tavily_browser_tool_address,
                        tool_metadata=ToolMetadata(
                            name='location address web browser tool',
                            description=f"Useful for browsing address of location."
                        )
                    )
                ],
                llm=Settings.llm,
                verbose=VERBOSE,
                system_prompt=f"""
                    You are an assistant that browses the web for the address of a location.
                    You must use the web browser tool provided to you to search for information required, and answer based on the findings returned.
                    Do not make up any information or reply on prior knowledge.
                    If no related information is found, just respond with the word 'None'.
                """
            ),
            metadata=ToolMetadata(
                name='location address web browser tool',
                description=f"Useful for browsing address of location."
            )
        )
    ]
)

list_destination_prompt = PromptTemplate(
    """
    You are an assistant that recommends a required number of tourist locations in Jeju Island to the end user.
    You will be given the preferences and characteristics of the end user.
    Your response MUST ONLY be a simple list (without any brackets), with items being separated by commas.
    
    Example Input:
    End User Preferences and Characteristics: 'Nature Lover, Outgoing, Female, 21, ENFJ'
    Number of Tourist Locations Required: 4
    
    Corresponding Output:
    Hallasan National Park, Seongsan Ilchulbong, Olle Trails, Cheonjeyeon Waterfall
    
    Query:
    End User Preferences and Characteristics: {end_user_specs}
    Number of Tourist Locations Required: {num_spots}
    """
)

list_accomodation_prompt = PromptTemplate(
    """
    You are an assistant that recommends a required number of tourist accomodations in Jeju Island to the end user.
    You will be given the preferences and/or characteristics and/or requirements of the end user.
    Your response MUST ONLY be a simple list (without any brackets), with items being separated by commas.
    
    Example Input:
    End User Preferences/Characteristics/Requirements: 'Nature Lover, Female, 21, ENFJ, Loves a view by the ocean, North part of Jeju island.'
    Number of Tourist Accomodations Required: 4
    
    Corresponding Output:
    Ramada Plaza Jeju Ocean Front, Ocean Suites Jeju Hotel, Hotel RegentMarine The Blue, Jeju Oriental Hotel & Casino
    
    Query:
    End User Preferences/Characteristics/Requirements: {end_user_specs}
    Number of Tourist Accomodations Required: {num_accomodations}
    """
)

same_location_check_prompt = PromptTemplate(
    """
    You are a classifation model. Given the two lcoations in Jeju Island below, determine if they both refer to the same location.
    If the two locations refer to the same place, respond with "yes".
    If the two locations do not refer to the same place, respond with "no".
    
    Query:
    Location 1: {loc_1}
    Location 2: {loc_2}
    """
)

json_check_prompt = PromptTemplate(
    """
    You are a classification model. Given the query below, determine if it follows the exact format of a JSON Object.
    If the query follows the format of a JSON Object, respond with "yes". 
    If it does not follow the format of a JSON Object, respond with "no".
    Query: {query_str}
    """
)

isDestination_check_prompt = PromptTemplate(
    """
    You are a classification model. Given the query below, determine if it contains information about any destination.
    If the query does contain information about a certain destination, respond with "yes". 
    If it does not contain information about any destination, respond with "no".
    Query: {query_str}
    """
)

isDuration_check_prompt = PromptTemplate(
    """
    You are a classification model. Given the query below, determine if it contains information about datetime or duration.
    If the query does contain information about any datetime or duration, respond with "yes". 
    If it does not contain information about any datetime or duration, respond with "no".
    Query: {query_str}
    """
)

isBudget_check_prompt = PromptTemplate(
    """
    You are a classification model. Given the query below, determine if it contains information about budget.
    If the query does contain information about budget, respond with "yes". 
    If it does not contain information about budget, respond with "no".
    Query: {query_str}
    """
)

isInterest_check_prompt = PromptTemplate(
    """
    You are a classification model. Given the query below, determine if it contains information about hobby or interests.
    If the query does contain information about hobby or interests, respond with "yes". 
    If it does not contain information about hobby or interests, respond with "no".
    Query: {query_str}
    """
)

get_starting_date_prompt = PromptTemplate(
    """
    You are an assistant that extracts the starting date in the format of YYYY-MM-DD.
    The date must strictly be all numbers. 
    You will be given a piece of context where you will extract the starting date from.
    
    Example Input:
    Context: I wish to visit Jeju Island from 5 - 9 April 2024. Can you plan me a trip? I saved up around MYR 5000 for this trip. City Lover, Male, 30, ENTJ, Loves beachball
    
    Corresponding Output:
    2024-04-05
    
    Context: {query_str}
    """
)

get_ending_date_prompt = PromptTemplate(
    """
    You are an assistant that extracending date in the format of YYYY-MM-DD.
    The date must strictly be all numbers. 
    You will be given a piece of context where you will extract the ending date from.
    
    Example Input:
    Context: I wish to visit Jeju Island from 5 - 9 April 2024. Can you plan me a trip? I saved up around MYR 5000 for this trip. City Lover, Male, 30, ENTJ, Loves beachball
    
    Corresponding Output:
    2024-04-09
    
    Context: {query_str}
    """
)

generate_trip_title_prompt = PromptTemplate(
    """
    You are a title generating assistant. 
    Given the details about the trip, including the tourist spots to be visited, generate a suitable short title about the trip
    Details about the trip: {details}
    """
)

generate_trip_description_prompt = PromptTemplate(
    """
    You are a description generating assistant. 
    Given the name and details about the trip, including the tourist spots to be visited, generate a suitable short description about the trip
    Name of the trip: {name}
    Details about the trip: {details}
    """
)

def get_most_similar_location_from_cache(embed_model: OpenAIEmbedding, cache_dict, location_name, top_k=1):
    cache_dict_names = list(cache_dict.keys())
    
    location_name_embed = embed_model.get_text_embedding(location_name)
    cache_dict_names_embed = embed_model.get_text_embedding_batch(cache_dict_names)
    
    location_name_embed_np = np.array(location_name_embed)
    cache_dict_names_embed_np = np.array(cache_dict_names_embed)
    assert location_name_embed_np.shape[0] == cache_dict_names_embed_np.shape[1]
    
    # cosine sim
    location_name_embed_norm = np.linalg.norm(location_name_embed_np)
    cache_dict_names_embed_norm = np.linalg.norm(cache_dict_names_embed_np, axis=1, keepdims=True)
    similarities = np.dot(cache_dict_names_embed_np, location_name_embed_np) / (cache_dict_names_embed_norm.flatten() * location_name_embed_norm)
    
    top_k_indices = np.argsort(similarities)[-int(top_k):][::-1]
    top_k_indices = top_k_indices.tolist()
    
    most_sim_location_from_cache = []
    for idx in top_k_indices:
        most_sim_location_from_cache.append(cache_dict_names[idx])
    return most_sim_location_from_cache



class Pipeline():
    def __init__(self, USE_CACHE=True, accomodation_cache_file="./cache/accomodations.json", destination_cache_file = "./cache/destinations.json"):
        self.USE_CACHE = USE_CACHE
        self.accomodation_cache_file = accomodation_cache_file
        self.destination_cache_file = destination_cache_file

    def get_accomodations_json(self, query: str, num_accomodations: int = 5, max_json_try: int = 3, check_match_from_cache_top_k: int = 2):
        # end_user_specs = 'City Lover, Male, 30, ENTJ, Loves beachball'
        # end_user_specs = "Loves a chill life, doesnt like crowded places, loves coffee, artistic, female, 25, ENTP"
        # num_accomodations = 5
        # max_json_try = 3
        # check_match_from_cache_top_k = 2

        accomodation_cache_available = os.path.exists(self.accomodation_cache_file)
        if accomodation_cache_available:
            with open(self.accomodation_cache_file, 'r') as f:
                accomodation_cache = json.load(f)
        else:
            accomodation_cache = dict()
            
        formatted_accomodation_prompt = list_accomodation_prompt.format(end_user_specs=query, num_accomodations=str(num_accomodations))
        accomodations_string = Settings.llm.complete(formatted_accomodation_prompt)
        list_of_accomodations = str(accomodations_string).split(',')

        locs_to_be_cached = []
        json_response_str = "["
        for accomodation in tqdm(list_of_accomodations):
            
            # check if accomodation data is available in cache
            found_location = False
            if self.USE_CACHE and accomodation_cache_available:
                # get most similar locations from cache
                most_sim_loc_from_cache_list = get_most_similar_location_from_cache(Settings.embed_model, accomodation_cache, accomodation, top_k=check_match_from_cache_top_k)
                for loc in most_sim_loc_from_cache_list:
                    # check if most similar locations from cache refer to the target location
                    is_same_loc = Settings.llm.complete(same_location_check_prompt.format(loc_1=str(accomodation), loc_2=str(loc)))
                    if str(is_same_loc).lower().strip() == "yes":
                        accomodation_json_str = json.dumps(accomodation_cache.get(str(loc)))
                        json_response_str += str(accomodation_json_str)
                        json_response_str += ","
                        found_location = True
                        print(f"Found location {accomodation} in cache ...")
                        break
                if found_location:
                    continue
            
            output_is_json = False
            counter = 0
            try:
                while (not output_is_json) and (counter < max_json_try):
                    accomodation_json_str = accomodation_detail_top_orchestrator_agent.query(str(accomodation))
                    is_json = Settings.llm.complete(json_check_prompt.format(query_str=str(accomodation_json_str)))
                    if str(is_json).lower().strip() == 'yes':
                        output_is_json = True
                        try:
                            accomodation_json_str = repair_json(accomodation_json_str)
                        except OSError:
                            accomodation_json_str = ""
                        if str(accomodation_json_str).strip() != "":
                            json_response_str += str(accomodation_json_str)
                            json_response_str += ","
                            # add new loc to cache
                            if self.USE_CACHE and accomodation_cache_available and (not found_location):
                                locs_to_be_cached.append(json.loads(accomodation_json_str))
                            break
                        else:
                            print(f"Error getting details in JSON format for accomodation: {accomodation}. JSON Repair failed.")
                    else:
                        counter += 1
                if counter == max_json_try:
                    print(f"Error getting details in JSON format for accomodation: {accomodation}. Max tries reached.")
            except:
                print(f"Error getting details for accomodation: {accomodation}")
        if str(json_response_str).endswith(','):
            json_response_str = str(json_response_str)[:-1]
        json_response_str += "]"

        # print()
        # print()
        # print(json_response_str)

        json_response = None
        try:
            json_response = json.loads(json_response_str)
        except:
            try:
                print(f"Repairing JSON string ...")
                good_json_response_str = repair_json(json_response_str, skip_json_loads=True)
            except OSError:
                print(f"Repair failed.")
                good_json_response_str = ""  # real bad json string / os error
            if not (str(good_json_response_str).strip() == ""):
                json_response = json.loads(good_json_response_str)

        # create/update cache
        cache_updated = False
        for loc_dict in locs_to_be_cached:
            try:
                loc_name = loc_dict.get("Name", None)
                if loc_name is not None:
                    accomodation_cache[str(loc_name)] = loc_dict
                    print(f"Added {loc_name} to cache ...")
                    cache_updated = True
            except:
                print(f"Failed to add {loc_name} to cache.")
        if cache_updated:
            if accomodation_cache_available:
                os.remove(self.accomodation_cache_file)
            with open(self.accomodation_cache_file, 'w') as json_file:
                json.dump(accomodation_cache, json_file, indent=4)
        
        # print("============")
        # print(json_response)
        return json_response


    def get_destinations_json(self, query: str, num_spots: int = 5, max_json_try: int = 3, check_match_from_cache_top_k: int = 2):
        destination_cache_available = os.path.exists(self.destination_cache_file)
        if destination_cache_available:
            with open(self.destination_cache_file, 'r') as f:
                destination_cache = json.load(f)
        else:
            destination_cache = dict()
    
        formatted_destination_prompt = list_destination_prompt.format(end_user_specs=query, num_spots=str(num_spots))
        destinations_string = Settings.llm.complete(formatted_destination_prompt)
        list_of_destinations = str(destinations_string).split(',')

        locs_to_be_cached = []
        json_response_str = "["
        for destination in tqdm(list_of_destinations):
            
            # check if destination data is available in cache
            found_location = False
            if self.USE_CACHE and destination_cache_available:
                # get most similar locations from cache
                most_sim_loc_from_cache_list = get_most_similar_location_from_cache(Settings.embed_model, destination_cache, destination, top_k=check_match_from_cache_top_k)
                for loc in most_sim_loc_from_cache_list:
                    # check if most similar locations from cache refer to the target location
                    is_same_loc = Settings.llm.complete(same_location_check_prompt.format(loc_1=str(destination), loc_2=str(loc)))
                    if str(is_same_loc).lower().strip() == "yes":
                        destination_json_str = json.dumps(destination_cache.get(str(loc)))
                        json_response_str += str(destination_json_str)
                        json_response_str += ","
                        found_location = True
                        print(f"Found location {destination} in cache ...")
                        break
                if found_location:
                    continue
            
            output_is_json = False
            counter = 0
            try:
                while (not output_is_json) and (counter < max_json_try):
                    destination_json_str = destination_detail_top_orchestrator_agent.query(str(destination))
                    is_json = Settings.llm.complete(json_check_prompt.format(query_str=str(destination_json_str)))
                    if str(is_json).lower().strip() == 'yes':
                        output_is_json = True
                        try:
                            destination_json_str = repair_json(destination_json_str)
                        except OSError:
                            destination_json_str = ""
                        if str(destination_json_str).strip() != "":
                            json_response_str += str(destination_json_str)
                            json_response_str += ","
                            # add new loc to cache
                            if self.USE_CACHE and destination_cache_available and (not found_location):
                                locs_to_be_cached.append(json.loads(destination_json_str))
                            break
                        else:
                            print(f"Error getting details in JSON format for destination: {destination}. JSON Repair failed.")
                    else:
                        counter += 1
                if counter == max_json_try:
                    print(f"Error getting details in JSON format for destination: {destination}. Max tries reached.")
            except:
                print(f"Error getting details for destination: {destination}")
        if str(json_response_str).endswith(','):
            json_response_str = str(json_response_str)[:-1]
        json_response_str += "]"

        # print()
        # print()
        # print(json_response_str)

        json_response = None
        try:
            json_response = json.loads(json_response_str)
        except:
            try:
                print(f"Repairing JSON string ...")
                good_json_response_str = repair_json(json_response_str, skip_json_loads=True)
            except OSError:
                print(f"Repair failed.")
                good_json_response_str = ""  # real bad json string / os error
            if not (str(good_json_response_str).strip() == ""):
                json_response = json.loads(good_json_response_str)

        # create/update cache
        cache_updated = False
        for loc_dict in locs_to_be_cached:
            try:
                loc_name = loc_dict.get("Name", None)
                if loc_name is not None:
                    destination_cache[str(loc_name)] = loc_dict
                    print(f"Added {loc_name} to cache ...")
                    cache_updated = True
            except:
                print(f"Failed to add {loc_name} to cache.")
        if cache_updated:
            if destination_cache_available:
                os.remove(self.destination_cache_file)
            with open(self.destination_cache_file, 'w') as json_file:
                json.dump(destination_cache, json_file, indent=4)

        # print("============")
        # print(json_response)
        return json_response


    def check_query_detail(self, query: str):
        isDestination = True if str(Settings.llm.complete(isDestination_check_prompt.format(query_str=query))).lower().strip() == "yes" else False
        isDuration    = True if str(Settings.llm.complete(isDuration_check_prompt.format(query_str=query))).lower().strip() == "yes" else False
        isBudget      = True if str(Settings.llm.complete(isBudget_check_prompt.format(query_str=query))).lower().strip() == "yes" else False
        isInterest    = True if str(Settings.llm.complete(isInterest_check_prompt.format(query_str=query))).lower().strip() == "yes" else False
        
        return {
            "isDestination": isDestination,
            "isDuration": isDuration,
            "isBudget": isBudget,
            "isInterest": isInterest
        }


    def generate_trip(self, end_user_specs: str, end_user_final_query: str, max_json_try: int = 3, check_match_from_cache_top_k: int = 2):
        # end_user_specs = "Loves a chill life, doesnt like crowded places, loves coffee, artistic, female, 25, ENTP"
        # end_user_specs = "Nature Lover, Photography, Solo-Traveller"
        # end_user_final_query = "Can you plan me a trip to Jeju Island from 18 January to 20 January in 2025? My budget is around MYR 3000. " + end_user_specs
        # end_user_final_query = "Can you plan me a 3-day trip to Jeju Island starting from 18 January 2025? My budget is around MYR 3000. " + end_user_specs

        starting_date = Settings.llm.complete(get_starting_date_prompt.format(query_str=end_user_final_query))
        ending_date = Settings.llm.complete(get_ending_date_prompt.format(query_str=end_user_final_query))

        print(starting_date)
        print(ending_date)

        def generate_dates(start_date, end_date):
            # Convert strings to datetime objects
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Generate list of dates
            dates = []
            current_date = start
            while current_date <= end:
                dates.append(current_date.strftime('%Y-%m-%d'))
                current_date += timedelta(days=1)
            
            return dates

        dates = generate_dates(str(starting_date), str(ending_date))
        print(dates)


        DESTINATIONS_PER_DAY = 2
        TOTAL_ACCOMODATIONS = 5
        TOTAL_DESTINATIONS = max(10, len(dates)*DESTINATIONS_PER_DAY)
        
        destinations_list_of_dicts = self.get_destinations_json(end_user_final_query, 
                                                                num_spots=TOTAL_DESTINATIONS, 
                                                                max_json_try=max_json_try, 
                                                                check_match_from_cache_top_k=check_match_from_cache_top_k)
        accomodations_list_of_dicts = self.get_accomodations_json(end_user_final_query, 
                                                                  num_accomodations=TOTAL_ACCOMODATIONS, 
                                                                  max_json_try=max_json_try, 
                                                                  check_match_from_cache_top_k=check_match_from_cache_top_k)

        def get_long_lat(long: str, lat: str):
            try:
                long_ft = float(long)
                lat_ft = float(lat)
                # return (long_ft, lat_ft)
                return (lat_ft, long_ft)
            except:
                return None

        # filter out accomodations and destinations without valid long lat info
        valid_accomodations = []
        invalid_accomodations = []
        accomodations_long_lat = []
        for accomodation in accomodations_list_of_dicts:
            long_lat_ft = get_long_lat(long=accomodation.get("Longitude", None), lat=accomodation.get("Latitude", None))
            if long_lat_ft is not None:
                valid_accomodations.append(accomodation)
                accomodations_long_lat.append(long_lat_ft)
            else:
                invalid_accomodations.append(accomodation)

        valid_destinations = []
        invalid_destinations = []
        destinations_long_lat = []
        for destination in destinations_list_of_dicts:
            long_lat_ft = get_long_lat(long=destination.get("Longitude", None), lat=destination.get("Latitude", None))
            if long_lat_ft is not None:
                valid_destinations.append(destination)
                destinations_long_lat.append(long_lat_ft)
            else:
                invalid_destinations.append(destination)

        # custom distance function
        def haversine_distance(coord1, coord2):
            return great_circle(coord1, coord2).km


        # Perform DBSCAN clustering, eps in km
        destinations_long_lat_np = np.array(destinations_long_lat)
        dbscan = DBSCAN(eps=CLUSTER_DESTINATION_DISTANCE, min_samples=1, metric=haversine_distance)
        clusters = dbscan.fit_predict(destinations_long_lat_np)
        clusters = clusters.tolist()
        assert len(clusters) == len(valid_destinations)


        def order_indices_by_values(lst):
            indices = list(range(len(lst)))
            sorted_indices = sorted(indices, key=lambda x: lst[x])
            return sorted_indices

        clusters_indices = order_indices_by_values(clusters)
        valid_destinations = [valid_destinations[i] for i in clusters_indices]



        if len(dates) > (len(valid_destinations)+len(invalid_destinations)):
            dates = dates[:(len(valid_destinations)+len(invalid_destinations)) - 1]
            ending_date = dates[-1]
            DESTINATIONS_PER_DAY = 1
        elif len(dates) == (len(valid_destinations)+len(invalid_destinations)):
            DESTINATIONS_PER_DAY = 1
        else:
            if ((len(valid_destinations)+len(invalid_destinations)) // len(dates)) < DESTINATIONS_PER_DAY:
                DESTINATIONS_PER_DAY = (len(valid_destinations)+len(invalid_destinations)) // len(dates)


        def collect_lat_long(list_of_destination_dicts):
            lat, long = 0., 0.
            counter = 0
            for destination_dict in list_of_destination_dicts:
                lat_long_ft = get_long_lat(long=destination_dict.get("Longitude", None), lat=destination_dict.get("Latitude", None))
                if lat_long_ft is not None:
                    lat += float(lat_long_ft[0])
                    long += float(lat_long_ft[1])
                    counter += 1
            if counter == 0:
                return None
            else: 
                return (lat/counter, long/counter)


        def find_closest_location(destination_lat, destination_lon, locations):
            def haversine(lat1, lon1, lat2, lon2):
                # convert latitude and longitude from degrees to radians
                lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                
                # haversine
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
                c = 2 * math.asin(math.sqrt(a))
                
                # earth radius in km
                r = 6371.0
                distance = r * c
                return distance
            distances = [haversine(destination_lat, destination_lon, lat, lon) for lat, lon in locations]
            closest_index = distances.index(min(distances))
            return closest_index


        date_dict = dict()
        valid_counter = 0
        invalid_counter = 0
        for date in dates:
            current_date_dict = dict()
            current_date_destination_list = []
            for i in range(DESTINATIONS_PER_DAY):
                if valid_counter > (len(valid_destinations)-1):
                    tmp_dest_dict = invalid_destinations[invalid_counter]
                    tmp_dest_dict['startDate'] = str(date)
                    tmp_dest_dict['endDate'] = str(date)
                    current_date_destination_list.append(tmp_dest_dict)
                    invalid_counter += 1
                else:
                    tmp_dest_dict = valid_destinations[valid_counter]
                    tmp_dest_dict['startDate'] = str(date)
                    tmp_dest_dict['endDate'] = str(date)
                    current_date_destination_list.append(tmp_dest_dict)
                    valid_counter += 1
            current_date_dict['destination'] = current_date_destination_list
            destination_average_lat_long = collect_lat_long(current_date_destination_list)
            
            if destination_average_lat_long is not None:
                list_of_accomodation_lat_long = [(float(acc_dict.get("Latitude")), float(acc_dict.get("Longitude"))) for acc_dict in valid_accomodations]
                accomodation_index = find_closest_location(destination_average_lat_long[0], destination_average_lat_long[1], list_of_accomodation_lat_long)
                accomodation = valid_accomodations[accomodation_index]
            else:
                if len(valid_accomodations) > 0:
                    accomodation = random.choice(valid_accomodations)
                elif len(invalid_accomodations) > 0:
                    accomodation = random.choice(invalid_accomodations)
                else:
                    raise Exception("No Accomodation Found")
            current_date_dict['destination'] = current_date_destination_list
            current_date_dict['accomodation'] = accomodation
            date_dict[str(date)] = current_date_dict

        # sort & unify accomodation datetime
        date_dict_accomodation = dict()
        for date, details in date_dict.items():
            date_dict_accomodation[date] = details['accomodation']['Name']

        def unify_accomodation_dates(date_dict_accomodation):
            new_dict = dict()
            # sort dates in case they are not in order
            sorted_dates = sorted(date_dict_accomodation.keys())

            for i, date in enumerate(sorted_dates):
                hotel = date_dict_accomodation[date]
                date_obj = datetime.strptime(date, "%Y-%m-%d")

                # if not in new dict, initialize
                if hotel not in new_dict:
                    new_dict[hotel] = [{"startDate": date, "endDate": (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")}]
                else:
                    # update end date if hotel same as prev day
                    prev_date = datetime.strptime(new_dict[hotel][-1]["endDate"], "%Y-%m-%d")
                    if prev_date < date_obj:
                        new_dict[hotel].append({"startDate": date, "endDate": (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")})
                    elif prev_date == date_obj:
                        new_dict[hotel][-1]["endDate"] = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            return new_dict

        unified_date_dict_accomodation = unify_accomodation_dates(date_dict_accomodation)


        # build final output
        trip_dict = {
            # "title": ?
            # "description": ?
            "startDate": str(starting_date),
            "endDate": str(ending_date),
            # "flights": ?
            "destinations": [],
            "accomodations": []
        }

        def get_accom_given_name(name, accomodation_list_of_dicts):
            for accom_dict in accomodation_list_of_dicts:
                if accom_dict.get("Name", None) == name:
                    return accom_dict
            return None

        trip_dict_accomodations = []
        for accom_name, date_lists in unified_date_dict_accomodation.items():
            accom_dict = get_accom_given_name(accom_name, accomodations_list_of_dicts)
            if accom_dict is not None:
                for dd in date_lists:
                    accom_dict['startDate'] = dd.get('startDate')
                    accom_dict['endDate'] = dd.get('endDate')
                    trip_dict_accomodations.append(accom_dict)
        trip_dict['accomodations'] = trip_dict_accomodations

        trip_dict_destinations = []
        for date, details in date_dict.items():
            for destination_dict in details['destination']:
                trip_dict_destinations.append(destination_dict)
        trip_dict['destinations'] = trip_dict_destinations


        trip_details_string = ""
        trip_details_string += f"Trip starting date: {trip_dict['startDate']}  "
        trip_details_string += f"Trip ending date: {trip_dict['endDate']}  "
        trip_details_string += f"Tourist Spots to be visited: {', '.join([str(spot['Name']) for spot in trip_dict['destinations']])}"

        title = Settings.llm.complete(generate_trip_title_prompt.format(details=trip_details_string))
        description = Settings.llm.complete(generate_trip_description_prompt.format(name=str(title), details=trip_details_string))

        trip_dict['title'] = str(title)
        trip_dict['description'] = str(description)
        
        return trip_dict