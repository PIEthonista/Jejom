import os
import copy
import json
import math
import random
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from serpapi import GoogleSearch
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from datetime import datetime, timedelta

# https://github.com/mangiucugna/json_repair
from json_repair import repair_json  # LLM JSON output fixing if necessary

from tavily import TavilyClient
from llama_index.core import VectorStoreIndex, StorageContext, PromptTemplate, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
# from llama_index.llms.openai import OpenAI
# from llama_index.llms.nvidia import NVIDIA
from llama_index.llms.groq import Groq
# from llama_index.llms.upstage import Upstage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.upstage import UpstageEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.schema import BaseNode



load_dotenv()
VERBOSE = False
CLUSTER_DESTINATION_DISTANCE = 15  # km
SERPAPI_API_KEY_VAE_NAME = "SERPAPI_API_KEY"


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

numPersons_check_prompt = PromptTemplate(
    """
    You are given a user query about generating a trip. Based on the query, strictly follow these instructions:

    - If the query does **not** mention any number of people involved in the trip, output only the word "no".
    - If the query mentions the **exact number of people** (in digit form), output only the number itself as an integer.
    - If the query mentions a **group of people** (e.g., "a family," "friends," "team") but does **not** specify an exact number, output only the word "yes".

    Query: {query_str}

    Your response should strictly follow the above rules and contain **no additional text**.
    """
)

numPersons_extraction = PromptTemplate(
    """
    You are given a user query about generating a trip. Based on the query, strictly follow these instructions:

    - If the query mentions the **exact number of people** (in digit form or written out as words), output only the number itself as an integer.
    - If the query mentions a **group of people** (e.g., "a family," "friends," "team") but does **not** specify an exact number, output an estimated default value of 3.
    - If the query **does not mention** any other people or groups involved, or if you are uncertain about the number of people, output 1.

    Query: {query_str}

    Your response should strictly follow the above rules and contain **no additional text**.
    """
)

departure_airport_IATA_extraction = PromptTemplate(
    """
    You are given a description about an end user that includes their residing address, area, or country. Based on the provided description, your task is to output the **3-character IATA code** of the most appropriate departure airport if the user were to travel on a direct flight to Jeju International Airport (CJU).

    - The model must output only one single IATA code for the closest major airport suitable for a direct flight to CJU.
    - If there are multiple possible airports, select the most prominent or largest one based on the user's location.

    Query: {user_specs}

    Output the airport's IATA code as a 3-character string with **no additional text**.
    """
)

flight_class_selection_prompt = PromptTemplate(
    """
    You are a decision-making model that selects the most suitable flight class for a user's trip to Jeju Island. 
    The user query will include information about their budget, preferences, and any additional details about the trip.

    --Goal--
    Your task is to review the provided user query and determine which of the following flight classes best suits the user's needs:
    1 - Economy
    2 - Premium economy
    3 - Business
    4 - First Class

    --Steps--
    1. Analyze the user's query, focusing on the budget and any preferences related to comfort or travel style.
    2. Select the most appropriate flight class for the user based on the information provided.
    3. Output strictly one of the following integers as your decision:
       - 1 for Economy
       - 2 for Premium economy
       - 3 for Business
       - 4 for First Class

    --User Query--
    {user_query}

    --Output--
    Output strictly one integer:
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

generate_accomodation_preference_prompt = PromptTemplate(
    """
    You are a description generation model that creates detailed accommodation preferences based on user inputs. 
    The generated description will be used to search for the most suitable accommodation for the user using vector similarity.

    --Goal--
    You will be given specific information about the user's preferences, possibly the user's personality, and lifestyle too. 
    Your task is to generate a description of the type of accommodation that this user is most likely to prefer or enjoy.

    --Steps--
    1. Review the user's preferences and characteristics provided below.
    2. Think about what kind of accommodation would best fit the user's lifestyle and preferences, focusing on aspects such as location, amenities, and environment.
    3. Generate a detailed description of the accommodation that the user would like, considering factors like nearby attractions, style, and atmosphere.

    --User Data--
    User Information: {user_query}

    --Example Input--
    City Lover, Male, 30, ENTJ, Loves beachball

    --Output--
    Generate a comprehensive description of the accommodation the user would likely prefer:
    """
)

generate_tourist_attraction_preference_prompt = PromptTemplate(
    """
    You are a description generation model that creates detailed tourist attraction preferences based on user inputs. 
    The generated description will be used to search for the most suitable tourist attractions for the user using vector similarity.

    --Goal--
    You will be given specific information about the user's preferences, possibly the user's personality, and lifestyle too. 
    Your task is to generate a description of the type of tourist attraction that this user is most likely to enjoy.

    --Steps--
    1. Review the user's preferences and characteristics provided below.
    2. Think about what kind of tourist attraction would best fit the user's lifestyle and interests, focusing on aspects such as the type of attraction, its atmosphere, and nearby activities.
    3. Generate a detailed description of the tourist attractions the user would enjoy, considering factors like location, nature of activities, and cultural or entertainment value.

    --User Data--
    User Information: {user_query}

    --Example Input--
    Nature Lover, Female, 28, INFJ, Enjoys hiking and photography

    --Output--
    Generate a comprehensive description of the tourist attractions the user would likely prefer:
    """
)

tourist_attraction_pipeline_selection_prompt = PromptTemplate(
    """
    You are a decision-making model that selects the most suitable type of destinations based on the user's preferences and query about a trip to Jeju Island. 
    The goal is to determine whether the recommended destinations should be:
    - any locations (any rating, any number of ratings),
    - locations with high ratings but any number of ratings, or
    - locations with both high ratings and a high number of ratings.

    --Context--
    Keep in mind the following:
    1. Locations with both high ratings and a high number of ratings are usually more popular and crowded with tourists. They may not be suitable for individuals who prefer quiet or peaceful environments.
    2. Locations with high ratings but fewer ratings could offer quality experiences without large crowds.
    3. Any locations (regardless of rating) might appeal to users who are open to exploring unconventional or less touristy spots, or those seeking a variety of experiences.

    --Goal--
    Your task is to review the user preferences and trip query provided below, and decide which type of locations should be recommended.

    --Output Criteria--
    Based on your analysis, output strictly:
    - 0 for any locations (any rating, any number of ratings),
    - 1 for locations with high ratings but any number of ratings,
    - 2 for locations with both high ratings and a high number of ratings (while noting that these locations are typically crowded and may not suit individuals seeking quieter settings).

    --User Data--
    User Preferences: {user_specs}
    Trip Query: {user_query}

    --Output--
    Based on the user's preferences and query, output either 0, 1, or 2:
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





class PipelineV2():
    def __init__(
        self, 
        embed_model_size: int = None,
        accomodations_sim_top_k: int = 500,
        restaurants_sim_top_k: int = 500,
        tourist_spots_sim_top_k: int = 500,
        accomodations_vec_db_uri: str = os.path.join('locations', 'descriptions_vector_store', 'hotels.db'),
        restaurants_vec_db_uri: str = os.path.join('locations', 'descriptions_vector_store', 'restaurants.db'),
        tourist_spots_vec_db_uri: str = os.path.join('locations', 'descriptions_vector_store', 'tourist_spots.db'),
        accomodations_json: str = os.path.join('locations', 'detailed', 'hotels_detailed.json'),
        restaurants_json: str = os.path.join('locations', 'detailed', 'restaurants_detailed.json'),
        tourist_spots_json: str = os.path.join('locations', 'detailed', 'tourist_spots_detailed.json'),
        ):
        # accomodations
        accomodations_vector_store = MilvusVectorStore(uri=accomodations_vec_db_uri, dim=embed_model_size, overwrite=False)
        accomodations_storage_context = StorageContext.from_defaults(vector_store=accomodations_vector_store)
        accomodations_index = VectorStoreIndex(nodes=[], storage_context=accomodations_storage_context, embed_model=Settings.embed_model)
        self.accomodations_retriever = accomodations_index.as_retriever(similarity_top_k=accomodations_sim_top_k)
        
        # restaurants
        restaurants_vector_store = MilvusVectorStore(uri=restaurants_vec_db_uri, dim=embed_model_size, overwrite=False)
        restaurants_storage_context = StorageContext.from_defaults(vector_store=restaurants_vector_store)
        restaurants_index = VectorStoreIndex(nodes=[], storage_context=restaurants_storage_context, embed_model=Settings.embed_model)
        self.restaurants_retriever = restaurants_index.as_retriever(similarity_top_k=restaurants_sim_top_k)
        
        # tourist spots
        tourist_spots_vector_store = MilvusVectorStore(uri=tourist_spots_vec_db_uri, dim=embed_model_size, overwrite=False)
        tourist_spots_storage_context = StorageContext.from_defaults(vector_store=tourist_spots_vector_store)
        tourist_spots_index = VectorStoreIndex(nodes=[], storage_context=tourist_spots_storage_context, embed_model=Settings.embed_model)
        self.tourist_spots_retriever = tourist_spots_index.as_retriever(similarity_top_k=tourist_spots_sim_top_k)
        
        # json file paths
        self.accomodations_json = accomodations_json
        self.restaurants_json = restaurants_json
        self.tourist_spots_json = tourist_spots_json
        with open(accomodations_json, 'r') as f:
            self.accomodations_json_data = json.load(f)
        with open(restaurants_json, 'r') as f:
            self.restaurants_json_data = json.load(f)
        with open(tourist_spots_json, 'r') as f:
            self.tourist_spots_json_data = json.load(f)


    def check_query_detail(self, query: str):
        # isDestination    = True if str(Settings.llm.complete(isDestination_check_prompt.format(query_str=query))).lower().strip() == "yes" else False
        # isInterest       = True if str(Settings.llm.complete(isInterest_check_prompt.format(query_str=query))).lower().strip() == "yes" else False
        isDuration       = True if str(Settings.llm.complete(isDuration_check_prompt.format(query_str=query))).lower().strip() == "yes" else False
        isBudget         = True if str(Settings.llm.complete(isBudget_check_prompt.format(query_str=query))).lower().strip() == "yes" else False
        numPerson_output = str(Settings.llm.complete(numPersons_check_prompt.format(query_str=query))).lower().strip()
        # mention group but not number of people
        if "yes" in numPerson_output:
            IsNumPerson = False
        # no mention group, assume 1
        elif "no" in numPerson_output:
            IsNumPerson = True
        # mentions exact number
        else:
            IsNumPerson = True
        
        return {
            # "isDestination": isDestination,
            # "isInterest": isInterest,
            "isDuration": isDuration,
            "isBudget": isBudget,
            "isNumPerson": IsNumPerson
        }


    def get_accomodations_list(self, query: str) -> list[BaseNode]:
        # end_user_specs = 'City Lover, Male, 30, ENTJ, Loves beachball'
        # end_user_specs = "Loves a chill life, doesnt like crowded places, loves coffee, artistic, female, 25, ENTP"
        
        formatted_accomodation_prompt = generate_accomodation_preference_prompt.format(user_query=query)
        accomodations_preference = Settings.llm.complete(formatted_accomodation_prompt)
        
        accoms_list = self.accomodations_retriever.retrieve(str(accomodations_preference))
        return accoms_list


    def get_destinations_list(self, query: str) -> list[BaseNode]:
        # end_user_specs = 'City Lover, Male, 30, ENTJ, Loves beachball'
        # end_user_specs = "Loves a chill life, doesnt like crowded places, loves coffee, artistic, female, 25, ENTP"
        
        formatted_destination_prompt = generate_tourist_attraction_preference_prompt.format(user_query=query)
        destinations_preference = Settings.llm.complete(formatted_destination_prompt)
        
        destinations_list = self.tourist_spots_retriever.retrieve(str(destinations_preference))
        return destinations_list


    def generate_dates(self, start_date, end_date):
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


    def get_flights(self, user_specs, query, outbound_date, return_date):
        try:
            num_person = int(str(Settings.llm.complete(numPersons_extraction.format(query_str=query))))
        except:
            num_person = 1
        
        departure_IATA = str(Settings.llm.complete(departure_airport_IATA_extraction.format(user_specs=user_specs))).strip().upper()
        
        travel_class = int(str(Settings.llm.complete(flight_class_selection_prompt.format(user_query=query))).strip())
        
        search_params = {
            "api_key": os.getenv(SERPAPI_API_KEY_VAE_NAME),
            "engine": "google_flights",
            "hl": "en",
            "gl": "kr",  # "my"
            "departure_id": str(departure_IATA),
            "arrival_id": "CJU",
            "outbound_date": str(outbound_date),
            "return_date": str(return_date),
            "currency": "KRW",
            "type": "1",
                # 1 - Round trip (default)
                # 2 - One way
                # 3 - Multi-city: multi_city_json required
            "travel_class": str(travel_class),
                # 1 - Economy (default)
                # 2 - Premium economy
                # 3 - Business
                # 4 - First
            "show_hidden": "true",
            "adults": str(num_person),
            # "children": "0",
            # "infants_in_seat": "0",
            # "infants_on_lap": "0",
            "stops": "0"
        }
        try:
            search = GoogleSearch(search_params)
            flights_dict = search.get_dict()
        except Exception as e:
            print(f"[Get Flights] Error: {e}")
            flights_dict = None
        
        departure_flight_backup = {}   # final resort
        found_departure = False
        found_return = False
        
        if flights_dict is not None:
            if "best_flights" in list(flights_dict.keys()):
                departure_flight_backup = copy.deepcopy(flights_dict["best_flights"][0])
                for idx in range(len(flights_dict["best_flights"])):
                    departure_flight_dict = flights_dict["best_flights"][idx]
                    found_departure = True
                    departure_token = departure_flight_dict["departure_token"]
                    
                    # construct search params
                    search_return_params = copy.deepcopy(search_params)
                    search_return_params["departure_token"] = departure_token
                    
                    # search
                    try:
                        search_return = GoogleSearch(search_return_params)
                        return_flights_dict = search_return.get_dict()
                    except Exception as e:
                        print(f"[Get Flights] Error: {e}")
                        return_flights_dict = None
                    
                    # get return flight in obtained return dict
                    if return_flights_dict is not None:
                        if "best_flights" in list(return_flights_dict.keys()):
                            return_flight_dict = return_flights_dict["best_flights"][0]
                            found_return = True
                        else:
                            if "other_flights" in list(return_flights_dict.keys()):
                                if len(return_flights_dict["other_flights"]) > 0:
                                    return_flight_dict = return_flights_dict["other_flights"][0]
                                    found_return = True
        
            if ("other_flights" in list(flights_dict.keys())) and (not found_return):
                if len(flights_dict["other_flights"]) > 0:
                    if not found_departure:
                        departure_flight_backup = copy.deepcopy(flights_dict["other_flights"][0])
                    for idx in range(len(flights_dict["other_flights"])):
                        departure_flight_dict = flights_dict["other_flights"][idx]
                        found_departure = True
                        departure_token = departure_flight_dict["departure_token"]
                        
                        # construct search params
                        search_return_params = copy.deepcopy(search_params)
                        search_return_params["departure_token"] = departure_token
                        
                        # search
                        try:
                            search_return = GoogleSearch(search_return_params)
                            return_flights_dict = search_return.get_dict()
                        except Exception as e:
                            print(f"[Get Flights] Error: {e}")
                            return_flights_dict = None
                        
                        # get return flight in obtained return dict
                        if return_flights_dict is not None:
                            if "best_flights" in list(return_flights_dict.keys()):
                                return_flight_dict = return_flights_dict["best_flights"][0]
                                found_return = True
                            else:
                                if "other_flights" in list(return_flights_dict.keys()):
                                    if len(return_flights_dict["other_flights"]) > 0:
                                        return_flight_dict = return_flights_dict["other_flights"][0]
                                        found_return = True
        
        # utility to format time from m to h:m
        def convert_duration_to_string(data):
            def convert_minutes_to_str(minutes):
                hours = minutes // 60
                minutes = minutes % 60
                return f"{hours} hr {minutes} min" if hours else f"{minutes} min"

            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        convert_duration_to_string(value)
                    elif isinstance(value, int) and "duration" in str(key).lower():
                        data[key] = convert_minutes_to_str(value)

            elif isinstance(data, list):
                for item in data:
                    convert_duration_to_string(item)
        
        if found_departure and found_return:
            _ = departure_flight_dict.pop("price")
            price_total = return_flight_dict.pop("price")
            
            # convert time format
            convert_duration_to_string(departure_flight_dict)
            convert_duration_to_string(return_flight_dict)
            
            return {
                "departureFlight": departure_flight_dict,
                "returnFlight": return_flight_dict,
                "priceTotal": int(price_total)
            }

        elif found_departure and (not found_return):
            price_total = departure_flight_backup.pop("price")
            
            # convert time format
            convert_duration_to_string(departure_flight_backup)
            
            return {
                "departureFlight": departure_flight_backup,
                "returnFlight": None,
                "priceTotal": int(price_total)
            }
        else:
            return {
                "departureFlight": None,
                "returnFlight": None,
                "priceTotal": None
            }




    def generate_trip(self, end_user_specs: str, end_user_query: str):
        # end_user_specs = "Loves a chill life, doesnt like crowded places, loves coffee, artistic, female, 25, ENTP"
        # end_user_specs = "Nature Lover, Photography, Solo-Traveller"
        # end_user_final_query = "Can you plan me a trip to Jeju Island from 18 January to 20 January in 2025? My budget is around MYR 3000. " + end_user_specs
        # end_user_final_query = "Can you plan me a 3-day trip to Jeju Island starting from 18 January 2025? My budget is around MYR 3000. " + end_user_specs

        starting_date = Settings.llm.complete(get_starting_date_prompt.format(query_str=end_user_query))
        ending_date = Settings.llm.complete(get_ending_date_prompt.format(query_str=end_user_query))

        print(starting_date)
        print(ending_date)

        dates = self.generate_dates(str(starting_date), str(ending_date))
        print(dates)


        query = f"User Details: {end_user_specs}. \n User Query: {end_user_query}"
        accomodations_nodelist = self.get_accomodations_list(query=query)
        destinations_nodelist = self.get_destinations_list(query=query)
        
        destinations_list_of_dict = []
        for node in destinations_nodelist:
            dest_name = str(node.text)
            price = "Not available"  # TODO: FIX
            address = str(self.tourist_spots_json_data[dest_name]['formatted_address'])
            lat = float(self.tourist_spots_json_data[dest_name]['geometry']['location']['lat'])
            lng = float(self.tourist_spots_json_data[dest_name]['geometry']['location']['lng'])
            # rating
            if 'rating' in self.tourist_spots_json_data[dest_name].keys():
                rating = str(self.tourist_spots_json_data[dest_name]['rating'])
            else:
                rating = "Not Available"
            # num ratings
            if 'user_ratings_total' in self.tourist_spots_json_data[dest_name].keys():
                num_ratings = int(self.tourist_spots_json_data[dest_name]['user_ratings_total'])
            else:
                num_ratings = 0
            # opening hours
            if "current_opening_hours" in self.tourist_spots_json_data[dest_name].keys():
                if "weekday_text" in self.tourist_spots_json_data[dest_name]["current_opening_hours"].keys():
                    opening_hours = self.tourist_spots_json_data[dest_name]["current_opening_hours"]["weekday_text"]
                else:
                    opening_hours = ['Not Available']
            else:
                opening_hours = ['Not Available']
            # photos
            if "photos" in self.tourist_spots_json_data[dest_name].keys():
                photos = self.tourist_spots_json_data[dest_name]["photos"]
            else:
                photos = []
            
            destinations_list_of_dict.append(
                {
                    "Name": dest_name,
                    "Description": node.metadata['description'],
                    "Price": price,
                    "Address": address,
                    "Latitude": lat,
                    "Longitude": lng,
                    "Rating": rating,
                    "NumRating": num_ratings,
                    "OpeningHours": opening_hours,
                    "Photos": photos,
                    "GooglePlaceID": self.tourist_spots_json_data[dest_name]['place_id']
                }
            )
        
        
        # rerank destinations based on both rating & number of ratings, or based on ratings only
        need_reranking = Settings.llm.complete(tourist_attraction_pipeline_selection_prompt.format(user_specs=end_user_specs, user_query=end_user_query))
        try:
            reranking_mode = int(str(need_reranking))
            print(f"[reranking_mode={reranking_mode}]")
        except:
            reranking_mode = 1
            print(f"[Exception: Default to reranking_mode=1, need_reranking={need_reranking}]")
        
        # - 0 for any locations (any rating, any number of ratings)
        if reranking_mode == 0:
            # keep destinations_list_of_dict the same
            pass
        
        # - 1 for locations with high ratings but any number of ratings,
        elif (reranking_mode == 1) or reranking_mode > 2:  # >2 for fallback in case
            destinations_list_of_dict = sorted(destinations_list_of_dict, key=lambda x: x["Rating"], reverse=True)
        
        # - 2 for locations with both high ratings and a high number of ratings (while noting that these locations are typically crowded and may not suit individuals seeking quieter settings).
        elif reranking_mode == 2:
            def custom_f1(x):
                NumRating = x["NumRating"]
                Rating = x["Rating"]
                num_rating_normalized = (NumRating / 1e6) if isinstance(NumRating, int) else 0
                rating_normalized = (Rating / 5) if isinstance(Rating, int) else 0
                f1 = 2 * (num_rating_normalized * rating_normalized) / (num_rating_normalized + rating_normalized + 1e-10)
                return f1
            destinations_list_of_dict = sorted(destinations_list_of_dict, key=custom_f1, reverse=True)
        
        # TODO: FIX
        DESTINATIONS_PER_DAY = 2
        TOTAL_DESTINATIONS = len(dates)*DESTINATIONS_PER_DAY
        if len(destinations_list_of_dict) <= TOTAL_DESTINATIONS:
            pass
        else:
            destinations_list_of_dict = destinations_list_of_dict[:TOTAL_DESTINATIONS]
        
        # custom distance function
        def haversine_distance(coord1, coord2):
            return great_circle(coord1, coord2).km

        # Perform DBSCAN clustering, eps in km
        destinations_long_lat = [(i["Latitude"], i["Longitude"]) for i in destinations_list_of_dict]
        destinations_long_lat_np = np.array(destinations_long_lat)
        dbscan = DBSCAN(eps=CLUSTER_DESTINATION_DISTANCE, min_samples=1, metric=haversine_distance)
        clusters = dbscan.fit_predict(destinations_long_lat_np)
        clusters = clusters.tolist()
        assert len(clusters) == len(destinations_list_of_dict)


        def order_indices_by_values(lst):
            indices = list(range(len(lst)))
            sorted_indices = sorted(indices, key=lambda x: lst[x])
            return sorted_indices

        # order destinations by cluster
        clusters_indices = order_indices_by_values(clusters)
        destinations_list_of_dict = [destinations_list_of_dict[i] for i in clusters_indices]

        # adjust destinations per day accordingly
        if len(dates) > len(destinations_list_of_dict):
            dates = dates[:len(destinations_list_of_dict) - 1]
            ending_date = dates[-1]
            DESTINATIONS_PER_DAY = 1
        elif len(dates) == len(destinations_list_of_dict):
            DESTINATIONS_PER_DAY = 1
        else:
            if (len(destinations_list_of_dict) // len(dates)) < DESTINATIONS_PER_DAY:
                DESTINATIONS_PER_DAY = len(destinations_list_of_dict) // len(dates)


        def get_lat_long_average(list_of_destination_dicts):
            lat, long = 0., 0.
            counter = 0
            for destination_dict in list_of_destination_dicts:
                lat += float(destination_dict.get("Latitude"))
                long += float(destination_dict.get("Longitude"))
                counter += 1
            if counter == 0:
                return None
            else: 
                return (lat/counter, long/counter)


        def get_batch_distance(destination_lat, destination_lon, locations_list_of_dict):
            def haversine(lat1, lon1, lat2, lon2):
                # convert latitude and longitude from degrees to radians
                lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                
                # haversine
                delta_lat = lat2 - lat1
                delta_lon = lon2 - lon1
                a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
                c = 2 * math.asin(math.sqrt(a))
                
                # earth radius in km
                r = 6371.0
                distance = r * c
                return distance
            
            return_locations_list_of_dict = []
            for location_dict in locations_list_of_dict:
                distances = haversine(destination_lat, destination_lon, location_dict['Latitude'], location_dict['Longitude'])
                dict_copy = copy.deepcopy(location_dict)
                dict_copy['_distance_from_target'] = float(distances)
                return_locations_list_of_dict.append(dict_copy)
            return return_locations_list_of_dict


        # form dict with some basic info for accomodatios first
        accomodations_list_of_dict = []
        for node in accomodations_nodelist:
            accom_name = str(node.text)
            address = str(self.accomodations_json_data[accom_name]['formatted_address'])
            lat = float(self.accomodations_json_data[accom_name]['geometry']['location']['lat'])
            lng = float(self.accomodations_json_data[accom_name]['geometry']['location']['lng'])
            # rating
            if 'rating' in self.accomodations_json_data[accom_name].keys():
                rating = str(self.accomodations_json_data[accom_name]['rating'])
            else:
                rating = "Not Available"
            # num ratings
            if 'user_ratings_total' in self.accomodations_json_data[accom_name].keys():
                num_ratings = int(self.accomodations_json_data[accom_name]['user_ratings_total'])
            else:
                num_ratings = 0
            accomodations_list_of_dict.append(
                {
                    "Name": accom_name,
                    "Description": node.metadata['description'],
                    "_cosine_sim_score": float(node.score),    # remove before returning
                    "Address": address,
                    "Latitude": lat,
                    "Longitude": lng,
                    "Rating": rating,           # possibly useful for reccomendations
                    "NumRating": num_ratings,
                }
            )


        date_dict = dict()
        list_of_selected_accom_dicts = []
        valid_counter = 0
        for date in dates:
            current_date_dict = dict()
            current_date_destination_list = []
            for i in range(DESTINATIONS_PER_DAY):
                tmp_dest_dict = destinations_list_of_dict[valid_counter]
                tmp_dest_dict['startDate'] = str(date)
                tmp_dest_dict['endDate'] = str(date)
                current_date_destination_list.append(tmp_dest_dict)
                valid_counter += 1
            current_date_dict['destination'] = current_date_destination_list
            destination_average_lat_long = get_lat_long_average(current_date_destination_list)
            
            distanced_accomodations_list_of_dict = get_batch_distance(destination_average_lat_long[0], destination_average_lat_long[1], accomodations_list_of_dict)
            
            # sort according to distance first, then only by preference
            sorted_accomodations_list_of_dict = sorted(
                distanced_accomodations_list_of_dict, 
                key=lambda x: (x['_distance_from_target'], -x['_cosine_sim_score'])
            )
            selected_accomodation = sorted_accomodations_list_of_dict[0]
            
            # remove unnecessary info from the accomodation dict
            selected_accomodation.pop('_distance_from_target')
            selected_accomodation.pop('_cosine_sim_score')
            accom_name = selected_accomodation['Name']
            
            # now add in full details to the accomodation dict
            # price
            price = "Not Available"  # TODO: FIX
            
            # opening hours
            if "current_opening_hours" in self.accomodations_json_data[accom_name].keys():
                if "weekday_text" in self.accomodations_json_data[accom_name]["current_opening_hours"].keys():
                    opening_hours = self.accomodations_json_data[accom_name]["current_opening_hours"]["weekday_text"]
                else:
                    opening_hours = ['Not Available']
            else:
                opening_hours = ['Not Available']
            
            # photos
            if "photos" in self.accomodations_json_data[accom_name].keys():
                photos = self.accomodations_json_data[accom_name]["photos"]
            else:
                photos = []
            
            # add to dict
            selected_accomodation["Price"] = price
            selected_accomodation["OpeningHours"] = opening_hours
            selected_accomodation["Photos"] = photos
            selected_accomodation["GooglePlaceID"] = self.accomodations_json_data[accom_name]['place_id']
            
            # add accomodations and destionations to date dict
            current_date_dict['destination'] = current_date_destination_list
            current_date_dict['accomodation'] = selected_accomodation
            date_dict[str(date)] = current_date_dict
            
            # save all selected accomodations for later use
            list_of_selected_accom_dicts.append(copy.deepcopy(selected_accomodation))


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
            accom_dict = get_accom_given_name(accom_name, list_of_selected_accom_dicts)
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



        # flights
        starting_date_obj_m1 = datetime.strptime(str(starting_date), "%Y-%m-%d") - timedelta(days=1)
        ending_date_obj_m1 = datetime.strptime(str(ending_date), "%Y-%m-%d") + timedelta(days=1)
        outbound_date = starting_date_obj_m1.strftime("%Y-%m-%d")
        return_date = ending_date_obj_m1.strftime("%Y-%m-%d")
        trip_dict["flightInfo"] = self.get_flights(user_specs=end_user_specs, query=end_user_query, outbound_date=outbound_date, return_date=return_date)


        trip_details_string = ""
        trip_details_string += f"Trip starting date: {trip_dict['startDate']}  "
        trip_details_string += f"Trip ending date: {trip_dict['endDate']}  "
        trip_details_string += f"Tourist Spots to be visited: {', '.join([str(spot['Name']) for spot in trip_dict['destinations']])}"

        title = Settings.llm.complete(generate_trip_title_prompt.format(details=trip_details_string))
        description = Settings.llm.complete(generate_trip_description_prompt.format(name=str(title), details=trip_details_string))

        trip_dict['title'] = str(title)
        trip_dict['description'] = str(description)
        
        return trip_dict