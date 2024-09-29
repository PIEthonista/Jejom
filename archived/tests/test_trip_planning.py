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



# Global wide models
load_dotenv() # api keys
Settings.llm = Groq(model='llama3-groq-70b-8192-tool-use-preview')
# Settings.llm = Upstage(model='solar-1-mini-chat')
Settings.embed_model = UpstageEmbedding(model='solar-embedding-1-large')


# Config
VERBOSE = False
CLUSTER_DESTINATION_DISTANCE = 15

# end_user_specs = 'City Lover, Male, 30, ENTJ, Loves beachball'
# end_user_specs = "Loves a chill life, doesnt like crowded places, loves coffee, artistic, female, 25, ENTP"
end_user_specs = "Nature Lover, Photography, Solo-Traveller"
# end_user_final_query = "Can you plan me a trip to Jeju Island from 18 January to 20 January in 2025? My budget is around MYR 3000. " + end_user_specs
end_user_final_query = "Can you plan me a 3-day trip to Jeju Island starting from 18 January 2025? My budget is around MYR 3000. " + end_user_specs



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


# get destinations and accomdoations here
DESTINATIONS_PER_DAY = 2
TOTAL_ACCOMODATIONS = 5
TOTAL_DESTINATIONS = max(10, len(dates)*DESTINATIONS_PER_DAY)
from mockup_test_data import destinations_list_of_dicts, accomodations_list_of_dicts


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

# locations = [(37.7749, -122.4194), (34.0522, -118.2437), (40.7128, -74.0060)]

# print(destinations_long_lat)
destinations_long_lat_np = np.array(destinations_long_lat)
# print(destinations_long_lat_np)

# Perform DBSCAN clustering, eps in km
dbscan = DBSCAN(eps=CLUSTER_DESTINATION_DISTANCE, min_samples=1, metric=haversine_distance)
clusters = dbscan.fit_predict(destinations_long_lat_np)
clusters = clusters.tolist()
assert len(clusters) == len(valid_destinations)


# more_than_one_day_destination_prompt = PromptTemplate(
#     """
#     You are a classification model. Given a tourist attraction's name and description in Jeju Island, determine if it is a tourist attraction that one should spend more than one day at.
#     If the tourist attraction is one that one should spend more than a day at, respond with "yes".
#     If the tourist attraction is one that one can finish visiting within a day, respond with "no".
    
#     Query:
#     Tourist Attraction Name: {name}
#     Tourist Attraction Description: {desc}
#     """
# )

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

trip_details_string = ""

trip_details_string += f"Trip starting date: {trip_dict['startDate']}  "
trip_details_string += f"Trip ending date: {trip_dict['endDate']}  "
trip_details_string += f"Tourist Spots to be visited: {', '.join([str(spot['Name']) for spot in trip_dict['destinations']])}"

title = Settings.llm.complete(generate_trip_title_prompt.format(details=trip_details_string))
description = Settings.llm.complete(generate_trip_description_prompt.format(name=str(title), details=trip_details_string))

trip_dict['title'] = str(title)
trip_dict['description'] = str(description)

# trip_dict_string = json.dumps(trip_dict)

with open("test_trip_planning_output.json", "w") as json_file:
    json.dump(trip_dict, json_file, indent=4)
