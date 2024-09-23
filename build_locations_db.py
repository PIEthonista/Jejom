import os
import time
import json
import googlemaps
from tqdm import tqdm


# set all to TRUE for a one-shot setup of the whole locations database
REFRESH_GMAPS = False    # 1: True to re-crawl all raw locations from google-maps
FILTER_RAW = False   # 2: filters out irrelevant locations from raw data
GET_DETAILED = True  # 3: get detailed information about each location

API_KEY = "AIzaSyD3W5PNMkYXp7NqU5RZjIhrf_GNt2GOM64"
gmaps = googlemaps.Client(key=API_KEY)

RAW_DIR = os.path.join("locations", "raw")
FILTERED_DIR = os.path.join("locations", "filtered")
DETAILED_DIR = os.path.join("locations", "detailed")


TO_OMIT = ['.DS_Store']  # mac cache
PLACE_TYPES = {
    "tourist spot": "tourist_spots",
    "hotel": "hotels",
    "restaurant": "restaurants"
}
PLACE_TYPES_KEYWORDS = {
    "tourist_spots": ['tourist', 'attraction'],
    "hotels": ['hotel', 'inn', 'accomodation', 'guesthouse', 'house', 'lodging'],
    "restaurants": ['cafe', 'restaurant', 'eatery', 'bakery', 'food'],
}
KEY_REGIONS = [
    'Jeju City (North)',
    'Aewol (Northwest)',
    'Seogwipo City (South)',
    'Jungmun Tourist Complex (Southwest)',
    'Seongsan (East)',
    'Hallim (West)',
    'Hallasan Mountain and National Park (Center)',
    'Pyoseon and Seongeup (East-Central)'
]
SEARCH_RADIUS = 10000  # m


def get_locations(query: str, type: str, radius: int=50000, lang: str='en-US', places_dict: dict = {}):

    response = gmaps.places(query=query, radius=radius, language=lang, type=type)

    for loc in response['results']:
        places_dict[loc['name']] = loc

    while 'next_page_token' in response:
        print(f"[next_page_token] : {response['next_page_token']}")
        time.sleep(2)
        response = gmaps.places(query=query, page_token=response['next_page_token'], radius=radius, language=lang, type=type)
        for loc in response['results']:
            if loc not in list(places_dict.keys()):
                places_dict[loc['name']] = loc

    return places_dict


if REFRESH_GMAPS:
    # this will re-crawl all raw locations in Jeju Island from Google Places API
    for place_type, file_basename in tqdm(PLACE_TYPES.items()):
        locations = {}
        for key_region in tqdm(KEY_REGIONS):
            locations_region = get_locations(query=f"{place_type}s in {key_region}, Jeju Island", radius=SEARCH_RADIUS, type=place_type)
            print(f"[key-region] : {key_region} got {len(list(locations_region.keys()))} results")

            filepath = os.path.join(RAW_DIR, file_basename)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            with open(os.path.join(filepath, f"{file_basename}_{str(key_region).replace(" ", "_")}.json"), "w") as f:
                json.dump(locations_region, f, indent=4)

            for loc, loc_dict in locations_region.items():
                if loc not in list(locations.keys()):
                    locations[loc] = loc_dict
            print(f"[all] : got {len(list(locations.keys()))} results")

            with open(os.path.join(filepath, f"{file_basename}_all.json"), "w") as f:
                json.dump(locations, f, indent=4)


if FILTER_RAW:
    # this will filter the raw results obtained from google maps
    for category, keywords in PLACE_TYPES_KEYWORDS.items():
        file_path = os.path.join(RAW_DIR, category, f"{category}_all.json")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        filtered_data = {}
        for loc, loc_detail in data.items():
            name_splits = str(loc).lower().split(' ')
            in_name = False
            for keyword in keywords:
                for name_split in name_splits:
                    if (keyword in name_split) or (name_split in keyword):
                        in_name = True
            
            in_type = False
            for keyword in keywords:
                for type in loc_detail['types']:
                    if (keyword in type) or (type in keyword):
                        in_type = True
            
            if in_name or in_type:
                filtered_data[loc] = loc_detail
        
        if not os.path.exists(FILTERED_DIR):
            os.makedirs(FILTERED_DIR)
        
        with open(os.path.join(FILTERED_DIR, f"{category}_filtered.json"), 'w') as f:
            json.dump(filtered_data, f, indent=4)
        
        print(f"[{category}] : {len(list(filtered_data.keys()))}")

# [tourist_spots] : 162
# [hotels] : 306
# [restaurants] : 424


if GET_DETAILED:
    # this will get the detailed information about each location from google maps
    # this will likely take up around 7-8 minutes due to rate control to prevent rate limits
    for category, keywords in PLACE_TYPES_KEYWORDS.items():
        file_path = os.path.join(FILTERED_DIR, f"{category}_filtered.json")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        detailed_dict = {}
        for loc, loc_detail in tqdm(data.items()):
            place_id = loc_detail["place_id"]
            time.sleep(0.5)
            try:
                detailed_info = gmaps.place(place_id=place_id)
                detailed_dict[loc] = detailed_info['result']
            except Exception as e:
                print(f"[ERROR] : {e}")
        
        if not os.path.exists(DETAILED_DIR):
            os.makedirs(DETAILED_DIR)
        
        with open(os.path.join(DETAILED_DIR, f"{category}_detailed.json"), 'w') as f:
            json.dump(detailed_dict, f, indent=4)
        
        print(f"[{category}] : {len(list(detailed_dict.keys()))}")