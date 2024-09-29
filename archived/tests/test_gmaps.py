import googlemaps
import time
import json

API_KEY = "AIzaSyD3W5PNMkYXp7NqU5RZjIhrf_GNt2GOM64"
gmaps = googlemaps.Client(key=API_KEY)


# def get_locations_in_jeju(query: str, radius: int=50000, lang: str='en-US'):
#     places_dict = {}

#     # Initial request
#     response = gmaps.places(query=query, radius=radius, language=lang)

#     for loc in response['results']:
#         places_dict[loc['name']] = loc

#     # Check if there is a next_page_token
#     while 'next_page_token' in response:
#         print(f"[next_page_token] : {response['next_page_token']}")
#         # Google requires a short pause before making the next page request
#         time.sleep(2)
#         # Get the next page of results
#         response = gmaps.places(query=query, page_token=response['next_page_token'])
#         for loc in response['results']:
#             if loc not in list(places_dict.keys()):
#                 places_dict[loc['name']] = loc

#     return places_dict

# # Fetch all lodging places
# locations = get_locations_in_jeju(query="tourist spots in Jeju Island")

# with open('locations.json', 'w') as f:
#     json.dump(locations, f, indent=4)



PLACE_ID = "ChIJTe2zdXlhDDUR8fhehnbJtJc"
place_details = gmaps.place(place_id=PLACE_ID)

# Extract detailed information
details = place_details['result']
print(details)

with open('locations.json', 'w') as f:
    json.dump(details, f, indent=4)