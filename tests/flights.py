from serpapi import GoogleSearch
import json

params = {
  "api_key": "d3755a4b3117eb33196b648f3b221a9c7cbfb6b3ff7c0ffc9eb025d1d6f418a3",
  "engine": "google_flights",
  "hl": "en",
  "gl": "kr",  # "my"
  "departure_id": "KUL",
  "arrival_id": "CJU",
  "outbound_date": "2024-09-27",
  "return_date": "2024-10-03",
  "currency": "KRW",
  "type": "1",
  "travel_class": "1",
  "show_hidden": "true",
  "adults": "1",
  "children": "0",
  "infants_in_seat": "0",
  "infants_on_lap": "0",
  "stops": "0"
}

search = GoogleSearch(params)
results = search.get_dict()

with open('flights.json', 'w') as f:
    json.dump(results, f, indent=4)