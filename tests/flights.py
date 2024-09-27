from serpapi import GoogleSearch
import json

params = {
  "api_key": "d3755a4b3117eb33196b648f3b221a9c7cbfb6b3ff7c0ffc9eb025d1d6f418a3",
  "engine": "google_flights",
  "hl": "en",
  "gl": "kr",  # "my"
  "departure_id": "KUL",
  "arrival_id": "CJU",
  "outbound_date": "2024-12-01",
  "return_date": "2024-12-20",
  "currency": "KRW",
  "type": "1",
  "travel_class": "1",
  "show_hidden": "true",
  "adults": "1",
  "children": "0",
  "infants_in_seat": "0",
  "infants_on_lap": "0",
  "stops": "0",
  # "departure_token": "WyJDalJJYUdwdWNub3dNMmhZWldOQlExOWpTMUZDUnkwdExTMHRMUzB0TFhsc1puWXhNVUZCUVVGQlIySXlhRXhuUTJOV05qWkJFZ3RhU0RNek5IeGFTRFkwTVJvTENJU2FKaEFBR2dOTFVsYzRISENsOUFJPSIsW1siS1VMIiwiMjAyNC0xMi0wMSIsIlNaWCIsbnVsbCwiWkgiLCIzMzQiXSxbIlNaWCIsIjIwMjQtMTItMDEiLCJDSlUiLG51bGwsIlpIIiwiNjQxIl1dXQ=="
}
# params = {
#     "api_key": "d3755a4b3117eb33196b648f3b221a9c7cbfb6b3ff7c0ffc9eb025d1d6f418a3",
#     "engine": "google_flights",
#     "hl": "en",
#     "gl": "kr",  # "my"
#     "departure_token": "WyJDalJJVFVoSFRWRkVTek54UkVGQlJuQXRhRUZDUnkwdExTMHRMUzB0TFMxMmRHZDVNa0ZCUVVGQlIySXhiMjVKUTBsZlFqSkJFZ3RhU0RNek5IeGFTRFkwTVJvTENKRHZmQkFBR2dOTFVsYzRISEM0dlFrPSIsW1siS1VMIiwiMjAyNC0xMS0zMCIsIlNaWCIsbnVsbCwiWkgiLCIzMzQiXSxbIlNaWCIsIjIwMjQtMTItMDEiLCJDSlUiLG51bGwsIlpIIiwiNjQxIl1dXQ=="
# }

search = GoogleSearch(params)
results = search.get_dict()

with open('flights.json', 'w') as f:
    json.dump(results, f, indent=4)