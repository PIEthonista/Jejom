import json
from json_repair import repair_json

json_response_str = """
[{
    "Name": "Seongsan Ilchulbong",
    "Description": "Seongsan Ilchulbong, also known as 'Sunrise Peak,' is a volcano on eastern Jeju Island, in Seongsan-ri, Seogwipo, Jeju Province, South Korea. It is 182 meters high and has a volcanic crater at the top. Considered one of South Korea's most beautiful tourist sites, it is famed for being the easternmost mountain on Jeju, and thus the best spot on the island to see the sunrise.",
    "Price": "KRW 2,000",
    "Address": "Seongsan-eup, Seogwipo-si, Jeju-do, South Korea",
    "Latitude": "33.461111",
    "Longitude": "126.940556"
},{
    "Name": "Jeju Olle Trails",
    "Description": "The Jeju Olle Trails are a series of walking trails that loop around the outskirts of Jeju-do. The trails cover a distance of 425 km and are divided into 21 main routes and 5 sub-routes. The word olle (올레) in the Jeju dialect means 'a narrow street that leads to a wider road'. The trails offer a unique way to explore Jeju Island, with options for a month-long through-hike or exploring in bite-sized sections.",
    "Price": "Free",
    "Address": "Jeju Island, South Korea",
    "Latitude": "33.373056",
    "Longitude": "126.531944"
}{
  "Name": "Cheonjeyeon Waterfall",
  "Description": "Cheonjeyeon Waterfall is a three-tier waterfall located on Jeju Island, South Korea. It is a popular tourist attraction and one of the three famous waterfalls of Jeju, alongside Cheonjiyeon Waterfall and Jeongbang Waterfall. The waterfall has three sections, with the first running from the floor of the mountain on the upper part of Jungmun-dong, which falls 22 meters. The water then falls again two more times to form the second and third sections, which then tributes to the sea. The first segment is usually a pond, but falls when it rains. The forest in which the fall is located is in a warm temperature zone so it is home to a variety of flora and fauna.",
  "Price": "None",
  "Address": "Jungmun-dong, Seogwipo, Jeju-do, South Korea",
  "Latitude": "33.254",
  "Longitude": "126.417"
},]
"""

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

print("============")
print(json_response)