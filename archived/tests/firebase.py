import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("/Users/gohyixian/Documents/GitHub/Jejom/jejom-d5d61-firebase-adminsdk-hxhng-6f02508a1f.json")
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()


docs = db.collection('script_restaurant').get()

for doc in docs:
    print(f'{doc.id} => {doc.to_dict()}\n\n')

print(len(docs))

# fee3c7b0-2d5e-4aae-a6a8-9c65d390f1d4 => 
# {
#     'description': "Bagdad Halal Jeju, located in Gwandeok-ro, Jeju Island, South Korea, is a highly-rated (4.2/5) restaurant specializing in authentic Middle Eastern cuisine, particularly Pakistani and Indian dishes. This establishment is ideal for foodies, families, and couples seeking a unique dining experience. The restaurant's signature offerings include flavorful Chicken Biryani, delicious Butter Nan, and aromatic Masala Chai, which have received rave reviews from guests. While the food is the main attraction, the exceptional service and warm ambiance also contribute to the overall positive dining experience. Staff members are polite, English-speaking, and the owner is known for her kindness and hospitality. Although the portions may be smaller than expected, the quality of ingredients and taste make up for it. Bagdad Halal Jeju is a hidden gem in Jeju Island, offering a fine dining experience that stands out among local and international cuisine options.", 
#     'website': None, 
#     'current_opening_hours': {
#         'weekday_text': 
#             ['Monday: 11:00\u202fAM\u2009–\u200910:30\u202fPM', 'Tuesday: 11:00\u202fAM\u2009–\u200910:30\u202fPM', 'Wednesday: 11:00\u202fAM\u2009–\u200910:30\u202fPM', 'Thursday: 11:00\u202fAM\u2009–\u200910:30\u202fPM', 'Friday: 11:00\u202fAM\u2009–\u200910:30\u202fPM', 'Saturday: 11:00\u202fAM\u2009–\u200910:30\u202fPM', 'Sunday: 11:00\u202fAM\u2009–\u200910:30\u202fPM']
#         }, 
#     'geometry': {
#         'location': {
#             'lat': 33.5102389, 
#             'lng': 126.5232707
#             }
#         }, 
#     'place_id': 'ChIJ0TlM86X8DDURHCrdQS1BswI', 
#     'url': 'https://maps.google.com/?cid=194570871541410332', 
#     'rating': 4.2, 
#     'images': 
#         ['AXCi2Q5qK-OtXafn6h61kWm2cdp6XApO7AMj5PWJ0FQf-_RveXT0XN-2LO2lvRVj32EUFYPoUeosK1-9QmbKRxLlniZTgfsZATGvP437S5KxoCL1h4_IixRalYw1r9HRVNcCEs1GBArtOVSUjupbJYOM2I5QHgjPQAlsx8q6turFDCuMsa_m', 'AXCi2Q5l-lT9eDWy-Y992HBp9wkAoYmNTcAOqnoJhxVqvmWWNBp-Oki2ne_2UEAtkUQFan8VnOThQwy3gkDFxe77-Iv_a7ZcHcgWxQKditCLm6hj7cGSb82h7USph4pVGDXBD6QysGgaWSLo1t63ayu4JMc41kEdFoFEgUOUETw4EnxXOI72', 'AXCi2Q5r7kbKRNabvtRS4_GG3zlihUlpNGNfkjXBy_-k3Y47kaU38PTT3t9FGRyYCm5AWkr4NWbKNTspJYo51crf_wwUMAtBeCFE4fQDVTMRUTBYwwlvF5aT6o9KYkbwVmqzcAl-smtyD7jU_dJUXmji2_F9YFNnDxGtR58qZ7Qs5kZYOOIk', 'AXCi2Q6DkIAUX0UxKu1rOcStibbnBjKeGlGawlFYZ1LM_eMlVfrefsZ1vAjQIwoaWDEExqcnUmRai1IsbHuYsLqS04IpouLDQBC5tHnt-c7ORlL6OHCzuhp7VFtApA1zIsdxmRARESJhDgLKLCC9KRidFQKvZpgmA47QyGYgsDlY5yv1JH9d', 'AXCi2Q5oJpKUltx8hglbcbF8hNwV658fX-4U7DSR23opyt_aUmm3nGGOe8lwNf8uboI_QQxImNREiURyk1XoFdd13x5OhrIc0jEuYh4HyrOQARS2Z38vLYdzSbcavwfFiPXEIOUc-ivDMwe7KvckMFRI4ORE0cBrrkP9hRumiWB3Pz3Y6Sqq', 'AXCi2Q4gp76BeABFV9Y3nfw3W0EVOBzNiXj22tlg-FNII1eyqivZOlKE36vF7Zpegv-2AYIQmXHSDvtSBTwGuYOGYo8xYpbbTQv18cOs8pCYF94O8SVnNXIAP7ejwK_cr7m-12CXya_yYz2W9rD-m4mfnm6JsQfAT0PHX_2cJWKgXjZ9j9h_', 'AXCi2Q6CGDFDwibNsGEaviA1TW0-ARiFPJgmojfALMgua2-kZVJTCOco1bEzKqFnHUQGaPh28mueqFEovNtvTN7lQZHX_oLqENyhWhQnZeNRUDKiYPVEqOaiV8QZPGkBFe-8--KY6CN2qvnagEX7ObsOb_4YGPMx2JXm66pdMJGTKgb9rxtY', 'AXCi2Q6l2zy8Fq6Jsjg8u4IDCBdEYr3xF760chkfGkFTOzF6L2-Q11TGlwYhkPMQPMqzVp8tDSb84bNRAA9j2gTd91i9MqhqwtggD3M6_MvxsXgZ8Tl7U6uq0iu-YZ8p7Wni3sy7jDsQrGmNgvUs0iCpQdcUGXfgVXhWD9Kgvac_uPGJwWiU', 'AXCi2Q72x98O5E4uydRE1Al6g0njyTeCQD7GufZ2rt88m3gVHdvPme79ldcjOa-ER8XYdadKQgzLTYE2XhDE4N0KfajhxHjl_MwsiYOqev0vsvDkfoj1GYF8pWW9gI04JcBGYqgZ3czr2bFDlzaL8DpSQdHUNVVmq4B2rxA23uRg747Zo7sb', 'AXCi2Q78Z7AhT6r_PWi3DtQIW1RbhJ26DrxrfJ_XaqWTSl8yHWR4OhqSa2n_q4RuXvBCNnOP67a_kqBUspmN9xtyjHFmC88KbmTZ-wkdxbfIlVl6JOKdibtdVCg-91BocXL5GrQNgusKhrue-xogM_DvuXN75qc4hUZXNf40zhyTRtQOc3RH'], 
#     'address': 'South Korea, Jeju-do, Cheju, 특별자치도, Gwandeok-ro, 8길 34', 
#     'business_status': 'OPERATIONAL', 
#     'user_ratings_total': 784, 
#     'types': 
#         ['restaurant', 'point_of_interest', 'food', 'establishment'], 
#     'name': 'Bagdad Halal Jeju'
# }