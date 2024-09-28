import json
import uuid
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firestore
cred = credentials.Certificate('google-services.json')  
firebase_admin.initialize_app(cred)
db = firestore.client()

def upload_restaurant_data():
    with open('locations/detailed/restaurants_detailed.json', 'r', encoding='utf-8') as details_file:
        detailed_places = json.load(details_file)

    with open('locations/descriptions/restaurants_descriptions.json', 'r', encoding='utf-8') as descriptions_file:
        place_descriptions = json.load(descriptions_file)

    collection_name = 'script_restaurant'

    place_count = 0 
    for place_name, place_data in detailed_places.items():
        if place_name not in place_descriptions:
            print(f"Skipping {place_name}: Description not found.")
            continue
        
        try:
            address = place_data['formatted_address']
            description = place_descriptions[place_name]
            images = [photo['photo_reference'] for photo in place_data.get('photos', [])]
            lat = place_data['geometry']['location']['lat']
            long = place_data['geometry']['location']['lng']
            name = place_data['name']
            phone_num = place_data.get('formatted_phone_number', '')

            restaurant_data = {
                'address': address,
                'description': description,
                'image_url': images,
                'lat': lat,
                'long': long,
                'name': name,
                'phone_num': phone_num,
                'name': name
            }

            # Generate a uuid v4 for the document
            uuid_v4 = str(uuid.uuid4())
            db.collection(collection_name).document(uuid_v4).set(restaurant_data)
            print(f"Uploaded {place_name} to Firestore.")

            place_count += 1

            if place_count >= 5:
                print("Uploaded 5 places. Stopping.")
                break

        except KeyError as e:
            print(f"Skipping {place_name}: Missing field {e}")

if __name__ == '__main__':
    upload_restaurant_data()