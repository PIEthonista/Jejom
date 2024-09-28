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
            address = place_data.get('formatted_address', None)
            description = place_descriptions.get(place_name, None)
            images = [photo['photo_reference'] for photo in place_data.get('photos', [])]
            lat = place_data.get('geometry', {}).get('location', {}).get('lat', None)
            long = place_data.get('geometry', {}).get('location', {}).get('lng', None)
            name = place_data.get('name', None)
            business_status = place_data.get('business_status', None)
            current_opening_hours = place_data.get('“current_opening_hours”', {}).get('weekday_text', None)
            place_id = place_data.get('place_id', None)
            rating = place_data.get('rating', None)
            types = place_data.get('types', None)
            url = place_data.get('url', None)
            user_ratings_total = place_data.get('user_ratings_total', None)
            website = place_data.get('website', None)

            restaurant_data = {
                'address': address,
                'description': description,
                'images': images,
                'lat': lat,
                'long': long,
                'name': name,
                'business_status': business_status,
                'current_opening_hours': current_opening_hours,
                'place_id': place_id,
                'rating': rating,
                'types': types,
                'url': url,
                'user_ratings_total': user_ratings_total,
                'website': website
            }

            # Generate a uuid v4 for the document
            uuid_v4 = str(uuid.uuid4())
            db.collection(collection_name).document(uuid_v4).set(restaurant_data)
            print(f"Uploaded {place_name} to Firestore.")

        except KeyError as e:
            print(f"Skipping {place_name}: Missing field {e}")

if __name__ == '__main__':
    upload_restaurant_data()