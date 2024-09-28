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

# tnpUaddZLYrK7gAv6Skn => 
# {
#     'description': "Warm scones and a cup of coffee, paired with an idyllic view of Jeju's western coastline - can you imagine a more calming scene? That's exactly what you'll get at Biyangnol, a charming cafe with wide glass windows made for you to sit by, as you sip on your coffee and soak in views of the west coast, as well as of Biyangdo, a small island located across the waters.", 
#     'image_url': ['https://res.klook.com/image/upload/fl_lossy.progressive,q_85/c_fill,w_1000/v1683606152/blog/hi1sgalwmzuutdt8cmgc.webp', 'https://res.klook.com/image/upload/fl_lossy.progressive,q_85/c_fill,w_1000/v1683606008/blog/ykuzvhlq1tdnd0lgcblc.webp', 'https://res.klook.com/image/upload/fl_lossy.progressive,q_85/c_fill,w_1000/v1683606288/blog/q08jylbqc0zb5f9xxxoj.webp'], 
#     'long': 126.2635914, 
#     'lat': 33.4289754, 
#     'address': '965 Suwon-ri, Hallim-eup, Jeju-si, Jeju-do, South Korea', 
#     'phone_num': '+821053994962', 
#     'name': 'Biyangnol'
# }