import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime

from llama_index.core import VectorStoreIndex, StorageContext, PromptTemplate, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
# from llama_index.llms.openai import OpenAI
# from llama_index.llms.nvidia import NVIDIA
from llama_index.llms.upstage import Upstage
from llama_index.embeddings.upstage import UpstageEmbedding
from llama_index.core.schema import TextNode


load_dotenv()
DETAILED_DIR = os.path.join("locations", "detailed")
DESCRIPTION_DIR = os.path.join("locations", "descriptions")
VECTOR_DB_DIR = os.path.join("locations", "descriptions_vector_store")

GENERATE_DESC = True     # generates description for vector db indexing, each location costs an LLM call
GENERATE_SUITABLE_TIMES = True    # optimizes suitable times for visiting, each location costs an LLM call (applicable for tourist attractions only)
GENERATE_VECTOR_INDEX = True   # takes each location as a Document object and embeds it into the vector space alongside its description
TEST_VECTOR_INDEX = True   # for testing purposes only, always set to False

TO_OMIT = ['.DS_Store']  # mac cache

# Settings.llm = Groq(model='llama3-groq-70b-8192-tool-use-preview')
Settings.llm = Upstage(model='solar-pro')
Settings.embed_model=UpstageEmbedding(model='solar-embedding-1-large')
EMBED_MODEL_SIZE = 4096

all_categories = [
    'hotels',
    'restaurants',
    'tourist_spots'
]

GENERATE_HOTEL_DESCRIPTION_PROMPT = PromptTemplate(
    """
    You are a description generator for a hotel in Jeju Island. 
    You will be provided the name, brief description, location, overall rating, and multiple reviews about the hotel.
    Your task is to generate a **comprehensive, well-rounded** description that covers important aspects of the hotel, making it suitable for embedding and retrieval based on user queries.

    --Steps--
    1. Carefully review the provided hotel information: name, description, location, and reviews.
    2. Generate a description as a **third-person narrator**, highlighting the hotel's strengths and distinctive qualities.
    3. Ensure that the description includes:
    - Who the hotel is ideal for (families, couples, business travelers, etc.).
    - Notable features (room types, amenities, dining options).
    - The hotel's proximity to **key landmarks, attractions, and transportation**.
    - Guest experiences, focusing on positives like comfort, cleanliness, and service quality.
    - Any details on the **overall value** or unique characteristics that make the hotel stand out.

    --Data Provided--
    Name: {name}
    Brief Description: {brief_desc}
    Location: {location}
    Overall Rating: {overall_rating}
    Rating & Reviews: {reviews}

    --Output--
    Generate a clear, concise paragraph based on the data above, suitable for search and retrieval.
    """
)

GENERATE_RESTAURANT_DESCRIPTION_PROMPT = PromptTemplate(
    """
    You are a description generator for a restaurant in Jeju Island. 
    You will be provided the name, brief description, location, overall rating, and multiple reviews about the restaurant.
    Your task is to generate a **comprehensive, well-rounded** description that highlights key aspects of the restaurant, making it suitable for embedding and retrieval based on user queries.

    --Steps--
    1. Carefully review the provided restaurant information: name, description, location, and reviews.
    2. Generate a description as a **third-person narrator**, focusing on the restaurant’s offerings, ambiance, and what makes it stand out.
    3. Ensure that the description includes:
    - **Type of cuisine** and signature dishes (local specialties, fusion, international cuisine, etc.).
    - **Who the restaurant is ideal for** (families, couples, solo diners, large groups, foodies, etc.).
    - The restaurant’s **atmosphere** (casual, fine dining, cozy, modern).
    - **Proximity to key landmarks** or tourist spots in Jeju Island.
    - Guest experiences from reviews, focusing on the **quality of food, service, and overall dining experience**.
    - Any notable **value propositions** (affordability, unique offerings, famous chefs, etc.).

    --Data Provided--
    Name: {name}
    Brief Description: {brief_desc}
    Location: {location}
    Overall Rating: {overall_rating}
    Rating & Reviews: {reviews}

    --Output--
    Generate a clear, concise paragraph based on the data above, suitable for search and retrieval.
    """
)

GENERATE_ATTRACTION_DESCRIPTION_PROMPT = PromptTemplate(
    """
    You are a description generator for a tourist attraction in Jeju Island.
    You will be provided with the name, brief description, location, overall rating, and visitor reviews about the attraction.
    Your task is to generate a **comprehensive, engaging** description that highlights the key aspects of the attraction, making it suitable for embedding and retrieval based on user queries.

    --Steps--
    1. Carefully review the provided information: name, brief description, location, and reviews.
    2. Generate a description as a **third-person narrator**, focusing on the attraction’s features, significance, and what makes it special.
    3. Ensure that the description includes:
    - **Type of attraction** (natural landmark, cultural site, historical monument, adventure spot, etc.).
    - **What visitors can expect** (scenic views, activities, exhibits, wildlife, cultural experiences, etc.).
    - **Who the attraction is ideal for** (families, adventurers, history enthusiasts, nature lovers, etc.).
    - The attraction’s **location and proximity to key landmarks** or other popular places in Jeju Island.
    - Visitor experiences from reviews, emphasizing **what makes the visit memorable** (e.g., beauty, accessibility, uniqueness).
    - Any notable **special features** (e.g., seasonal events, guided tours, hidden gems, rare wildlife, etc.).

    --Data Provided--
    Name: {name}
    Brief Description: {brief_desc}
    Location: {location}
    Overall Rating: {overall_rating}
    Visitor Reviews: {reviews}

    --Output--
    Generate a clear, concise paragraph based on the data above, suitable for search and retrieval.
    """
)

GENERATE_ATTRACTION_SUITABLE_TIMES_PROMPT = PromptTemplate(
    """
    You are given the name of a tourist destination in Jeju along with its description, opening times, and public reviews. Based on this information, determine which times of day the location is suitable for a visit.

    Consider the following factors:
    - **Opening Times**: Ensure that the suggested times fall within the destination's opening hours (if available).
    - **Safety**: Avoid suggesting darker times like dawn and night, only suggest them if there are special or scenic views, but also take into account any potential safety concerns for visitors during these times.
    - **Heat**: Be cautious about suggesting times like noon and afternoon if the destination is in an open area and might be too hot.
    - **Scenic Views**: For scenic locations such as hilltops, beaches, cliffs, etc., prioritize dawn and evening as these times generally offer better views due to lighting and atmosphere.

    Output the suitable times of day as a continuous list, using only the following words: **dawn, morning, noon, afternoon, evening, night**. Separate each time with a comma "," and **do not include any additional text**.

    Here is the information about the destination:
    - Name: {dest_name}
    - Description: {dest_desc}
    - Opening Times: {opening_times}

    Output the suitable times:
    """
)



if GENERATE_DESC:
    # this will generate the descriptions for each location bassed on the detailed info obtained from Places API
    for category in tqdm(all_categories):
        json_file = os.path.join(DETAILED_DIR, f"{category}_detailed.json")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if not os.path.exists(DESCRIPTION_DIR):
            os.makedirs(DESCRIPTION_DIR)
        
        description_dict = {}
        for loc, loc_details in tqdm(data.items()):
            
            # name
            name = loc_details['name']
            
            # brief desc
            if "editorial_summary" in loc_details.keys():
                brief_desc = loc_details["editorial_summary"]["overview"]
            else:
                brief_desc = "Not Available"
            
            # location
            location = loc_details["formatted_address"]
            
            # rating
            if "rating" in loc_details.keys():
                overall_rating = f"{loc_details['rating']}/5"
            else:
                overall_rating = "Not Available"
            
            # reviews
            if "reviews" in loc_details.keys():
                if len(loc_details["reviews"]) > 0:
                    reviews = ""
                    for review in loc_details["reviews"]:
                        if "language" in review.keys():
                            if review['language'].lower() == "en":
                                reviews += f" - Rating: {review['rating']}/5. Comment: {review['text']} \n"
                else:
                    reviews = "Not Available"
            else:
                reviews = "Not Available"
            
            # print(name, brief_desc, location, overall_rating, reviews)
            
            if "hotels" in category.lower():
                template = GENERATE_HOTEL_DESCRIPTION_PROMPT
            elif "restaurant" in category.lower():
                template = GENERATE_RESTAURANT_DESCRIPTION_PROMPT
            elif  "tourist" in category.lower():
                template = GENERATE_ATTRACTION_DESCRIPTION_PROMPT
            else:
                raise NotImplementedError(f"Category: {category}")
            
            # llm call
            time.sleep(0.5)
            description = Settings.llm.complete(template.format(
                name=name,
                brief_desc=brief_desc,
                location=location,
                overall_rating=overall_rating,
                reviews=reviews
            ))
            description = str(description).strip()
            description_dict[str(loc)] = description
            
            print(f"{loc} : {description}")
            
            with open(os.path.join(DESCRIPTION_DIR, f"{category}_descriptions.json"), 'w') as f:
                json.dump(description_dict, f, indent=4)


if GENERATE_SUITABLE_TIMES:
    category = 'tourist_spots'
    with open(os.path.join(DESCRIPTION_DIR, f"{category}_descriptions.json"), 'r') as f:
        desc_json = json.load(f)
    with open(os.path.join(DETAILED_DIR, f"{category}_detailed.json"), 'r') as f:
        data_json = json.load(f)
    
    details_dict = {}
    for loc, loc_details in tqdm(data_json.items()):
        
        # name
        name = loc_details['name']
        desc = desc_json[str(loc)]
        
        if "opening_hours" in list(loc_details.keys()):
            if "weekday_text" in list(loc_details["opening_hours"].keys()):
                opening_times = loc_details["opening_hours"]["weekday_text"]
            else:
                opening_times = "Not Available"
        else:
            opening_times = "Not Available"

        # llm call
        time.sleep(0.5)
        suitable_times = Settings.llm.complete(GENERATE_ATTRACTION_SUITABLE_TIMES_PROMPT.format(
            dest_name=name,
            dest_desc=desc,
            opening_times=opening_times
        ))
        
        preset_times = ['dawn', 'morning', 'noon', 'afternoon', 'evening', 'night']
        suitable_times_list = []
        for t in preset_times:
            if t in str(suitable_times).strip().lower():
                suitable_times_list.append(t)
        
        print(f"{loc} : {suitable_times_list}")
        details_dict[str(loc)] = loc_details
        details_dict[str(loc)]["suitable_times_list"] = suitable_times_list
        
        with open(os.path.join(DETAILED_DIR, f"{category}_detailed.json"), 'w') as f:
            json.dump(details_dict, f, indent=4)


if GENERATE_VECTOR_INDEX:
    # This will embed the location name and descriptions into a vector database
    for category in tqdm(all_categories):
        with open(os.path.join(DESCRIPTION_DIR, f"{category}_descriptions.json"), 'r') as f:
            data = json.load(f)
        if category == "tourist_spots":
            with open(os.path.join(DETAILED_DIR, f"{category}_detailed.json"), 'r') as f:
                detailed_json = json.load(f)
        
        nodes = []
        for loc, loc_desc in data.items():
            if category == "tourist_spots":
                nodes.append(
                    TextNode(
                        text=str(loc), 
                        metadata={"description": str(loc_desc), "suitable_visiting_times": detailed_json[str(loc)]["suitable_times_list"]}, 
                        id=datetime.now().strftime('%Y%m%d%H%M%S%f')
                    )
                )
            else:
                nodes.append(
                    TextNode(text=str(loc), metadata={"description": str(loc_desc)}, id=datetime.now().strftime('%Y%m%d%H%M%S%f'))
                )
        
        uri = os.path.join(VECTOR_DB_DIR)
        if not os.path.exists(uri):
            os.makedirs(uri)
        
        start_time = time.time()
        vector_store = MilvusVectorStore(uri=os.path.join(uri, f"{category}.db"), dim=EMBED_MODEL_SIZE, overwrite=True)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=Settings.embed_model)
        end_time = time.time()
        
        print(f"[indexing] : {category} took {end_time - start_time} secs")


if TEST_VECTOR_INDEX:
    # user_query = "Get me a restaurant that has a nice view beside the ocean. Also, i need halal food!"
    # user_query = "i love korean food! especially bibimbap. get me one near the jeju international airport"
    user_query = "i love scenic views, get me one that is suitable for evening times!"
    similarity_top_k = 400
    # uri = os.path.join("locations", "descriptions_vector_store", "hotels.db")
    # uri = os.path.join("locations", "descriptions_vector_store", "restaurants.db")
    uri = os.path.join("locations", "descriptions_vector_store", "tourist_spots.db")
    
    vector_store = MilvusVectorStore(uri=uri, dim=EMBED_MODEL_SIZE, overwrite=False)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=[], storage_context=storage_context, embed_model=Settings.embed_model)
    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    nodes = retriever.retrieve(user_query)
    for node in nodes:
        print(node.text, node.metadata, node.score)

print("\nDONE.")