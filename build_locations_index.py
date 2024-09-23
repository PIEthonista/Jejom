import os
import json
import math
import time
import random
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from datetime import datetime, timedelta

# https://github.com/mangiucugna/json_repair
from json_repair import repair_json  # LLM JSON output fixing if necessary

from tavily import TavilyClient
from llama_index.core import PromptTemplate, Settings
# from llama_index.llms.openai import OpenAI
# from llama_index.llms.nvidia import NVIDIA
from llama_index.llms.groq import Groq
from llama_index.llms.upstage import Upstage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.upstage import UpstageEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.agent.openai import OpenAIAgent

load_dotenv()
DETAILED_DIR = os.path.join("locations", "detailed")
DESCRIPTION_DIR = os.path.join("locations", "descriptions")
GENERATE_DESC = True     # generates description for vector db indexing, each location costs an LLM call
GENERATE_VECTOR_INDEX = False   # takes each location as a Document object and embeds it into the vector space alongside its description

TO_OMIT = ['.DS_Store']  # mac cache

# Settings.llm = Groq(model='llama3-groq-70b-8192-tool-use-preview')
Settings.llm = Upstage(model='solar-1-mini-chat')
Settings.embed_model=UpstageEmbedding(model='solar-embedding-1-large')

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


if GENERATE_DESC:
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
