import os
from typing import List
import requests
from utils import generate_upstage_response
from dotenv import load_dotenv; load_dotenv()

def parse_chunk_string(input_string: str) -> List[dict]:
    chunks = [chunk.strip() for chunk in input_string.split(';') if chunk.strip()]
    parsed_content = []

    for chunk in chunks:
        if '>' in chunk:
            content, keywords = chunk.split('>')
            content = content.strip()
            keywords = keywords.strip()
            
            parsed_content.append({
                "content": content,
                "keywords": keywords
            })

    return parsed_content

def generate_keyword_from_script(script: str):
    chunking_keywords_prompt = '''
You are given a script. You need to chunk the script into different sections and generate one keyword to represent the main highlights of each section. 
Yuu should generate based on the following guidelines:
- You should chunk the script into different sections based on the content.
- The keyword should be representing the main highlights of the section in a general term.
- Only use GENERAL, NON-specific keywords. For example, use "Ancient city" instead of "Ancient city of Seongju", use "mountain" instead of "Mount Halla". Use "Island" instead of "Jeju Island".
- Generate only 1 keyword for each section.
- DO NOT generate any additional explanation for your response.
- You should include all the original script in the response, with the keyword following each of the chunks after a ">" symbol.
- Your response should strictly follow the format of the example below.
<Content of chunk 1> > <Keyword for content of chunk 1>;
<Content of chunk 2> > <Keyword for content of chunk 2>;

For example,
Script:
The murder mystery unfolds in the ancient city of Seongju, located on the outskirts of Jeju Island. This city, once a thriving hub of trade and culture, has fallen into decay, its grand palaces and temples now reclaimed by the surrounding jungle. The city's history is steeped in the legends of the island's guardian spirits, the Dokkaebi, and the mythical sea serpent, the Yeouija. The story begins with the discovery of a murdered body in the heart of the city, near the ruins of the ancient royal palace. The victim, a wealthy merchant named Lee, was known for his ruthless business practices and exploitation of the island's resources. His body is found with a dagger plunged into his heart, adorned with intricate carvings of the Dokkaebi and the Yeouija.
Response:
The murder mystery unfolds in the ancient city of Seongju, located on the outskirts of Jeju Island. This city, once a thriving hub of trade and culture, has fallen into decay, its grand palaces and temples now reclaimed by the surrounding jungle. The city's history is steeped in the legends of the island's guardian spirits, the Dokkaebi, and the mythical sea serpent, the Yeouija. > ancient city;
The story begins with the discovery of a murdered body in the heart of the city, near the ruins of the ancient royal palace. The victim, a wealthy merchant named Lee, was known for his ruthless business practices and exploitation of the island's resources. His body is found with a dagger plunged into his heart, adorned with intricate carvings of the Dokkaebi and the Yeouija. > murder;
Script:
{script}
Response:

'''
    input_text = chunking_keywords_prompt.format(script=script)
    response = generate_upstage_response(input_text)

    return parse_chunk_string(response)


def generate_keyword_image(scripts: List[dict]):
    img_urls = []
    for part in scripts:
        img_urls.append(get_pexel_img(part["keywords"]))

    return {
        "content": "<image>".join([part["content"] for part in scripts]),
        "images": img_urls
    }

def get_pexel_img(query: str):
    url = f"https://api.pexels.com/v1/search?query={query}"
    api_key = os.getenv('PEXELS_API_KEY')
    headers = {
        "Authorization": api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36",
        "Referer": "https://www.pexels.com/"
    }
    
    response = requests.get(url, headers=headers)
    # print(f"{query}: {response.json()['photos']}")
    return response.json()['photos'][0]['src']['original']   

def get_place_img(title: str):
    place_name = extract_name_from_title(title)
    return get_pexel_img(place_name)

def extract_name_from_title(title: str):
    # extract_title_prompt = '''
    # You are given a title of a travel itenerary. You need to extract the name of the travel destination from the title.
    # You should generate based on the following guidelines:
    # - ONLY extract and output the name of the travel destination from the title.
    
    # For example,
    # Title: Exploring Jeju's Natural Wonders: A Week-Long Adventure
    # Response: Jeju

    # Title: Discovering the Hidden Gems of Seoul: A 5-Day Itinerary
    # Response: Seoul

    # Title: Exploring the Temples of Kyoto: A Cultural Journey
    # Response: Kyoto

    # Title: {title}
    # Response: 
    # '''
    # input_text = extract_title_prompt.format(title=title)
    # response = generate_upstage_response(input_text)

    # return response
    return 'Jeju'

def add_images_to_script(script: dict):
    keyword_dict = generate_keyword_from_script(script['Script Planner'])
    res = generate_keyword_image(keyword_dict)

    script['Script Planner'] = res['content']
    script['images'] = res['images']
    
    return script

if __name__ == "__main__":
    print(get_place_img("Jeju"))