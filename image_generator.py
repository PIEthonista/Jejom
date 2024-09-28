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
        "img_urls": img_urls
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
    return response.json()['photos'][0]['url']   

def get_place_img(title: str):
    place_name = extract_name_from_title(title)
    return get_pexel_img(place_name)

def extract_name_from_title(title: str):
    extract_title_prompt = '''
    You are given a title of a travel itenerary. You need to extract the name of the travel destination from the title.
    You should generate based on the following guidelines:
    - ONLY extract and output the name of the travel destination from the title.
    
    For example,
    Title: Exploring Jeju's Natural Wonders: A Week-Long Adventure
    Response: Jeju

    Title: Discovering the Hidden Gems of Seoul: A 5-Day Itinerary
    Response: Seoul

    Title: Exploring the Temples of Kyoto: A Cultural Journey
    Response: Kyoto

    Title: {title}
    Response: 
    '''
    input_text = extract_title_prompt.format(title=title)
    response = generate_upstage_response(input_text)

    return response

if __name__ == '__main__':
    input_str = "The murder mystery unfolds in the ancient city of Seongju, located on the outskirts of Jeju Island. This city, once a thriving hub of trade and culture, has fallen into decay, its grand palaces and temples now reclaimed by the surrounding jungle. The city's history is steeped in the legends of the island's guardian spirits, the Dokkaebi, and the mythical sea serpent, the Yeouija.\n\nThe story begins with the discovery of a murdered body in the heart of the city, near the ruins of the ancient royal palace. The victim, a wealthy merchant named Lee, was known for his ruthless business practices and exploitation of the island's resources. His body is found with a dagger plunged into his heart, adorned with intricate carvings of the Dokkaebi and the Yeouija.\n\nThe four key characters are:\n\n1. Detective Kim, a seasoned investigator with a deep understanding of Jeju Island's culture and mythology.\n2. Soo, a local historian and archaeologist who holds the key to the city's hidden secrets.\n3. Hye-won, a young woman with a mysterious past, connected to the island's guardian spirits.\n4. Joon, a former soldier turned bodyguard, tasked with protecting Hye-won from the dangerous forces that seek to exploit her powers.\n\nDetective Kim is called to the scene of the crime and immediately senses that this is no ordinary murder. The intricate carvings on the dagger suggest a deep understanding of Jeju Island's mythology, and the victim's past makes it difficult to determine who could have committed such a heinous act.\n\nAs Detective Kim delves deeper into the investigation, he enlists the help of Soo, who provides valuable insights into the city's history and the legends of the Dokkaebi and the Yeouija. Together, they uncover a series of clues that lead them to Hye-won, a woman with a mysterious connection to the island's guardian spirits.\n\nHye-won's powers have attracted the attention of a powerful criminal organization, led by a man known only as the Serpent. The organization seeks to harness Hye-won's powers to control the Dokkaebi and the Yeouija, using them to gain control over the island's resources and its people.\n\nAs Detective Kim and Soo race to solve the murder and protect Hye-won, they are aided by Joon, who has sworn to protect her from the dangerous forces that seek to exploit her powers. Together, the four characters must navigate a web of secrets, betrayal, and danger as they unravel the truth behind the murder and the dark forces that threaten Jeju Island.\n\nThe story culminates in a thrilling confrontation with the Serpent and his organization, as the four characters must use their unique skills and knowledge to defeat the forces of evil and restore peace to the island. Along the way, they uncover the true nature of the Dokkaebi and the Yeouija, and the crucial role they play in the balance of power on Jeju Island.\n\nIn the end, the murder mystery is solved, and the four characters are forever changed by their experiences. The city of Seongju, once a symbol of decay and despair, is reborn as a beacon of hope and unity, its people united in their commitment to protect the island's rich cultural heritage and its mythical guardians.",

    keyword_dict = generate_keyword_from_script(input_str)
    res = generate_keyword_image(keyword_dict)

    print(res)