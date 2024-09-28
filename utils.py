import json
import os
import re
from llama_index.llms.upstage import Upstage
from llama_index.core.llms import ChatMessage

def read_file(file_path: str, file_type: str):
    if file_type == "json":
        with open(file_path, 'r') as file:
            return json.load(file)
        
    elif file_type == "txt":
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
def parse_output(output: str, fields: list[str]) -> dict:
    parsed_data = {}
    for field in fields:
        pattern = re.compile(rf"{field}-\s*(.*?)(?=\n\S|$)", re.DOTALL)
        
        match = pattern.search(output)
        parsed_data[field] = match.group(1).strip() if match else None
    
    return parsed_data

def generate_upstage_response(input: str):
    llm = Upstage(api_key=os.getenv('UPSTAGE_API_KEY'))
 
    response = llm.complete(
        prompt=input,
    )
    
    return str(response)

def extract_photo_reference(trip: dict):
    for key, value in trip.items():
        if not key in ['accomodations', 'destinations']:
            continue
        for entry in value:
            if not isinstance(entry['Photos'], list):
                continue
            photo_refs = []
            for photo in entry['Photos']:
                photo_refs.append(photo['photo_reference'])
            entry['Photos'] = photo_refs
    return trip

# if __name__ == '__main__':
#     trip_dict = read_file('sample_usages/generate_trip.json', 'json')['data']
#     print(extract_photo_reference(trip_dict))