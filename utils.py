import json

def read_file(file_path: str, file_type: str):
    if file_type == "json":
        with open(file_path, 'r') as file:
            return json.load(file)
        
    elif file_type == "txt":
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()