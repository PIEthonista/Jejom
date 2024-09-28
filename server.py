from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import time
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.upstage import Upstage
from llama_index.embeddings.upstage import UpstageEmbedding
from image_generator import get_place_img
from scripts.script import ScriptGenerator, Translator
from utils import read_file


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://10.168.105.128:5000", "*"]}})

load_dotenv()
Settings.llm = Upstage(model='solar-pro')
Settings.embed_model=UpstageEmbedding(model='solar-embedding-1-large')

from pipelinev2 import PipelineV2
pipeline = PipelineV2(
    embed_model_size         = 4096,
    accomodations_sim_top_k  = 500,
    restaurants_sim_top_k    = 500,
    tourist_spots_sim_top_k  = 500,
    accomodations_vec_db_uri = os.path.join('locations', 'descriptions_vector_store', 'hotels.db'),
    restaurants_vec_db_uri   = os.path.join('locations', 'descriptions_vector_store', 'restaurants.db'),
    tourist_spots_vec_db_uri = os.path.join('locations', 'descriptions_vector_store', 'tourist_spots.db'),
    accomodations_json       = os.path.join('locations', 'detailed', 'hotels_detailed.json'),
    restaurants_json         = os.path.join('locations', 'detailed', 'restaurants_detailed.json'),
    tourist_spots_json       = os.path.join('locations', 'detailed', 'tourist_spots_detailed.json')
)



@app.route('/')
def home():
    return 'Hello World'


@app.route('/generate_script', methods=['POST'])
def generate_script():
    try:
        start_time = time.time()
        data = request.json

        # Extract the required fields from the JSON body
        characters_num = data.get('characters_num')
        cafe_name = data.get('cafe_name')
        cafe_environment = data.get('cafe_environment')

        # Ensure the required fields are provided
        if not characters_num or not cafe_name or not cafe_environment:
            return jsonify({"error": "Missing required fields"}), 400

        # Create an instance of ScriptGenerator
        script_generator = ScriptGenerator(
            characters_num=characters_num,
            cafe_name=cafe_name,
            cafe_environment=cafe_environment
        )

        # Run tasks and generate script
        output_json_path, cafe_name = script_generator.run_tasks()

        translator = Translator(api_key=os.getenv('UPSTAGE_API_KEY'))

        
        input_directory = "output_folder"

        input_file_path = os.path.join(input_directory, "script.json")

        output_file_path = os.path.join(input_directory, "translated_output.json")

        translator.translate_and_save(input_file_path, output_file_path)
        print(f"[Generate Trip took {time.time() - start_time} secs]")


        return jsonify({
            "eng_script": read_file(output_json_path, "json"),
            "kor_script": read_file(output_file_path, "json"),
        })
    except Exception as e:
        return jsonify({"error": f"Error generating script: {e}"})

app.route('/script', methods=['POST'])
def display_script():

    input_directory = "output_folder"

    input_file_path = os.path.join(input_directory, "output.json")

    with open(input_file_path, 'r') as file:
        script = json.load(file)
        script_planner = script["Script Planner"]
        character_designer = script["Character Designer"]
        script_writer = script["Script Writer"]
        clue_generator = script["Clue Generator"]
        player_instruction_writer = script["Player Instruction Writer"]
        title = script["Title"]
        duration = script["Duration"]
    
    return jsonify({'Script Planner': script_planner, "Character Designer": character_designer,"Script Writer": script_writer, "Clue Generator":clue_generator, "Player Instruction Writer":player_instruction_writer, "Title":title, "Duration":duration})

    

@app.route('/translate', methods=['POST'])
def translate():
    
    input_directory = "output_folder"

    output_file_path = os.path.join(input_directory, "translated_output.json")


    try:
        # Call the translate_and_save method of the Translator class
        
        with open(output_file_path, 'r') as file:
            script = json.load(file)
            script_planner = script["Script Planner"]
            character_designer = script["Character Designer"]
            script_writer = script["Script Writer"]
            clue_generator = script["Clue Generator"]
            player_instruction_writer = script["Player Instruction Writer"]
            title = script["Title"]
            duration = script["Duration"]
        
        return jsonify({'Script': script_planner, "Character Designer": character_designer,"Script Writer": script_writer, "Clue Generator":clue_generator, "Player Instruction Writer":player_instruction_writer, "Title":title, "Duration":duration}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/check_init_input', methods=['POST'])
def check_init_input():
    query = request.form.get('query')
    print("check_init_input: ", query)
    check_result_dict = pipeline.check_query_detail(query=str(query))
    return jsonify(check_result_dict)

@app.route('/generate_trip', methods=['POST'])
def generate_trip():
    user_query = request.form.get('query')
    user_properties = request.form.get('user_props')
    mode = request.form.get('mode')  # test
    
    print("generate_trip: ", user_query, user_properties)
    print(str(mode).lower().strip())
    
    if "test" in str(mode).lower().strip():
        with open('sample_usages/generate_trip_v2.json', 'r') as file:
            trip_dict = json.load(file)
            trip_dict = trip_dict["data"]
    else:
        start_time = time.time()
        trip_dict = pipeline.generate_trip(
            end_user_specs=str(user_properties), 
            end_user_query=str(user_query)
        )
        trip_dict['thumbnail'] = get_place_img(trip_dict['title'])
        print(f"[Generate Trip took {time.time() - start_time} secs]")
    return jsonify({'data': trip_dict})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', use_reloader=False)
    # app.run(debug=True, host='0.0.0.0', use_reloader=False)