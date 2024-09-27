from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.upstage import UpstageEmbedding



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://10.168.105.128:5000", "*"]}})

load_dotenv()
MAX_JSON_TRY = 3
CHECK_MATCH_FROM_CACHE_TOP_K = 2
Settings.llm = Groq(model='llama3-groq-70b-8192-tool-use-preview')
Settings.embed_model=UpstageEmbedding(model='solar-embedding-1-large')

from pipeline import Pipeline
pipeline = Pipeline(USE_CACHE=True,
                    accomodation_cache_file="./cache/accomodations.json", 
                    destination_cache_file = "./cache/destinations.json")



@app.route('/')
def home():
    return 'Hello World'

@app.route('/check_init_input', methods=['POST'])
def check_init_input():
    query = request.form.get('query')
    print("check_init_input: ", query)
    check_result_dict = pipeline.check_query_detail(query=str(query))
    return jsonify(check_result_dict)

@app.route('/get_accomodations', methods=['POST'])
def get_accomodations():
    query = request.form.get('query')
    num_accomodations = request.form.get('num_accomodations')
    print("get_accomodations: ", query, num_accomodations)
    accomodations_list_of_dicts = pipeline.get_accomodations_json(query=str(query), 
                                                                  num_accomodations=int(num_accomodations),
                                                                  max_json_try=MAX_JSON_TRY,
                                                                  check_match_from_cache_top_k=CHECK_MATCH_FROM_CACHE_TOP_K)
    return jsonify({"data": accomodations_list_of_dicts})

@app.route('/get_destinations', methods=['POST'])
def get_destinations():
    query = request.form.get('query')
    num_destinations = request.form.get('num_destinations')
    print("get_destinations: ", query, num_destinations)
    destinations_list_of_dicts = pipeline.get_destinations_json(query=str(query), 
                                                                  num_spots=int(num_destinations),
                                                                  max_json_try=MAX_JSON_TRY,
                                                                  check_match_from_cache_top_k=CHECK_MATCH_FROM_CACHE_TOP_K)
    return jsonify({"data": destinations_list_of_dicts})

@app.route('/generate_trip', methods=['POST'])
def generate_trip():
    user_query = request.form.get('query')
    user_properties = request.form.get('user_props')
    mode = request.form.get('mode')  # test
    
    print("generate_trip: ", user_query, user_properties)
    print(str(mode).lower().strip())
    
    if "test" in str(mode).lower().strip():
        with open('sample_usages/generate_trip.json', 'r') as file:
            trip_dict = json.load(file)
            trip_dict = trip_dict["data"]
    else:
        trip_dict = pipeline.generate_trip(end_user_specs=str(user_properties),
                                        end_user_final_query=str(user_query)+str(user_properties),
                                        max_json_try=MAX_JSON_TRY,
                                        check_match_from_cache_top_k=CHECK_MATCH_FROM_CACHE_TOP_K)
    return jsonify({'data': trip_dict})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')