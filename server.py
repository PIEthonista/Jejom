from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import time
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.upstage import Upstage
from llama_index.embeddings.upstage import UpstageEmbedding



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://10.168.105.128:5000", "*"]}})

load_dotenv()
Settings.llm = Upstage(model='solar-1-mini-chat')
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
        with open('sample_usages/generate_trip.json', 'r') as file:
            trip_dict = json.load(file)
            trip_dict = trip_dict["data"]
    else:
        start_time = time.time()
        trip_dict = pipeline.generate_trip(
            end_user_specs=str(user_properties), 
            end_user_query=str(user_query)
        )
        print(f"[Generate Trip took {time.time() - start_time} secs]")
    return jsonify({'data': trip_dict})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', use_reloader=False)
    # app.run(debug=True, host='0.0.0.0', use_reloader=False)