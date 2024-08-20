# Jejom

## Env Setup

- <code>conda create -n jejom python=3.10</code>
- <code>pip install llama-index</code>
- <code>pip install llama-index-llms-upstage</code>
- <code>pip install llama-hub</code>
- <code>pip install tavily-python==0.2.8</code>
- <code>pip install llama-index-tools-tavily-research</code>


##API Keys
Create a file named .env at the same directory level as this README.md, and define the below:
TAVILY_API_KEY = "get_your_tavily_api_key"
UPSTAGE_API_KEY = "get_your_upstage_api_key"

##Python: Conda ENV Setup
conda env create -f environment.yml


##Run solution 
conda activate jejom-llama 
python scripts/scripts.py






<!-- 
- conda create -n jejom_lc python=3.10
- conda acitvate jejom_lc
- conda install langchain
- pip install tavily-python==0.2.8
- pip install langchain-openai
- pip install langchain_openai
- pip install beautifulsoup4
- pip install faiss-cpu
- pip install langchainhub
- pip install "langserve[all]"
- pip install numexpr
- pip install flask
- pip install flask_cors
- pip install openai
- pip install pandas
- pip install langchain-groq -->
