## API Keys
Create a file named <code>.env</code> at the same directory level as this <code>README.md</code>, and define the below:
- <code>GROQ_API_KEY = "get_your_groq_api_key_<a href='https://console.groq.com/keys'>here</a>"</code>
- <code>TAVILY_API_KEY = "get_your_tavily_api_key_<a href='https://app.tavily.com/sign-in'>here</a>"</code>
- <code>UPSTAGE_API_KEY = "get_your_upstage_api_key_<a href='https://developers.upstage.ai/docs/getting-started/quick-start'>here</a>"</code>


<br/>

## Python: Conda ENV Setup
- <code>conda env create -f environment.yml</code>

Due to different compatibility from machine to machine, if you happen to face any conflicts while resolving the above environment, please head to <code>requirements.txt</code> and install the dependencies one by one.



<br/>

## Running the Backend
- <code>conda activate jejom-llama</code>
- <code>python server.py</code>


<br/>

## Upstage API Usage
Upstage API is being used int th below sections:
- Upstage Query Embedding Model: This model is used to embed and cache information that has been searched and generated. This is to prevent the process of re-generation and research which is computationaly expensive. The cache functions just like VectorStores where cache information can be retrieved via cosine similarity computation.
- Upstage Chat Model: This pipeline heavily relies on nested Multi-Orchestrated Agents. At the heart of it all lies the most important LLM Model for directing the whole pipeline. Both Upstage API's LLM or LLMs from other distributors that extend the OpenAI class and support function calling can be used here. 