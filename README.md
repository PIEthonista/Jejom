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