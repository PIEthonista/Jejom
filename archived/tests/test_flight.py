from tqdm import tqdm
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from tavily import TavilyClient
# from pymilvus.model.reranker import BGERerankFunction
from llama_index.core import VectorStoreIndex, PromptTemplate, Settings
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.llms.nvidia import NVIDIA
from llama_index.llms.groq import Groq
from llama_index.llms.upstage import Upstage
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.agent.openai import OpenAIAgent
from tests.prompts import contextualize_prompt, classify_prompt, qa_system_prompt, qa_prompt, flight_prompt

load_dotenv()

# Global wide models
Settings.llm = Groq(model='llama3-groq-70b-8192-tool-use-preview')

# Config
VERBOSE = False
CLASSES = [
    # 'accomodations',
    # 'tourist spots',
    # 'local ground transports',
    'flights',
    # 'eateries'
]
# CLASS_SPECIFIC_PROMPTS = {
#     'accomodations': "Name of the place, Location, Availability, Cost, etc.",
#     'tourist spots': "Name of the place, Location, Opening Times, Entrance Fees, etc.",
#     'local ground transports': "Transport Name, Transport ID, Transport Line, Cost, Scheduled times, Starting location, Ending location, Intermediary locations, etc.",
#     'flights': "Flight Company Name, Flight Number, Departure location, Destination location, Scheduled Time, Cost, Seats available, etc.",
#     'eateries': "Name of the place, Location, Opening Times, Menu, What is it famous for, Entrance Fees, Is booking required beforehand, etc."
# }


def tavily_browser_tool(input: str) -> str:
    """
    browses the internet for relevant information about the input and returns it.
    """
    tavily = TavilyClient()
    info = tavily.search(query=input, search_depth="advanced")
    print(">>> tavily", str(info))
    return str(info)


agents = {}
for CLASS in tqdm(CLASSES):
    # 1 tavily tool for each class agent
    # add other tools if needed
    class_agent_tools = [
        FunctionTool.from_defaults(
            fn=tavily_browser_tool,
            tool_metadata=ToolMetadata(
                name='web_browser_tool',
                description=f"Useful for browsing answers for questions related to specific aspects of {CLASS}."
            )
        )
    ]

    # one agent for each doc
    agent = OpenAIAgent.from_tools(
        class_agent_tools,
        llm=Settings.llm,
        verbose=VERBOSE,
        system_prompt=flight_prompt
        # system_prompt=f"""\
        #     You are a specialized agent designed to answer queries about {CLASS}. 
        #     YOU ARE TO answer in a detailed manner, which are the {CLASS_SPECIFIC_PROMPTS[CLASS]}.
        #     You must ALWAYS use at least one of the tools provideed when answering a question.
        #     DO NOT rely on prior knowledge. 
        #     You must provide answers that are detailed and logical.
        #     """
    )


    agents[CLASS] = agent


# wrap each class agent as tools
all_tools = []
for CLASS in tqdm(CLASSES):
    doc_agent_as_tool = QueryEngineTool(
        query_engine=agents[CLASS],
        metadata=ToolMetadata(
            name=f"tool_{CLASS}",
            description=(
                f"This tool helps you search information about {CLASS}."
                f"Use this tool if you want to answer any questions about {CLASS}."
            )
        )
    )
    all_tools.append(doc_agent_as_tool)

# # index tools (doc agents) as objects
# obj_index = ObjectIndex.from_objects(
#     all_tools,
#     index_cls=VectorStoreIndex
# )

# top agent that we interact with
top_orchestrator_agent = OpenAIAgent.from_tools(
    tools=all_tools,
    # tool_retriever=obj_index.as_retriever(similarity_top_k=len(CLASSES)),
    system_prompt=qa_system_prompt,
    verbose=VERBOSE
)


class ChatMessage(BaseModel):
    role: str
    content: str

    def __str__(self):
        return f"{self.role}: {self.content}"

class RAGQueryEngine(CustomQueryEngine):
    llm: OpenAI   # base class extended by Groq, etc.
    top_orchestrator_agent: OpenAIAgent
    qa_prompt: PromptTemplate
    qa_system_prompt: str
    contextualize_prompt: PromptTemplate
    classify_prompt: PromptTemplate
    chat_history: List[ChatMessage] = []


    def classify_query(self, query_str: str) -> str:
        if len(self.chat_history) == 0:
            return "relevant"
        else:
            classify_prompt_text = self.classify_prompt.format(query_str=query_str)
            response = self.llm.complete(classify_prompt_text)
            classification = response.text.strip().lower()
            return classification

    def contextualize_query(self, query_str: str) -> str:
        if len(self.chat_history) == 0:
            return query_str
        else:
            chat_history_str = "\n".join([str(msg) for msg in self.chat_history])
            contextualize_prompt_text = self.contextualize_prompt.format(
                chat_history=chat_history_str,
                latest_message=query_str
            )
            response = self.llm.complete(contextualize_prompt_text)
            return response.text.strip() if hasattr(response, 'text') else response.choices[0].text.strip()

    def custom_query(self, query_str: str):
        contextualized_query = self.contextualize_query(query_str)
        print(f">>> Contextualized Query: {contextualized_query}") if VERBOSE else None
        classification = self.classify_query(contextualized_query)

        if classification == "irrelevant":
            updated_prompt = self.qa_prompt.format(context_str="", query_str=contextualized_query)
            response = self.llm.complete(updated_prompt)
            # return None
            
        elif classification == "relevant":
            # response = self.top_orchestrator_agent.query(contextualized_query)
            try:
                response = self.top_orchestrator_agent.query([
                    {"role": "system", "content": str(self.qa_system_prompt)},
                    {"role": "user", "content": str(contextualized_query)}
                ])
            except:
                response = self.top_orchestrator_agent.query(contextualized_query)

        self.chat_history.append(ChatMessage(role="user", content=query_str))
        self.chat_history.append(ChatMessage(role="assistant", content=str(response)))

        return str(response)


query_engine = RAGQueryEngine(
    llm=Settings.llm,
    top_orchestrator_agent=top_orchestrator_agent,
    qa_prompt=qa_prompt,
    qa_system_prompt=qa_system_prompt,
    contextualize_prompt=contextualize_prompt,
    classify_prompt=classify_prompt
)



# while True:
#     user_query = input("👱🏻 USER: ")
#     response = query_engine.custom_query(user_query)
#     print("\n🤖 JEJOM: ", response, "\n")

user_query = "Can you help me look for flights from Malaysia to Jeju Island? preferrably flying at night, on 1 September 2024"
response = agents['flights'].query(user_query)
print(response)
