from flask import Flask, request, session, render_template, jsonify
from langchain.document_loaders import RecursiveUrlLoader
from langchain.document_loaders import TextLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory, )
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
import os
import openai
import tempfile
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document
# Generate a random hexadecimal string, 24 bytes long
secret_key = os.urandom(24).hex()

app = Flask(__name__)
app.secret_key = secret_key  # Set a secret key for session management


@app.route('/', methods=['GET'])
def index():
    return "Welcome to bot."


def configure_retriever(api):
    loader = RecursiveUrlLoader(
        "https://en.wikipedia.org/wiki/Personal_finance")
    raw_documents = loader.load()
    docs = Html2TextTransformer().transform_documents(raw_documents)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=api)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def configure_retriever2(data_personal_finance, api):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = [Document(page_content=x)
                 for x in text_splitter.split_text(data_personal_finance)]
    embeddings = OpenAIEmbeddings(openai_api_key=api)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


@app.route('/ai_response', methods=['POST'])
def ai_response_endpoint():
    # Get the message from the POST request
    data = request.json
    query = data.get('query')
    api_key = data.get('api_key')
    history = data.get('history')
    personal_data = data.get('personal_data')
    print("query", query)
    openai.api_key = api_key
    financeTool = create_retriever_tool(
        configure_retriever(api_key),
        "search_personal_finance_docs",
        "Searches and returns documents regarding personal finance and money management.",
    )

    myFinanceData = create_retriever_tool(
        configure_retriever2(personal_data, api_key),
        "get_live_personal_finance_data",
        "Searches and returns documents related to current user's precise, live financial data to utilize for any personalization.",
    )

    tools = [financeTool, myFinanceData]

    llm = ChatOpenAI(temperature=0, streaming=True,
                     model="gpt-4", openai_api_key=api_key)
    message = SystemMessage(content=(
        "You are a helpful chatbot who is tasked with answering questions about personal finance and money management strategy or budget."
        "Unless otherwise explicitly stated, it is probably fair to assume that questions are about personal finance."
        "Personalize responses as best as you can with understanding of user's current financial position using their live data. Live data has cumulative spend, spend by month, spend by online vs in store, top spend categories and finally every transacation within the last year."
        "If there is any ambiguity, instead of giving response, ask a follow up question to give better results. Ask unlimited follow up questions if necessary for a quality response."
        "Your response should be all in one simple plain text paragraph. Adjust format accordingly. No special characters or line breaks/formatters, plain text."
    ))
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
    )
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )
    memory = AgentTokenBufferMemory(llm=llm, return_messages=True)
    starter_message = "Your personal finance AI assistant! Ask anything related to your finances to get started."
    msg = []
    msg.append(AIMessage(content=starter_message))
    response = agent_executor(
        {
            "input": query,
            "history": msg,
        },
        include_run_info=True,
    )

    memory.chat_memory.add_message(response["output"])
    return jsonify(response["output"])


if __name__ == '__main__':
    app.run(debug=True)
