import os
from datetime import datetime

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

os.environ["GOOGLE_API_KEY"] = st.secrets["API_KEY"]
LLM_MODEL = "models/gemini-1.5-flash-exp-0827"  # "gemini-1.5-flash-8b-exp-0827"
EMBEDDINGS_MODEL = "models/text-embedding-004"
MODEL_TEMPERATURE = 0

DOCS = "Documents/FAQ.txt"
PERSIST_DIRECTORY = "Documents/embedding_db"
CURRENT_DATE = datetime.now().strftime("%d %B %Y")


class QASplitter(TextSplitter):
    def split_text(self, text):
        return [
            "Q: " + split.replace("\n", "").strip()
            for split in text.split("Q:")
            if split.strip()
        ]


def load_and_split_document(file_path):
    loader = TextLoader(file_path)
    docs = loader.load()

    qa_splitter = QASplitter()
    qa_splits = qa_splitter.split_documents(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
    final_splits = []

    for qa_split in qa_splits:
        if len(qa_split.page_content) > 2500:
            final_splits.extend(text_splitter.split_documents([qa_split]))
        else:
            final_splits.append(qa_split)

    return final_splits


def get_vectorstore(persist_directory=PERSIST_DIRECTORY):
    embedding = GoogleGenerativeAIEmbeddings(model=EMBEDDINGS_MODEL)

    if os.path.exists(persist_directory):
        print("Loading existing vectorstore...")
        return Chroma(persist_directory=persist_directory, embedding_function=embedding)

    print("Creating new vectorstore...")
    documents = load_and_split_document(DOCS)
    return Chroma.from_documents(
        documents=documents, embedding=embedding, persist_directory=persist_directory
    )


template = """
You are 2PotsGPT🤖, a helpful and humorous expert on the Two-Pot System. 

The Two-Pot Retirement system divides contributions into a "Savings Pot" (one-third, accessible before retirement) 
and a "Retirement Pot" (two-thirds, preserved for retirement income), promoting financial security and long-term savings effective on 01 September 2024 in South Africa.

Highlight the required and important parts in a markdown.

Use context as your knowledge base and if the user question is out of context ask for clarification.

<Current Date>
{current_date}
</Current Date>

<Chat History>
{chat_history} 
</Chat History>

<Context (FAQ)>
{context}
</Context (FAQ)>

<User Question>
{question}
</User Question>

**Helpful answer: **
""".replace("{current_date}", CURRENT_DATE)

prompt_template = PromptTemplate.from_template(template)


def initialize_conversation_chain():
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=MODEL_TEMPERATURE,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )

    vectorstore = get_vectorstore()
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
    )

    return conversation_chain