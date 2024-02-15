#APIキーを参照するためのモジュール
import openai
from dotenv import load_dotenv
import os

#sqlite3のバージョン補完(現状のローカル開発環境では修正不可)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


#プロンプト定義用モジュール
from langchain import PromptTemplate

#対話履歴保持用モジュール
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

#OPENAIモデル+検索モデル
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

#ベクトルストア作成用モジュール
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb

#UI用ライブラリ
import streamlit as st

#UI
st.title("QA")

#.envからOpenAIのAPI Keyを参照
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#ベクトルストアの参照パス
PRESIST_PATH = "./vectorstore"

#ベクトルストア作成時と同様のEmbeddings
embeddings = OpenAIEmbeddings()

#作成済みベクトルストア(ChromaDB)の参照
client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=PRESIST_PATH, 
        anonymized_telemetry=False
        )

vectorstore = Chroma(
        collection_name="langchain_store",
        embedding_function=embeddings,
        client_settings=client_settings,
        persist_directory=PRESIST_PATH,
        )

retriever = vectorstore.as_retriever()

#回答を生成するモデル(Model)
model = ChatOpenAI(
        temperature=0.5,
        model_name=os.environ["OPENAI_API_MODEL"],
        )

#対話履歴を初期化(内部)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        )

memory = st.session_state.memory

chain = ConversationalRetrievalChain.from_llm(
                    llm=model,
                    retriever=retriever,
                    memory=memory,
                    )

#プロンプト設定
# template = """
# あなたは夏目漱石のこころの主人公です。
# ##以下の##情報をもとに、質問に対して、主人公になりきって回答してください。
# ##情報: {context}


# ##質問: {question}
# 回答:"""

# prompt = PromptTemplate(
#     template=template,
#     input_variables=["context", "question"],
# )

#対話履歴の初期化(UI用)
if "messages" not in st.session_state:
    st.session_state.messages = []

#UI用の対話履歴を表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("質問をどうぞ。")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Tinking..."):
            response = chain(
            {"question": prompt}
        )
            st.markdown(response["answer"])

    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

print(memory)