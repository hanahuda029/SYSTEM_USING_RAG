#APIキーを参照するためのモジュール
import openai
from dotenv import load_dotenv
import os

#sqlite3のバージョン補完(現状のローカル開発環境では修正不可)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#.envからOpenAIのAPI Keyを参照
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#ベクトルストア作成用モジュール
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb

#Embeddings
embeddings = OpenAIEmbeddings()

#データベースの設定
CHUNK_SIZE = 900
CHUNK_OVERLAP = 0
PRESIST_PATH = "./vectorstore"
# MODEL_NAME = "text-embedding-3-large"
COLLECTION_NAME = "langchain"

#文書の読み込み(.txt用)
loader = DirectoryLoader(
    "./documents/", 
    glob="**/kokoro.txt", 
    loader_cls=TextLoader
)
data = loader.load()

print(data)

#ベクトル変換用の文書分割
text_splitter = CharacterTextSplitter(
    separator='\n\n',
    chunk_size=900,
    chunk_overlap=0,
    length_function=len
)
documents = text_splitter.create_documents([doc.page_content for doc in data])

print(str(documents))

#ベクトルストア(ChromaDB)の設定
client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=PRESIST_PATH, 
        anonymized_telemetry=False
    )

#インスタンス作成
vectorstore = Chroma(
    collection_name="langchain_store",
    embedding_function=embeddings,
    client_settings=client_settings,
    persist_directory=PRESIST_PATH,
)

#ベクトルストアに文書を追加
vectorstore.add_documents(documents=documents, embedding=embeddings)

#ベクトルストアが存在した場合、persist()を宣言することで永続化
if vectorstore:
    vectorstore.persist()
    vectorstore = None
else:
    print("Chroma DB has not been initialized.")