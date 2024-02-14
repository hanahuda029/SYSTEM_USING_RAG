#APIキーを参照するためのモジュール
import openai
from dotenv import load_dotenv
import os

#sqlite3のバージョン補完(現状のローカル開発環境では修正不可)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#.envからAPIキーを参照
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#LangChainの各種モジュール
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

#データベースの設定
CHUNK_SIZE = 900
CHUNK_OVERLAP = 0
PRESIST_PATH = "./vectorstore"
MODEL_NAME = "text-embedding-3-large"
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

#データベース構築
db = Chroma.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(),
    persist_directory='./vectorstore'

)
#データベースが存在した場合、persist()を宣言することで永続化
if db:
    db.persist()
    db = None
else:
    print("Chroma DB has not been initialized.")