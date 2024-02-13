#vector-db-create.py
# create a vector database from a pdf file

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


loader = PyPDFDirectoryLoader("data-grants/")

docs = loader.load()
#split text to chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
docs = text_splitter.split_documents(docs)
#embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})


embedding_function = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={'device': 'cuda'})

#print(len(docs))

vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db_grants")

print(vectorstore._collection.count())
print("Done")