import pickle
import re
from datetime import datetime

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS

def add_description(doc):
    source = doc.metadata['source']
    dt_str = re.search(r'(\d{4})-(\d{2})-(\d{2}) (\d{2})-(\d{2})-(\d{2})', source).group(0)
    dt = datetime.strptime(dt_str, '%Y-%m-%d %H-%M-%S')
    doc.page_content = 'Transcript of telephone conversation on ' + str(dt) + '\n' + doc.page_content
    return doc

def ingest_transcripts():
    loader = DirectoryLoader("whiskey-jack", loader_cls=TextLoader)
    docs = loader.load()
    docs = [add_description(doc) for doc in docs]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

if __name__ == "__main__":
    ingest_transcripts()    
