from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss



# List of PDF URLs to load
pdfs = [
    "background-note_carbon-tax.pdf"
]

docs = []
for url in pdfs:
    loader = PyPDFLoader(url)
    docs.extend(loader.load())


text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)

all_splits = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="phi3:latest")

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(embedding_function=embeddings,index = index, docstore=InMemoryDocstore(),index_to_docstore_id={})

vs = vector_store.add_documents(documents=all_splits)

model = ChatOllama(model="phi3:latest")

prompt = ChatPromptTemplate.from_template(
    """
        You are a knowledgeable assistant specializing in question-answering. Please utilize the provided context to formulate your response. If the answer is not available, simply state that you do not know. Limit your response to a maximum of three concise sentences.

        Question: {question}

        Context: {context}

        Answer:
    """
)

question = input("enter your query: ")

ret_doc = vector_store.similarity_search(question)

for doc in ret_doc:
    print("Content: ",doc.page_content)
    print("Metadata" ,doc.metadata)
    
    
