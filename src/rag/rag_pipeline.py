from document_utils import load_test_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")

# 1. Load documents
documents = load_test_documents()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    api_version="2024-10-21"
)

# 4. Create vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 6. Create LLM
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT"),
    api_version="2024-12-01-preview",
    temperature=0
)

# 7. Create QA chain (using LangChain classic)
from langchain_classic.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 8. Query!
result = qa_chain.invoke({
    "query": "¿Qué carrera debo estudiar si me gusta la inteligencia artificial?"
})

print(result["result"])