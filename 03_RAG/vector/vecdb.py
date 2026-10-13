from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Load PDF
loader = PyPDFLoader("mydocument.pdf")
documents = loader.load()

print("Number of pages:", len(documents))

# Split PDF into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

# Initialize Google Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)
vectorstore = FAISS.from_documents(chunks,embeddings)
print(("vector store created!"))

query = input("\nEnter your question")
results = vectorstore.similarity_search(query,k=3)

# Display embedding information
for i, result in enumerate(results):
    print("\n--------------------")
    print("Chunk:", i + 1)
    print("Text:", chunks[i].page_content[:100])
    print("Dimensions:", len(results))
    print("First 5 values:", results[:5])
