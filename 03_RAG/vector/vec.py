from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

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

# Generate embeddings
vectors = embeddings.embed_documents(
    [chunk.page_content for chunk in chunks]
)

# Display embedding information
for i, vector in enumerate(vectors):
    print("\n--------------------")
    print("Chunk:", i + 1)
    print("Text:", chunks[i].page_content[:100])
    print("Dimensions:", len(vector))
    print("First 5 values:", vector[:5])