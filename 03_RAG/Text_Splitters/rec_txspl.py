
# Recursive textSplitters

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("mydocuments.pdf")
documents = loader.load()

print("Number of pages:", len(documents))

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split documents into chunks
chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

# Display chunks 
for i, chunk in enumerate(chunks):
    print("\n====================")
    print("Chunk:", i + 1)
    print("====================")
    print(chunk.page_content)