from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

loader = PyPDFLoader("mydocument.pdf")
documents = loader.load()

print("Pages:", len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)
print("Chunks:", len(chunks))

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

print("Vector store created!")

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

question = input("\nAsk a question: ")
docs = retriever.invoke(question)

print("\n* RETRIEVED DOCUMENTS *")

for i, doc in enumerate(docs):
    print(f"\nResult {i + 1}")
    print("Page:", doc.metadata.get("page"))
    print(doc.page_content)