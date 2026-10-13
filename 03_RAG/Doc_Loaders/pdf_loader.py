from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("mydocument.pdf")
documents = loader.load()


print("Number of pages :",len(documents))


for doc in documents:
  print("\n----------------")
  print("Page :",doc.metadata.get("page"))
  print(doc.page_content[:500])