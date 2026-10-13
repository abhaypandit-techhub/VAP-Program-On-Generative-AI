from langchain_community.document_loaders import TextLoader

loader = TextLoader("sample.txt")
documents = loader.load()
print(documents)