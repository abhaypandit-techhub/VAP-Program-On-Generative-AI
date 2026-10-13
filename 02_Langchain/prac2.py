from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=100
)

llm = HuggingFacePipeline(
    pipeline=generator
)

user_input = input("Enter your question: ")

response = llm.invoke(user_input)

print("Model Answer:")
print(response)