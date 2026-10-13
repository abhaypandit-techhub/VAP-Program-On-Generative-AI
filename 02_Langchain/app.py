from langchain_huggingface import HuggingFacePipeline # hugging face API
from transformers import pipeline
from langchain_core.prompts import PromptTemplate

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=200
)

llm = HuggingFacePipeline(
    pipeline=generator
)

prompt = PromptTemplate.from_template("""
You are a teaching assistant.

Explain the following topic.

Topic: {topic}

Give the answer in this format:

Definition:
Explanation:
Example:
""")

chain = prompt | llm

topic = input("Enter a topic: ")

response = chain.invoke({
    "topic": topic
})

print("\n========== Answer ==========\n")
print(response)