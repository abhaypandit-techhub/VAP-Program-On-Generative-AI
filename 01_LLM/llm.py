from transformers import pipeline

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

prompt = input("Enter your question: ")

response = llm(prompt, max_new_tokens=50)

print("\nLLM Response:")
print(response[0]["generated_text"])