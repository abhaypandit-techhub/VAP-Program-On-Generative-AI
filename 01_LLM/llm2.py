from transformers import pipeline

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-large"
)
print("Simple LLM Chatbot")



print("Type 'exit' to stop")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = llm(
        user_input,
        max_new_tokens=200,
        #         num_beams=4,
        # repetition_penalty=2.0,
        # no_repeat_ngram_size=2
    )
    print("Bot:", response[0]["generated_text"])