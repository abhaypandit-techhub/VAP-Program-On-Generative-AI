# from langchain_core.prompts import PromptTemplate

# prompt = PromptTemplate.from_template(
#     "Explain {topic} in simple words."
# )

# formatted_prompt = prompt.invoke({"topic": "ML"})

# print(formatted_prompt)


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI teacher."),
    ("human", "Explain {topic} simply.")
])

prompt_value = prompt.invoke({
    "topic": "Attention mechanism"
})

print(prompt_value)