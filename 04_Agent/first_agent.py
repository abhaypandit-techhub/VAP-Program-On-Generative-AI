from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid expression."

@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city."""
    weather_data = {"hyderabad": "32°C, Sunny",
        "delhi": "30°C, Cloudy",
        "mumbai": "28°C, Rainy",
        "bangalore": "25°C, Pleasant"}

    return weather_data.get(city.lower(), "Weather information not available.")

parser = StrOutputParser()

agent = create_agent(model=model, tools=[calculator, get_weather])

print("===================================")
print("         AI Agent")
print("===================================")
print("Type 'exit' to stop.\n")

while True:

    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    final_answer = parser.invoke(result["messages"][-1])

    print("AI:", final_answer)