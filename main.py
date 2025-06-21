from langchain_core.messages import HumanMessage
from langchain_together import ChatTogether
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()


def main():
    model = ChatTogether(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        together_api_key=os.environ["TOGETHER_API_KEY"]
    )

    tools = []
    agent_executor = create_react_agent(model, tools)

    print("Welcome! Chat and if wanna quit Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break

        print("\nNovus: ", end="")
        try:
            for chunk in agent_executor.stream({"messages": [HumanMessage(content=user_input)]}):
                if "agent" in chunk and "messages" in chunk["agent"]:
                    for message in chunk["agent"]["messages"]:
                        print(message.content, end="")
        except Exception as e:
            print(f"\nError: {e}")
        print()

if __name__ == "__main__":
    main()
