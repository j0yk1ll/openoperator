import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from openoperator import Agent
from openoperator.controller.service import Controller

load_dotenv()

# Initialize the model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
)
controller = Controller()


task = "Find the names of the browser-use founders"

agent = Agent(llm=llm, controller=controller)


async def main():
    await agent.run()

    agent.add_task("Find the emails for each founder")
    agent.add_tasks(
        [
            "Draft a short thank you message for each",
            "Send each founder a thank you email",
        ]
    )

    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
