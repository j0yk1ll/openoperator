import asyncio
import os

from dotenv import load_dotenv

from openoperator import Agent
from openoperator.llm import LLM

load_dotenv()

# Initialize the model
llm = LLM('openai/gpt-4o')

# Get credentials from environment
X_USERNAME = os.environ.get("X_USERNAME")

if not X_USERNAME:
    raise Exception("Must set X_USERNAME for this example to work.")

X_PASSWORD = os.environ.get("X_PASSWORD")

if not X_PASSWORD:
    raise Exception("Must set X_PASSWORD for this example to work.")


async def main():
    agent = Agent(llm=llm)
    agent.add_task(
        'Sign in to x.com',
        additional_information={"username": X_USERNAME, "password": X_PASSWORD},  # type: ignore
    )
    await agent.run()


if __name__ == '__main__':
    asyncio.run(main())
