import asyncio

from dotenv import load_dotenv

from openoperator import Agent
from openoperator.llm import LLM

load_dotenv()

# Initialize the model
llm = LLM('ollama/llama3.2')

agent = Agent(llm=llm)
agent.add_task('Find the founders of browser-use and draft them a short personalized message')


async def main():
    await agent.run()


if __name__ == '__main__':
    asyncio.run(main())
