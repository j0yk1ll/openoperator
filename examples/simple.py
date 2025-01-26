import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent

load_dotenv()

# Initialize the model
llm = ChatOpenAI(
	model='gpt-4o',
	temperature=0.0,
)

# Create agent with the model
agent = Agent(task='Go to amazon.com and search for "laptop"', llm=llm)


async def main():
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
