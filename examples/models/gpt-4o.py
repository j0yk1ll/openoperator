import asyncio

from langchain_openai import ChatOpenAI

from openoperator import Agent

llm = ChatOpenAI(model='gpt-4o')
agent = Agent(
    task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
    llm=llm,
)


async def main():
    await agent.run(max_steps=10)
    input('Press Enter to continue...')


asyncio.run(main())
