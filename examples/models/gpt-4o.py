import asyncio

from openoperator import LLM, Agent

llm = LLM(model='openai/gpt-4o')
agent = Agent(
    llm=llm,
)

agent.add_task('Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result')


async def main():
    await agent.run(max_steps=10)
    input('Press Enter to continue...')


asyncio.run(main())
