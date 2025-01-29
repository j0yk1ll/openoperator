import asyncio

from openoperator import LLM, Agent

llm = LLM(model='openai/gpt-4o')


async def main():
    agent = Agent(
        llm=llm,
    )

    agent.add_task('open 3 tabs with elon musk, trump, and steve jobs, then go back to the first and stop')

    await agent.run()


asyncio.run(main())
