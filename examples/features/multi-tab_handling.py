import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI

from openoperator import Agent

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')

agent = Agent(
    task='open 3 tabs with elon musk, trump, and steve jobs, then go back to the first and stop',
    llm=llm,
)


async def main():
    await agent.run()


asyncio.run(main())
