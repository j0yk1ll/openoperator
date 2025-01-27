import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from openoperator import Agent

load_dotenv()

# Initialize the model
llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0.0,
)

agent = Agent(llm=llm)
agent.add_task('Find the founders of browser-use and draft them a short personalized message')


async def main():
    await agent.run()


if __name__ == '__main__':
    asyncio.run(main())
