from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from openoperator import Agent

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')

initial_actions = [
    {'open_tab': {'url': 'https://www.google.com'}},
    {'open_tab': {'url': 'https://en.wikipedia.org/wiki/Randomness'}},
    {'scroll_down': {'amount': 1000}},
    {'extract_content': {'include_links': False}},
]


async def main():
    agent = Agent(
        initial_actions=initial_actions,
        llm=llm,
    )

    agent.add_task('What theories are displayed on the page?')

    await agent.run(max_steps=10)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
