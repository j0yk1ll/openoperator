from dotenv import load_dotenv

from openoperator import LLM, Agent

load_dotenv()

llm = LLM(model='openai/gpt-4o')

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
