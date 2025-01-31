import asyncio
import os

from dotenv import load_dotenv

from openoperator import LLM, Agent

# dotenv
load_dotenv()

api_key = os.getenv('DEEPSEEK_API_KEY', '')
if not api_key:
    raise ValueError('DEEPSEEK_API_KEY is not set')


async def run_search():
    llm = LLM(model='deepseek/deepseek-chat')
    agent = Agent(
        llm=llm,
        use_vision=False,
    )

    agent.add_task('Go to https://www.reddit.com/r/LocalLLaMA')
    agent.add_task("Search for 'OpenOperator' in the search bar")
    agent.add_task('Click on first result')
    agent.add_task('Return the first comment')

    await agent.run()


if __name__ == '__main__':
    asyncio.run(run_search())
