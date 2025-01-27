import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from openoperator import Agent

# dotenv
load_dotenv()

api_key = os.getenv('DEEPSEEK_API_KEY', '')
if not api_key:
    raise ValueError('DEEPSEEK_API_KEY is not set')


async def run_search():
    agent = Agent(
        llm=ChatOpenAI(
            base_url='https://api.deepseek.com/v1',
            model='deepseek-chat',
            api_key=SecretStr(api_key),
        ),
        use_vision=False,
    )

    agent.add_tasks(
        [
            'Go to https://www.reddit.com/r/LocalLLaMA',
            "Search for 'OpenOperator' in the search bar",
            'Click on first result',
            'Return the first comment',
        ]
    )

    await agent.run()


if __name__ == '__main__':
    asyncio.run(run_search())
