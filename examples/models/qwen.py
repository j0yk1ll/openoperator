import asyncio

from langchain_ollama import ChatOllama

from openoperator import Agent


async def run_search():
    agent = Agent(
        llm=ChatOllama(
            # model='qwen2.5:32b-instruct-q4_K_M',
            # model='qwen2.5:14b',
            model='qwen2.5:latest',
            num_ctx=128000,
        ),
        max_actions_per_step=1,
    )

    agent.add_tasks(
        [
            'Go to https://www.reddit.com/r/LocalLLaMA',
            "Search for 'OpenOperator' in the search bar",
            'Click search',
            'Call done',
        ]
    )

    await agent.run()


if __name__ == '__main__':
    asyncio.run(run_search())
