import asyncio

from langchain_ollama import ChatOllama

from openoperator import Agent
from openoperator.agent.views import AgentHistoryList


async def run_search() -> AgentHistoryList:
    agent = Agent(
        llm=ChatOllama(
            model='qwen2.5:32b-instruct-q4_K_M',
            num_ctx=32000,
        ),
    )

    agent.add_task("Search for a 'OpenOperator' post on the r/LocalLLaMA subreddit and open it.")

    result = await agent.run()
    return result


async def main():
    result = await run_search()
    print('\n\n', result)


if __name__ == '__main__':
    asyncio.run(main())
