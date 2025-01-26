import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI

from openoperator.agent.service import Agent
from openoperator.browser.browser import Browser, BrowserConfig
from openoperator.browser.context import BrowserContextConfig

browser = Browser(
    config=BrowserConfig(
        disable_security=True,
        headless=False,
        new_context_config=BrowserContextConfig(save_recording_path='./tmp/recordings'),
    )
)
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')


async def main():
    agents = [
        Agent(task=task, llm=llm, browser=browser)
        for task in [
            'Search Google for weather in Tokyo',
            'Check Reddit front page title',
            'Look up Bitcoin price on Coinbase',
            'Find NASA image of the day',
            # 'Check top story on CNN',
            # 'Search latest SpaceX launch date',
            # 'Look up population of Paris',
            # 'Find current time in Sydney',
            # 'Check who won last Super Bowl',
            # 'Search trending topics on Twitter',
        ]
    ]

    await asyncio.gather(*[agent.run() for agent in agents])

    # async with await browser.new_context() as context:
    agentX = Agent(
        task='Go to apple.com and return the title of the page',
        llm=llm,
        browser=browser,
        # browser_context=context,
    )
    await agentX.run()

    await browser.close()


if __name__ == '__main__':
    asyncio.run(main())
