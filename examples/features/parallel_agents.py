import asyncio

from openoperator.agent.service import LLM, Agent
from openoperator.browser.browser import Browser, BrowserConfig
from openoperator.browser.context import BrowserContextConfig

browser = Browser(
    config=BrowserConfig(
        disable_security=True,
        headless=False,
        new_context_config=BrowserContextConfig(save_recording_path='./tmp/recordings'),
    )
)
llm = LLM(model='openai/gpt-4o')


async def main():
    agents = []
    for task in [
        'Search Google for weather in Tokyo',
        'Check Reddit front page title',
        'Look up Bitcoin price on Coinbase',
        'Find NASA image of the day',
    ]:
        agent = Agent(llm=llm, browser=browser)
        agent.add_task(task)
        agents.append(agent)

    await asyncio.gather(*[agent.run() for agent in agents])

    agentX = Agent(
        llm=llm,
        browser=browser,
    )

    agentX.add_task(
        'Go to apple.com and return the title of the page',
    )

    await agentX.run()

    await browser.close()


if __name__ == '__main__':
    asyncio.run(main())
