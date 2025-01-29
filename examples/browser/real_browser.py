import asyncio

from openoperator import LLM, Agent
from openoperator.browser.browser import Browser, BrowserConfig

browser = Browser(
    config=BrowserConfig(
        # NOTE: you need to close your chrome browser - so that this can open your browser in debug mode
        chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    )
)


async def main():
    agent = Agent(
        llm=LLM(model='openai/gpt-4o'),
        browser=browser,
    )

    agent.add_task('In docs.google.com write my Papa a quick letter')

    await agent.run()
    await browser.close()

    input('Press Enter to close...')


if __name__ == '__main__':
    asyncio.run(main())
