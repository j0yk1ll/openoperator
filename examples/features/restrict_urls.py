import asyncio

from openoperator import LLM, Agent
from openoperator.browser.browser import Browser, BrowserConfig
from openoperator.browser.context import BrowserContextConfig

llm = LLM(model='openai/gpt-4o')

allowed_domains = ['google.com']

browser = Browser(
    config=BrowserConfig(
        chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        new_context_config=BrowserContextConfig(
            allowed_domains=allowed_domains,
        ),
    ),
)


async def main():
    agent = Agent(
        llm=llm,
        browser=browser,
    )

    agent.add_task(
        'go to google.com and search for openai.com and click on the first link then extract content and scroll down - whats there?'
    )

    await agent.run(max_steps=25)

    input('Press Enter to close the browser...')
    await browser.close()


asyncio.run(main())
