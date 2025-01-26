import asyncio

from langchain_openai import ChatOpenAI

from openoperator import Agent
from openoperator.browser.browser import Browser, BrowserConfig
from openoperator.browser.context import BrowserContextConfig

llm = ChatOpenAI(model='gpt-4o', temperature=0.0)
task = (
    'go to google.com and search for openai.com and click on the first link then extract content and scroll down - whats there?'
)

allowed_domains = ['google.com']

browser = Browser(
    config=BrowserConfig(
        chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        new_context_config=BrowserContextConfig(
            allowed_domains=allowed_domains,
        ),
    ),
)

agent = Agent(
    task=task,
    llm=llm,
    browser=browser,
)


async def main():
    await agent.run(max_steps=25)

    input('Press Enter to close the browser...')
    await browser.close()


asyncio.run(main())
