import asyncio
import os

from langchain_google_genai import ChatGoogleGenerativeAI

from openoperator import Agent
from openoperator.browser.browser import Browser, BrowserConfig
from openoperator.browser.context import BrowserContext, BrowserContextConfig

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')


browser = Browser(
    config=BrowserConfig(
        # chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    )
)
file_path = os.path.join(os.path.dirname(__file__), 'twitter_cookies.txt')
context = BrowserContext(browser=browser, config=BrowserContextConfig(cookies_file=file_path))


async def run_search():
    agent = Agent(
        browser_context=context,
        task=('go to https://x.com. write a new post with the text "openoperator ftw", and submit it'),
        llm=llm,
        max_actions_per_step=4,
    )
    await agent.run(max_steps=25)
    input('Press Enter to close the browser...')


if __name__ == '__main__':
    asyncio.run(run_search())
