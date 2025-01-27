import asyncio

from langchain_openai import ChatOpenAI

from openoperator.agent.service import Agent
from openoperator.browser.browser import Browser
from openoperator.browser.context import BrowserContextConfig

llm = ChatOpenAI(model='gpt-4o', temperature=0.0)


async def main():
    browser = Browser()

    async with await browser.new_context(config=BrowserContextConfig(trace_path='./tmp/traces/')) as context:
        agent = Agent(
            llm=llm,
            browser_context=context,
        )
        agent.add_task('Go to hackernews, then go to apple.com and return all titles of open tabs')
        await agent.run()

    await browser.close()


asyncio.run(main())
