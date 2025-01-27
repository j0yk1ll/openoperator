import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI

from openoperator import Agent
from openoperator.browser.browser import Browser, BrowserConfig

"""
Example: Using the 'Scroll down' action.

This script demonstrates how the agent can navigate to a webpage and scroll down the content.
If no amount is specified, the agent will scroll down by one page height.
"""

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')

agent = Agent(
    llm=llm,
    browser=Browser(config=BrowserConfig(headless=False)),
)

agent.add_task("Navigate to 'https://en.wikipedia.org/wiki/Internet' and scroll to the string 'The vast majority of computer'")


async def main():
    await agent.run()


if __name__ == '__main__':
    asyncio.run(main())
