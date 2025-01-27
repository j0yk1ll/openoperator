import asyncio
from pprint import pprint

from langchain_google_genai import ChatGoogleGenerativeAI

from openoperator import Agent
from openoperator.agent.views import AgentHistoryList
from openoperator.browser.browser import Browser, BrowserConfig
from openoperator.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')
browser = Browser(
    config=BrowserConfig(
        headless=False,
        disable_security=True,
        extra_chromium_args=['--window-size=2000,2000'],
    )
)


async def main():
    async with await browser.new_context(
        config=BrowserContextConfig(
            trace_path='./tmp/result_processing',
            no_viewport=False,
            browser_window_size=BrowserContextWindowSize(width=1280, height=1000),
        )
    ) as browser_context:
        agent = Agent(
            llm=llm,
            browser_context=browser_context,
        )
        agent.add_task(
            "go to google.com and type 'OpenAI' click search and give me the first url",
        )
        history: AgentHistoryList = await agent.run(max_steps=3)

        print('Final Result:')
        pprint(history.final_result(), indent=4)

        print('\nErrors:')
        pprint(history.errors(), indent=4)

        # e.g. xPaths the model clicked on
        print('\nModel Outputs:')
        pprint(history.model_actions(), indent=4)

        print('\nThoughts:')
        pprint(history.model_thoughts(), indent=4)
    # close browser
    await browser.close()


if __name__ == '__main__':
    asyncio.run(main())
