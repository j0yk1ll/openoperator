import asyncio

from openoperator.agent.service import LLM, Agent
from openoperator.browser.browser import Browser, BrowserConfig, BrowserContextConfig

browser = Browser(
    config=BrowserConfig(
        headless=False,  # This is True in production
        disable_security=True,
        new_context_config=BrowserContextConfig(
            disable_security=True,
            minimum_wait_page_load_time=1,  # 3 on prod
            maximum_wait_page_load_time=10,  # 20 on prod
            # no_viewport=True,
            browser_window_size={
                'width': 1280,
                'height': 1100,
            },
            # trace_path='./tmp/web_voyager_agent',
        ),
    )
)
llm = LLM(model='openai/gpt-4o')


async def main():
    agent = Agent(
        llm=llm,
        browser=browser,
        validate_output=True,
    )
    agent.add_task(
        'Find and book a hotel in Paris with suitable accommodations for a family of four (two adults and two children) offering free cancellation for the dates of February 14-21, 2025. on https://www.booking.com/'
    )
    history = await agent.run(max_steps=50)
    history.save_to_file('./tmp/history.json')


if __name__ == '__main__':
    asyncio.run(main())
