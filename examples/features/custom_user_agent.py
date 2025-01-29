import asyncio

from openoperator import LLM, Agent
from openoperator.browser.browser import Browser, BrowserConfig
from openoperator.browser.context import BrowserContext, BrowserContextConfig
from openoperator.controller.service import Controller

# NOTE: This example is to find your current user agent string to use it in the browser_context
task = 'go to https://whatismyuseragent.com and find the current user agent string '

controller = Controller()

browser = Browser(
    config=BrowserConfig(
        # chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    )
)

browser_context = BrowserContext(config=BrowserContextConfig(user_agent='foobarfoo'), browser=browser)

agent = Agent(
    llm=LLM(model='openai/gpt-4o'),
    controller=controller,
    # browser=browser,
    browser_context=browser_context,
    use_vision=True,
    max_actions_per_step=1,
)

agent.add_task(task)


async def main():
    await agent.run(max_steps=25)

    input('Press Enter to close the browser...')
    await browser_context.close()


asyncio.run(main())
