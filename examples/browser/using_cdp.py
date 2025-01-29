"""
Simple demonstration of the CDP feature.

To test this locally, follow these steps:
1. Create a shortcut for the executable Chrome file.
2. Add the following argument to the shortcut:
   - On Windows: `--remote-debugging-port=9222`
3. Open a web browser and navigate to `http://localhost:9222/json/version` to verify that the Remote Debugging Protocol (CDP) is running.
4. Launch this example.

@dev You need to set the `GEMINI_API_KEY` environment variable before proceeding.
"""

import asyncio

from openoperator import LLM, Agent, Controller
from openoperator.browser.browser import Browser, BrowserConfig

browser = Browser(
    config=BrowserConfig(
        headless=False,
        cdp_url='http://localhost:9222',
    )
)
controller = Controller()


async def main():
    agent = Agent(
        llm=LLM(model='openai/gpt-4o'),
        controller=controller,
        browser=browser,
    )

    agent.add_task(
        'In docs.google.com write my Papa a quick thank you for everything letter \n - Magnus and save the document as pdf'
    )

    await agent.run()
    await browser.close()

    input('Press Enter to close...')


if __name__ == '__main__':
    asyncio.run(main())
