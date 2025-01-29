"""
Automated news analysis and sentiment scoring using Bedrock.

@dev Ensure AWS environment variables are set correctly for Bedrock access.
"""

import argparse
import asyncio

from openoperator import LLM, Agent
from openoperator.browser.browser import Browser, BrowserConfig
from openoperator.controller.service import Controller

# Define the task for the agent
task = (
    "Visit cnn.com, navigate to the 'World News' section, and identify the latest headline. "
    'Open the first article and summarize its content in 3-4 sentences. '
    'Additionally, analyze the sentiment of the article (positive, neutral, or negative) '
    'and provide a confidence score for the sentiment. Present the result in a tabular format.'
)

parser = argparse.ArgumentParser()
parser.add_argument('--query', type=str, help='The query for the agent to execute', default=task)
args = parser.parse_args()

browser = Browser(
    config=BrowserConfig(
        # chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    )
)


async def main():
    llm = LLM(model='bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0')
    agent = Agent(
        llm=llm,
        controller=Controller(),
        browser=browser,
        validate_output=True,
    )
    agent.add_task(args.query)
    await agent.run(max_steps=30)
    await browser.close()


asyncio.run(main())
