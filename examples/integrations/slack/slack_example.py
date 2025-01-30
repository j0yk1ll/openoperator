import os

from dotenv import load_dotenv

from examples.integrations.slack.slack_api import SlackBot, app
from openoperator import LLM, BrowserConfig

load_dotenv()

# load credentials from environment variables
bot_token = os.getenv('SLACK_BOT_TOKEN')
if not bot_token:
    raise ValueError('Slack bot token not found in .env file.')

signing_secret = os.getenv('SLACK_SIGNING_SECRET')
if not signing_secret:
    raise ValueError('Slack signing secret not found in .env file.')

llm = LLM(model='openai/gpt-4o')

slack_bot = SlackBot(
    llm=llm,  # required; instance of LLM
    bot_token=bot_token,  # required; Slack bot token
    signing_secret=signing_secret,  # required; Slack signing secret
    ack=True,  # optional; whether to acknowledge task receipt with a message, defaults to False
    browser_config=BrowserConfig(
        headless=True
    ),  # optional; useful for changing headless mode or other browser configs, defaults to headless mode
)

app.dependency_overrides[SlackBot] = lambda: slack_bot

if __name__ == '__main__':
    import uvicorn

    uvicorn.run('integrations.slack.slack_api:app', host='0.0.0.0', port=3000)
