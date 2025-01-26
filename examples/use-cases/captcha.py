import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI

from openoperator import Agent

# NOTE: captchas are hard. For this example it works. But e.g. for iframes it does not.
# for this example it helps to zoom in.
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')
agent = Agent(
    task='go to https://captcha.com/demos/features/captcha-demo.aspx and solve the captcha',
    llm=llm,
)


async def main():
    await agent.run()
    input('Press Enter to exit')


asyncio.run(main())
