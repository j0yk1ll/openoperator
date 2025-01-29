import asyncio

from openoperator import LLM, Agent

# NOTE: captchas are hard. For this example it works. But e.g. for iframes it does not.
# for this example it helps to zoom in.
llm = LLM(model='openai/gpt-4o')
agent = Agent(
    llm=llm,
)

agent.add_task('go to https://captcha.com/demos/features/captcha-demo.aspx and solve the captcha')


async def main():
    await agent.run()
    input('Press Enter to exit')


asyncio.run(main())
