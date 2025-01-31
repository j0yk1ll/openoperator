import asyncio

from openoperator import LLM, Agent

llm = LLM(model='gemini/gemini-2.0-flash-exp')


async def run_search():
    agent = Agent(
        llm=llm,
        max_actions_per_step=4,
    )

    agent.add_task('Go to url r/LocalLLaMA subreddit and search for "OpenOperator" in the search bar')
    agent.add_task('Click on the first post and find the funniest comment')

    await agent.run(max_steps=25)


if __name__ == '__main__':
    asyncio.run(run_search())
