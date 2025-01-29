import asyncio

from openoperator import LLM, Agent, Browser


async def main():
    browser = Browser()
    async with await browser.new_context() as context:
        model = LLM(model='openai/gpt-4o')

        # initialize agents
        agent = Agent(
            llm=model,
            browser_context=context,
        )
        executor = Agent(
            llm=model,
            browser_context=context,
        )

        coder = Agent(
            llm=model,
            browser_context=context,
        )

        # assign tasks
        agent.add_task('Open an online code editor programiz.')
        executor.add_task('Executor. Execute the code written by the coder and suggest some updates if there are errors.')
        coder.add_task(
            'Coder. Your job is to write and complete code. You are an expert coder. Code a simple calculator. Write the code on the coding interface after agent1 has opened the link.'
        )

        # run agents
        await agent.run()
        await executor.run()
        await coder.run()


asyncio.run(main())
