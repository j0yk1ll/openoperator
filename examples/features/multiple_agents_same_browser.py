import asyncio

from openoperator import LLM, Agent, Browser


async def main():
    # Persist the browser state across agents

    browser = Browser()
    context = await browser.new_context()
    async with context as context:
        model = LLM(model='openai/gpt-4o')
        current_agent = None

        async def get_input():
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: input('Enter task (p: pause current agent, r: resume, b: break): '),
            )

        while True:
            task = await get_input()

            if task.lower() == 'p':
                # Pause the current agent if one exists
                if current_agent:
                    current_agent.pause()
                continue
            elif task.lower() == 'r':
                # Resume the current agent if one exists
                if current_agent:
                    current_agent.resume()
                continue
            elif task.lower() == 'b':
                # Break the current agent's execution if one exists
                if current_agent:
                    current_agent.stop()
                    current_agent = None
                continue

            # If there's a current agent running, pause it before starting new one
            if current_agent:
                current_agent.pause()

            # Create and run new agent with the task
            current_agent = Agent(
                llm=model,
                browser_context=context,
            )

            current_agent.add_task(task)

            # Run the agent asynchronously without blocking
            asyncio.create_task(current_agent.run())


asyncio.run(main())
