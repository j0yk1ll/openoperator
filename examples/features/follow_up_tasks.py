import asyncio

from dotenv import load_dotenv

from openoperator import LLM, Agent
from openoperator.controller.service import Controller

load_dotenv()

controller = Controller()


async def main():
    agent = Agent(llm=LLM(model='openai/gpt-4o'), controller=controller)

    agent.add_task('Find the names of the browser-use founders')

    await agent.run()

    agent.add_task('Find the emails for each founder')
    agent.add_tasks(
        [
            'Draft a short thank you message for each',
            'Send each founder a thank you email',
        ]
    )

    await agent.run()


if __name__ == '__main__':
    asyncio.run(main())
