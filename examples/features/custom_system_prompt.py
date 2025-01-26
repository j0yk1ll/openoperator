import asyncio
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from openoperator import Agent, SystemPrompt

load_dotenv()


class MySystemPrompt(SystemPrompt):
    def important_rules(self) -> str:
        existing_rules = super().important_rules()
        new_rules = 'REMEMBER the most important RULE: ALWAYS open first a new tab and go first to url wikipedia.com no matter the task!!!'
        return f'{existing_rules}\n{new_rules}'

        # other methods can be overridden as well (not recommended)


async def main():
    task = "do google search to find images of Elon Musk's wife"
    model = ChatOpenAI(model='gpt-4o')
    agent = Agent(task=task, llm=model, system_prompt_class=MySystemPrompt)

    print(
        json.dumps(
            agent.message_manager.system_prompt.model_dump(exclude_unset=True),
            indent=4,
        )
    )

    await agent.run()


if __name__ == '__main__':
    asyncio.run(main())
