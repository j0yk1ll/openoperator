import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel

from openoperator import LLM, ActionResult, Agent, Controller

load_dotenv()

controller = Controller()


class DoneResult(BaseModel):
    title: str
    comments: str
    hours_since_start: int


# we overwrite done() in this example to demonstrate the validator
@controller.registry.action('Done with task', param_model=DoneResult)
async def done(params: DoneResult):
    result = ActionResult(is_done=True, extracted_content=params.model_dump_json())
    print(result)
    # NOTE: this is clearly wrong - to demonstrate the validator
    return 'blablabla'


async def main():
    model = LLM(model='openai/gpt-4o')
    agent = Agent(llm=model, controller=controller, validate_output=True)
    agent.add_task('Go to hackernews hn and give me the top 1 post')
    # NOTE: this should fail to demonstrate the validator
    await agent.run(max_steps=5)


if __name__ == '__main__':
    asyncio.run(main())
