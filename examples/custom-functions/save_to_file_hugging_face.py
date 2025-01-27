import asyncio
from typing import List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from openoperator.agent.service import Agent
from openoperator.controller.service import Controller

# Initialize controller first
controller = Controller()


class Model(BaseModel):
    title: str
    url: str
    likes: int
    license: str


class Models(BaseModel):
    models: List[Model]


@controller.action('Save models', param_model=Models)
def save_models(params: Models):
    with open('models.txt', 'a') as f:
        for model in params.models:
            f.write(f'{model.title} ({model.url}): {model.likes} likes, {model.license}\n')


# video: https://preview.screen.studio/share/EtOhIk0P
async def main():
    model = ChatOpenAI(model='gpt-4o')
    agent = Agent(llm=model, controller=controller)

    agent.add_task('Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.')

    await agent.run()


if __name__ == '__main__':
    asyncio.run(main())
