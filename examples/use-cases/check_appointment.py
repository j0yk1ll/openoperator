import asyncio

import dotenv
from pydantic import BaseModel

from openoperator.agent.service import LLM, Agent
from openoperator.controller.service import Controller

dotenv.load_dotenv()


controller = Controller()


class WebpageInfo(BaseModel):
    link: str = 'https://appointment.mfa.gr/en/reservations/aero/ireland-grcon-dub/'


@controller.action('Go to the webpage', param_model=WebpageInfo)
def go_to_webpage(webpage_info: WebpageInfo):
    return webpage_info.link


async def main():
    model = LLM(model='openai/gpt-4o')
    agent = Agent(model, controller=controller, use_vision=True)

    agent.add_tasks(
        [
            'Go to the Greece MFA webpage via the link I provided you.',
            'Check the visa appointment dates. If there is no available date in this month, check the next month.',
            'If there is no available date in both months, tell me there is no available date.',
        ]
    )

    await agent.run()


if __name__ == '__main__':
    asyncio.run(main())
