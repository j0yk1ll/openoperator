import asyncio
import os

from openoperator.agent.service import Agent
from openoperator.llm import LLM


async def main():
    llm = LLM('gemini/gemini-2.0-flash-exp')

    agent = Agent(llm=llm)

    agent.add_task(
        'go to https://kzmpmkh2zfk1ojnpxfn1.lite.vusercontent.net/ and upload both files',
        additional_information={
            "file1": os.path.join(os.path.dirname(__file__), 'nrysj1hke.txt'),
            "file2": os.path.join(os.path.dirname(__file__), 'vMJNZmV3e.txt'),
        },
    )

    await agent.run()


if __name__ == '__main__':
    asyncio.run(main())
