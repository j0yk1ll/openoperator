import asyncio
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel

from openoperator import LLM, Agent, Controller

load_dotenv()


class Post(BaseModel):
    post_title: str
    post_url: str
    num_comments: int
    hours_since_post: int


class Posts(BaseModel):
    posts: List[Post]


controller = Controller(output_model=Posts)


async def main():
    model = LLM(model='openai/gpt-4o')
    agent = Agent(llm=model, controller=controller)

    agent.add_task('Go to hackernews show hn and give me the first  5 posts')

    history = await agent.run()

    result = history.final_result()
    if result:
        parsed: Posts = Posts.model_validate_json(result)

        for post in parsed.posts:
            print('\n--------------------------------')
            print(f'Title:            {post.post_title}')
            print(f'URL:              {post.post_url}')
            print(f'Comments:         {post.num_comments}')
            print(f'Hours since post: {post.hours_since_post}')
    else:
        print('No result')


if __name__ == '__main__':
    asyncio.run(main())
