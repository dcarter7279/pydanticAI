# Simple example of using PydanticAI to construct a Pydantic model from a text input.
# Demonstrates: structured result_type
# Running the Example
# With dependencies installed and environment variables set, run:

import os
from typing import cast

import logfire
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName

# 'if-token-present' means nothing will be sent (and the example will work) if you dont have logfire configured
logfire.configure(send_to_logfire="if-token-present")

class MyModel(BaseModel):
    city: str
    coununtry: str
    
model = cast(KnownModelName, os.getenv("MODEL_NAME", "openai:gpt-4o"))
print(f'Using model: {model}')
agent = Agent(model, result_type=MyModel)

if __name__ == "__main__":
    result = agent.run('The windy city in the US of A.')
    print(result.data)
    print(result.usage())