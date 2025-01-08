from typing import Literal, Union
from pydantic import BaseModel, Field
from rich.prompt import Prompt

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.result import Usage, UsageLimits

class FlightDetails(BaseModel):
    flight_number: str
    
class Failed(BaseModel):
    """Unable to find a satisfactory choice."""
    
flight_search_agent = Agent[None, Union[FlightDetails, Failed]](
    'openai:gpt-4o mini',
    result_type=Union[FlightDetails, Failed], # type: ignore
    system_prompt=(
        'Use the "flight_search" tool to find a flight '
        'from the given origin to the given destination.'
    ),
)

@flight_search_agent.tool
async def flight_search(
    ctx: RunContext[None], origin: str, destination: str
) -> Union[FlightDetails, None]:
    # in reality, this would call a flight search API or
    # use a browser to scrape a flight search website
    return FlightDetails(flight_number='123')

usage_limits = UsageLimits(request_limits=15)

async def find_flight(usage: Usage) -> Union[FlightDetails, None]:
    message_history: Union[list[ModelMessage], None] = None
    for _ in range(3):
        prompt = Prompt.ask(
            'Where would you like to fly from and to?',
        )
        result = await flight_search_agent.run(
            prompt,
            message_history=message_history,
            usage=usage,
            usage_limits=usage_limits,
        )
        if isinstance(result.data, FlightDetails):
            return result.data
        else:
            message_history = result.all_messages(
                result_tool_return_content='Please try again.'
            )

class SeatPreference(BaseModel):
    row: int = Field(ge=1, le=30)
    seat: Literal['A', 'B', 'C', 'D', 'E', 'F']
    
# This agent is responsible for extracting the user's seat selection
seat_preference_agent = Agent[None, Union[SeatPreference, Failed]](
    'openai:gpt-4o-mini',
    result_type=Union[SeatPreference, Failed], # type: ignore
    system_prompt=(
        "Extract the users a seat preference. "
        'Seats A and F are window seats.'
        'Row 1 is the front row and has extra leg room. '
        'Rows 14, and 20 also have extra leg room.'
        ),
)
async def find_seat(usage: Usage) -> SeatPreference:
    message_history: Union[list[ModelMessage], None] = None
    while True:
        answer = Prompt.ask('What seat would you like?')
        
        result = await seat_preference_agent.run(
            answer,
            message_history=message_history,
            usage=usage,
            usage_limits=usage_limits,
        )
        if isinstance(result.data, SeatPreference):
            return result.data
        else:
            print('Could not understand seat preference. Please try again. ')
            message_history = result.all_messages()
            
async def main():
    usage: Usage = Usage()
    
    opt_flight_details = await find_flight(usage)
    if opt_flight_details is not None:
        print(f'Found flight: {opt_flight_details}')
        #> Flight found: AK456
        seat_preference = await find_seat(usage)
        print(f'Seat preference: {seat_preference}')
        #> Seat preference: row=1 seat='A'

