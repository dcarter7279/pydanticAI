import sys
import datetime
from dataclasses import dataclass
from typing import Literal

import logfire
from pydantic import BaseModel, Field, HttpUrl
from rich.prompt import Prompt

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage  import Usage, UsageLimits

class FlightDetails(BaseModel):
    """Details of the most suitable flight."""
    flight_number: str
    price: int
    origin: str = Field(description='Three-letter airport code')
    destination: str = Field(description='Three-letter airportcode')
    date: datetime.date
    
class NoFlightFound(BaseModel):
    """When no valid flight is found."""
    
@dataclass
class Deps:
    web_page_text: str # This will be replaced with scraped data later
    req_origin: str
    req_destination: str
    req_date: datetime.date
    
# This agent is responsible for controlling the flow of the conversation.
search_agent = Agent[Deps, FlightDetails | NoFlightFound](
    'openai:gpt-4o-mini',
    result_type = FlightDetails | NoFlightFound, # type: ignore
    retries=4,
    system_prompt=(
        'Your job is to find the cheapest flight for the user on the given date.'
    ),
)

# This agent is responsible for extracting flight details from web page text.
extraction_agent = Agent(
    'openai:gpt-4o-mini',
    result_type=[FlightDetails],
    system_prompt='Extract all the flight details from the given text.',
)

class WebScrapeInput(BaseModel):
    url: str
        
class WebScrapeOutput(BaseModel):
    data: str | None # Scraped data
    
class WebScraperAgent(Agent):
    def run(self, input: WebScrapeInput) -> WebScrapeOutput:
        try:
            from bs4 import BeautifulSoup
            import requests
            
            response = requests.get(str(input.url)) # Convert HttpUrl to str
            response.raise_for_status() # Raise an exception for bad status codes
            soup = BeautifulSoup(response.content, 'lxml') # Use the lxml parser
            text = soup.get_text(strip=True)
            return WebScrapeOutput(data=text)
        except requests.exceptions.RequestException as e:
            print(f"HTTP Request Error: {e}")
            return WebScrapeOutput(data=None)
        except Exception as e:
            print(f"Scraping Error: {e}")
            return WebScrapeOutput(data=None)




@search_agent.tool
async def extract_flights(ctx: RunContext[Deps]) -> list[FlightDetails]:
    """Get detailss of all flights."""
    # we pass the usage to the search agent so request within this agent are counted
    result = await extraction_agent.run(ctx.deps.web_page_text, usage=ctx.usage)
    logfire.info('found {flight_count} flights', flight_count=len(result.data))
    return result.data

@search_agent.result_validator
async def validate_result(
    ctx: RunContext[Deps], result: FlightDetails | NoFlightFound
) -> FlightDetails | NoFlightFound:
    """Procedural validation that the flight meets the constraints."""
    if isinstance(result, NoFlightFound):
        return result
    
    errors: list[str] = []
    if result.origin != ctx.deps.req_origin:
        errors.append(
            f'Flight should have origin {ctx.deps.req_origin}, not {result.origin}'
        )
    if result.destination != ctx.deps.req_destination:
        errors.append(
            f'Flight should have destination {ctx.deps.req_destination}, not {result.destination}'
        )
    if result.date != ctx.deps.req_date:
        errors.append(f'Flight should have date {ctx.deps.req_date}, not {result.date}')
    
    if errors:
        raise ModelRetry('\n'.join(errors))
    else:
        return result
    
class SeatPreference(BaseModel):
    row: int = Field(ge=1, le=30)
    seat: Literal['A', 'B', 'C', 'D', 'E', 'F']
    
class Failed(BaseModel):
    """Unable to extract a seat selection."""

# This agent is responsible for extracting the user's seat selection 
seat_preference_agent = Agent[
    None, SeatPreference | Failed
](
    'openai:gpt-4o-mini',
    result_type=SeatPreference | Failed, # type : ignore
    system_prompt=(
        'Extract the users seat preference. '
        'Seats A and F are window seats.'
        'Row 1 is the front row and has extra leg room. '
        'Rows 14, and 20 also have extra leg room.'
    ),
)



# in reality this would be downloaded from a booking site,
# potentially using another agent to navigate the site
flights_web_page = """
1. Flight SFO-AK123
- Price: $350
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

2. Flight SFO-AK456
- Price: $370
- Origin: San Francisco International Airport (SFO)
- Destination: Fairbanks International Airport (FAI)
- Date: January 10, 2025

3. Flight SFO-AK789
- Price: $400
- Origin: San Francisco International Airport (SFO)
- Destination: Juneau International Airport (JNU)
- Date: January 20, 2025

4. Flight NYC-LA101
- Price: $250
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

5. Flight CHI-MIA202
- Price: $200
- Origin: Chicago O'Hare International Airport (ORD)
- Destination: Miami International Airport (MIA)
- Date: January 12, 2025

6. Flight BOS-SEA303
- Price: $120
- Origin: Boston Logan International Airport (BOS)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 12, 2025

7. Flight DFW-DEN404
- Price: $150
- Origin: Dallas/Fort Worth International Airport (DFW)
- Destination: Denver International Airport (DEN)
- Date: January 10, 2025

8. Flight ATL-HOU505
- Price: $180
- Origin: Hartsfield-Jackson Atlanta International Airport (ATL)
- Destination: George Bush Intercontinental Airport (IAH)
- Date: January 10, 2025
"""

# restrict how many requests this app can make to the LLM
usage_limits = UsageLimits(request_limit=15)

async def main():
    deps = Deps(
        web_page_text=None,
        req_origin='SFO',
        req_destination='ANC',
        req_date=datetime(2025, 1, 10)
    )
    message_history: list[ModelMessage] | None = None
    usage: Usage = Usage()
    # run the agent until a satisfactory flight is found
    while True:
        result = await search_agent.run(
            f'Find me a flight from {deps.req_origin} to {deps.req_destination} on {deps.req_date}',
            deps=deps,
            usage=usage,
            message_history=message_history,
            usage_limits=usage_limits,
        )
        if isinstance(result.data, NoFlightFound):
            print('No flight found')
            break
        else:
            flight = result.data
            print(f'Flight found: {flight}')
            answer = Prompt.ask(
                'Do you want to buy this flight, or keep searching? (buy/*search)',
                choices=['buy', 'search', ''],
                show_choices=False
            )
            if answer == 'buy':
                seat = await find_seat(usage)
                await buy_tickets(flight, seat)
                break
            else:
                message_history = result.all.messages(
                    result_tool_return_content='Please suggest another flight'
                )
                
    scraper_agent = WebScraperAgent()
    scrape_input = WebScrapeInput(url="https://www.example-flight-site.com/search?...") # Replace with actual URL
    scrape_output = await scraper_agent.run(scrape_input)
    
    if scrape_output.data:
        deps.web_page_text = scrape_output.data.text
    else:
        print("Failed to scrape flight information.")
        return

                
async def find_seat(usage: Usage) -> SeatPreference:
    message_history: list[ModelMessage] | None = None
    while True:
        answer = Prompt.ask('What seat would you like?')
        
        result = await seat_preference_agent.run(
            answer,
            message_history=message_history,
            usage=usage,
            usage_limits=usage_limits
        )
        if isinstance(result.data, SeatPreference):
            return result.data
        else:
            print('Could not understand seat preference. Please try again.')
            message_history = result.all.messages()
            
async def buy_tickets(flight_details: FlightDetails, seat: SeatPreference):
    print(f'Purchasing flight {flight_details=!r} {seat=!r}...')
    
if __name__ == '__main__':
    import asyncio
    asyncio.run(main())