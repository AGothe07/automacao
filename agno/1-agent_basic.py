from dotenv import load_dotenv
import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

agent = Agent(
    model=OpenAIChat(
        api_key=API_KEY,
        id="gpt-4o-mini"
    ),
    markdown=True
)

agent.print_response("Compartilhe uma história engraçada sobre programação")








