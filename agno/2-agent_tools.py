from dotenv import load_dotenv
import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

agent = Agent(
    model=OpenAIChat(id = "gpt-4o-mini",api_key=API_KEY),
    tools=[YFinanceTools(stock_price=True)], #Habilita consulta de preços de ações
    markdown=True
)

agent.print_response("Qual o preço do Bitcoin em Dolar?", stream=True)
