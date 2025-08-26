from dotenv import load_dotenv
import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")


agent = Agent(
    model = OpenAIChat(id="gpt-4o-mini", api_key=API_KEY),
    tools=[YFinanceTools(stock_price=True)],
    instructions=[
        "Usar tabelas para exibir os dadods",
        "Exibir apenas o relatório , nenhum outro texto"
    ],
    markdown=True
)

agent.print_response("Qual é o preço da ação da apple?", stream=True)