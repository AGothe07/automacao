from dotenv import load_dotenv
import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools
from agno.tools.reasoning import ReasoningTools

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini", api_key=API_KEY),
    tools=[
        ReasoningTools(),
        YFinanceTools(
            stock_price=True,
            company_info=True,
            analyst_recommendations=True,
            company_news=True,
        )
    ],
    instructions=[
        "Usar tabelas para exibir os dados",
        "Exibir apenas o relatório final, nenhum outro texto"
    ],
    markdown=True
)

agent.print_response("Escreva um realatório sobre a empresa Banco do Brasil",
                     stream=True,
                     show_full_reasoning=True,
                     stream_intermediate_steps=True)    