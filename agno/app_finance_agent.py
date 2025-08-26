from dotenv import load_dotenv
import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools
from agno.tools.reasoning import ReasoningTools
from agno.playground import Playground, serve_playground_app

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

app = Playground(agents = [agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("app_finance_agent:app", reload=True)