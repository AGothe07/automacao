from agno import agent
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.youtube import YouTubeTools
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini", api_key=API_KEY),
    tools=[YouTubeTools()],
    show_tool_calls=True,
    description="""Você é um agente do Youtube, 
    obtenha as legendas de um video 
    do youtube e responda as perguntas"""
)

agent.print_response("Sumarize esse video: https://www.youtube.com/watch?v=AHqqzUkjRjw")