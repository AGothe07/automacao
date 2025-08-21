from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage  
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os


# Carrega a API KEY
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


# Definição do modelo
llm_model = ChatOpenAI(model="gpt-3.5-turbo", api_key=API_KEY)

# Define o prompt do sistema
system_message = SystemMessage(content="""
Você é uma assistente. Se o usuário pedir contas, use a ferramente 'somar'. Caso o contrário, apenas responda normalmente
""")

# definindo a ferramenta de soma
@tool("somar")
def somar(valores: str) -> str:
    """Soma dois numeros separados por virgura"""
    try:
        a, b = map(float, valores.split(","))
        return str(a + b)
    except Exception as e:
        return f"Erro ao somar: {str(e)}"
    

# Criação do agente com LangGraph
tools = [somar]
graph = create_react_agent(
    model=llm_model,
    tools=tools,
    prompt=system_message
)
export_graph = graph

# extrair a resposta final
def extrair_resposta_final(result):
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
    if ai_messages:
        return ai_messages[-1].content
    else:
        return "Nenhyma resposta final encontrada"
    
# testando o agente
if __name__ == "__main__":
    entrada1 = HumanMessage(content="Quanto é 8 + 5")
    result1 = export_graph.invoke({"messages": [entrada1]})
    for m in result1["messages"]:
        print(m)
    resposta_texto_1 = extrair_resposta_final(result1)
    print("resposta 1: ", resposta_texto_1) 
    print()
    
    entrada2 = HumanMessage(content="Quem pintou a monalisa?")
    result2 = export_graph.invoke({"messages": [entrada2]})
    for m in result2["messages"]:
        print(m)
    resposta_texto_2 = extrair_resposta_final(result2)
    print("resposta 2: ", resposta_texto_2)




    png_bytes = graph.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API
)

    with open("grafo_exemplo3.png", "wb") as f:
        f.write(png_bytes)