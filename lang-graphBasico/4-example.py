from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage 
from langgraph.graph import StateGraph
from pydantic import BaseModel
from dotenv import load_dotenv
import os


# Carrega a API KEY
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


# Definição do modelo
llm_model = ChatOpenAI(model="gpt-3.5-turbo", api_key=API_KEY)

# Definir o estado do Graph
class GraphState(BaseModel):
    input: str
    output: str
    tipo: str = None

#Função de realizar calculo
def realizar_calculo(state: GraphState) -> GraphState:
    return GraphState(input=state.input, output="Resposta de calculo ficticio: 42")

# Função para responder perguntar normais
def responder_curiosidade(state: GraphState) -> GraphState:
    response = llm_model.invoke([HumanMessage(content=state.input)])
    return GraphState(input=state.input, output=response.content)

# função para tratar perguntas não reconhecidas 
def responder_erro(state: GraphState) -> GraphState:
    return GraphState(input=state.input, output="Desculpe, não entendi a sua pergunta")

# Função de classificação dos nodes
def classificar(state: GraphState) -> GraphState:
    pergunta = state.input.lower()
    if any(palavra in pergunta for palavra in ["soma", "quanto é", "+", "calcular"]):
        tipo = "calculo"
    elif any(palavra in pergunta for palavra in ["quem", "onde", "porque", "qual"]):
        tipo = "Curiosidade"
    else:
        tipo = "desconhecido"
    return GraphState(input=state.input, output="", tipo=tipo)

#Criando Graph e adicionadno os node
graph = StateGraph(GraphState)
graph.add_node("classificar", classificar)
graph.add_node("realizar_calculo", realizar_calculo)
graph.add_node("responder_curiosidade", responder_curiosidade)
graph.add_node("responder_erro", responder_erro)

#Adicionadno condicionais
graph.add_conditional_edges(
    "classificar",
    lambda state:{
        "calculo": "realizar_calculo",
        "Curiosidade": "responder_curiosidade",
        "desconhecido": "responder_erro"
    } [state.tipo]
)

# Definindo entrada e saída e compilação
graph.set_entry_point("classificar")
graph.set_finish_point(["realizar_calculo", "responder_curiosidade", "responder_erro"])
export_graph = graph.compile()

#Testando o projeto
if __name__ == "__main__":
    exemplos = [
        "Quanto é 10 + 5",
        "Quem inventou a eletricidade",
        "Me diga um comando especial"
    ]

    for exemplo in exemplos:
        result = export_graph.invoke(GraphState(input=exemplo, output=""))
        print(f"Pergunta: {exemplo}\nResposta: {result['output']}\n")

