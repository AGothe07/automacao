from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel
from dotenv import load_dotenv
import os


# Carrega a API KEY
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


# Definição do modelo
llm_model = ChatOpenAI(model="gpt-3.5-turbo", api_key=API_KEY)

# Definição do GraphState
class GraphState(BaseModel):
    input: str
    output: str

# Função de resposta
def responder(state):
    input_message = state.input
    response = llm_model.invoke([HumanMessage(content=input_message)])
    return GraphState(input=input_message, output=response.content)

# criando o Graph
graph = StateGraph(GraphState)
graph.add_node("responder", responder)
graph.set_entry_point("responder")
graph.set_finish_point("responder")

#compilando o Grafo
export_graph = graph.compile()
 
# testando o agente
if __name__ == "__main__":
    result = export_graph.invoke(GraphState(input="Quem descobriu a America", output=""))
    print(result)

    #  Visualizar o Grafo
    print(export_graph.get_graph().draw_mermaid())
    