import os
import operator
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

# Імпортуємо логіку MCP-сервера та Guardrails
from mcp_server import load_sales_data, calculate_kpis, detect_sales_anomalies, export_dashboard
from guardrails import input_guardrail, tool_guardrail, output_guardrail

load_dotenv()

# ── 1. ІНТЕГРАЦІЯ ІНСТРУМЕНТІВ ───────────────────────────

@tool
def tool_load_sales_data(period: str, region: str) -> str:
    """Завантаження даних продажів (період, регіон)."""
    if not tool_guardrail("data_loader", "load_sales_data"): return "Access Denied"
    return load_sales_data(period, region)

@tool
def tool_calculate_kpis(period: str, metric: str, group_by: str) -> str:
    """Розрахунок ключових показників."""
    if not tool_guardrail("sales_analyst", "calculate_kpis"): return "Access Denied"
    return calculate_kpis(period, metric, group_by)

@tool
def tool_detect_sales_anomalies(period: str, metric: str) -> str:
    """Виявлення аномалій у продажах."""
    if not tool_guardrail("anomaly_detector", "detect_sales_anomalies"): return "Access Denied"
    return detect_sales_anomalies(period, metric)

@tool
def tool_export_dashboard(report_name: str, format: str) -> str:
    """Експорт або публікація dashboard (ПОТРЕБУЄ ПІДТВЕРДЖЕННЯ)."""
    if not tool_guardrail("report_reviewer", "export_dashboard"): return "Access Denied"
    
    # HITL. Зупинка графа для підтвердження ризикової операції
    approval = interrupt({
        "action": "export_dashboard",
        "message": f"Увага: запит на експорт дашборду '{report_name}' у форматі {format}. Схвалити?"
    })
    
    if isinstance(approval, dict) and approval.get('approved'):
        return export_dashboard(report_name, format)
    else:
        return "Експорт відхилено оператором."

# ── 2. СТРУКТУРА СТАНУ ТА SUPERVISOR ─────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str

members = ["data_loader", "sales_analyst", "anomaly_detector", "report_reviewer"]

class Router(BaseModel):
    next_agent: Literal["data_loader", "sales_analyst", "anomaly_detector", "report_reviewer", "FINISH"] = Field(
        description="Виберіть наступного агента для виконання роботи або FINISH, якщо завдання повністю виконано."
    )

llm = ChatOpenAI(
    model="google/gemini-2.5-flash", 
    api_key=os.getenv("OPENROUTER_API_KEY"), 
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1
)

# ── 3. Nodes ─────────────────────────────────────────────────────
def supervisor_node(state: AgentState) -> dict:
    """Координатор, який вирішує, хто діє наступним."""
    system_prompt = (
        "Ти Supervisor у команді аналітики продажів.\n"
        "Твоя мета - координувати процес обробки запиту користувача кроками.\n"
        "Учасники команди:\n"
        " - data_loader: завантажує сирі дані (ПЕРШИЙ КРОК ЗАВЖДИ)\n"
        " - sales_analyst: розраховує метрики та KPI (ТІЛЬКИ ПІСЛЯ data_loader)\n"
        " - anomaly_detector: шукає аномалії в даних (ТІЛЬКИ ПІСЛЯ sales_analyst)\n"
        " - report_reviewer: готує фінальний висновок та робить експорт (ОСТАННІЙ КРОК)\n"
        "ПРАВИЛО: Передавай хід агентам строго по черзі. Не викликай одного агента двічі поспіль. "
        "Якщо звіт експортовано, поверни FINISH."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    return {"next_agent": response.next_agent}

def create_agent_node(agent_name: str, system_prompt: str, tools: list):
    agent_llm = llm.bind_tools(tools)
    def node(state: AgentState) -> dict:
        strict_prompt = system_prompt + "\nВАЖЛИВО: ТИ ПОВИНЕН викликати свій інструмент! Не задавай питань. Просто виконай дію."
        messages = [SystemMessage(content=strict_prompt)] + state["messages"]
        response = agent_llm.invoke(messages)
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_outputs = []
            for tc in response.tool_calls:
                tool_fn = {t.name: t for t in tools}.get(tc["name"])
                if tool_fn:
                    res = tool_fn.invoke(tc["args"])
                    tool_outputs.append(f"[{agent_name} викликав {tc['name']}]: {res}")
            return {"messages": [AIMessage(content="\n".join(tool_outputs), name=agent_name)]}
        else:
            fallback_msg = f"[{agent_name} завершив свою частину аналізу. Передай хід наступному.]"
            return {"messages": [AIMessage(content=fallback_msg, name=agent_name)]}
    return node

data_loader_node = create_agent_node(
    "data_loader", "Ти Data Loader. Викликай tool_load_sales_data (period='2026-Q1', region='West').", [tool_load_sales_data]
)
sales_analyst_node = create_agent_node(
    "sales_analyst", "Ти Sales Analyst. Викликай tool_calculate_kpis (metric='revenue', group_by='region').", [tool_calculate_kpis]
)
anomaly_detector_node = create_agent_node(
    "anomaly_detector", "Ти Anomaly Detector. Викликай tool_detect_sales_anomalies.", [tool_detect_sales_anomalies]
)
report_reviewer_node = create_agent_node(
    "report_reviewer", "Ти Report Reviewer. Викликай tool_export_dashboard (format='PDF').", [tool_export_dashboard]
)
# ── 4. ГРАФ ────────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("data_loader", data_loader_node)
graph.add_node("sales_analyst", sales_analyst_node)
graph.add_node("anomaly_detector", anomaly_detector_node)
graph.add_node("report_reviewer", report_reviewer_node)

graph.add_edge(START, "supervisor")

for member in members:
    graph.add_edge(member, "supervisor") # повернення управління до Supervisor після дії агента

graph.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {**{m: m for m in members}, "FINISH": END}
)

import sqlite3
conn = sqlite3.connect("mas_state.db", check_same_thread=False)
saver = SqliteSaver(conn)
app = graph.compile(checkpointer=saver)


# ── 5. Демонстрація ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    user_request = "Проаналізуй продажі за 2026-Q1 в регіоні West: завантаж дані, обчисли виручку, вияви аномалії та підготуй дашборд 'Q1_West_Report' у форматі PDF."
    
    # 1. INPUT GUARDRAIL
    if not input_guardrail(user_request):
        print("Виявлено підозрілий запит (Prompt Injection). Роботу зупинено.")
        sys.exit()

    config = {"configurable": {"thread_id": "mas_session_001"}}
    
    print(f"\n[USER]: {user_request}\n")
    print("-" * 50)
    
    # 2. Виконання графа із зупинкою на HITL
    for event in app.stream({"messages": [HumanMessage(content=user_request)]}, config):
        if "__interrupt__" in event:
            interrupt_data = event["__interrupt__"][0].value
            print(f"\n ГРАФ ПРИЗУПИНЕНО!")
            print(f"Повідомлення: {interrupt_data['message']}")
            break
        
        for key, value in event.items():
            if key != "supervisor":
                msg = value["messages"][-1].content
                print(f"[{key.upper()}]: {msg}")

    # 3. Імітація схвалення оператором
    print("\n[Оператор]: Схвалює публікацію дашборду...")
    
    final_output = ""
    for event in app.stream(Command(resume={"approved": True}), config):
        for key, value in event.items():
            if key != "supervisor":
                msg = value["messages"][-1].content
                print(f"[{key.upper()}]: {msg}")
                final_output = msg
                
    # 4. OUTPUT GUARDRAIL
    safe_output = output_guardrail(final_output)
    print("\n" + "=" * 50)
    print("ФІНАЛЬНИЙ БЕЗПЕЧНИЙ ВИВІД (PII Redacted):")
    print(safe_output)