import os
import operator
import asyncio
import aiosqlite
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import interrupt, Command
from langchain_mcp_adapters.client import MultiServerMCPClient
from guardrails import input_guardrail, tool_guardrail, output_guardrail

load_dotenv()

# ── 1. СТРУКТУРА СТАНУ ТА SUPERVISOR ─────────────────────────────────────
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

async def supervisor_node(state: AgentState) -> dict:
    """Координатор, який вирішує, хто діє наступним (Асинхронний)."""
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
    response = await llm.with_structured_output(Router).ainvoke(messages)
    return {"next_agent": response.next_agent}

# ── 2. ОСНОВНА АСИНХРОННА ФУНКЦІЯ (MCP Client + Graph) ───────────────────
async def main():
    client = MultiServerMCPClient({
        "mcp": {
            "command": "python",
            "args": ["mcp_server.py"],
            "transport": "stdio"
        }
    })
    all_tools = await client.get_tools()

    async with aiosqlite.connect("mas_state.db") as db_conn:
        
        def get_tools_for_agent(agent_name: str):
            return [t for t in all_tools if tool_guardrail(agent_name, t.name.replace("mcp_", ""))]
        def get_tools_for_agent(agent_name: str):
            return [t for t in all_tools if tool_guardrail(agent_name, t.name.replace("mcp_", ""))]

        def create_agent_node(agent_name: str, system_prompt: str, tools: list):
            agent_llm = llm.bind_tools(tools)
            
            async def node(state: AgentState) -> dict:
                strict_prompt = system_prompt + "\nВАЖЛИВО: ТИ ПОВИНЕН викликати свій інструмент! Не задавай питань. Просто виконай дію."
                messages = [SystemMessage(content=strict_prompt)] + state["messages"]
                response = await agent_llm.ainvoke(messages)
                
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    tool_outputs = []
                    for tc in response.tool_calls:
                        # Вбудований HITL Guardrail перед викликом експорту
                        if "export_dashboard" in tc["name"]:
                            approval = interrupt({
                                "action": "export_dashboard",
                                "message": f"Увага: запит на експорт дашборду. Схвалити?"
                            })
                            if not (isinstance(approval, dict) and approval.get('approved')):
                                return {"messages": [AIMessage(content="Експорт відхилено оператором.", name=agent_name)]}
                        
                        tool_fn = {t.name: t for t in tools}.get(tc["name"])
                        if tool_fn:
                            res = await tool_fn.ainvoke(tc["args"])
                            tool_outputs.append(f"[{agent_name} викликав {tc['name']}]: {res}")
                    return {"messages": [AIMessage(content="\n".join(tool_outputs), name=agent_name)]}
                else:
                    fallback_msg = f"[{agent_name} завершив свою частину аналізу. Передай хід наступному.]"
                    return {"messages": [AIMessage(content=fallback_msg, name=agent_name)]}
            return node

        # Ініціалізація вузлів з динамічними інструментами
        data_loader_node = create_agent_node(
            "data_loader", "Ти Data Loader. Викликай інструмент завантаження (period='2026-Q1', region='West').", get_tools_for_agent("data_loader")
        )
        sales_analyst_node = create_agent_node(
            "sales_analyst", "Ти Sales Analyst. Викликай інструмент розрахунку KPI (metric='revenue', group_by='region').", get_tools_for_agent("sales_analyst")
        )
        anomaly_detector_node = create_agent_node(
            "anomaly_detector", "Ти Anomaly Detector. Викликай інструмент виявлення аномалій.", get_tools_for_agent("anomaly_detector")
        )
        report_reviewer_node = create_agent_node(
            "report_reviewer", "Ти Report Reviewer. Викликай інструмент експорту (format='PDF').", get_tools_for_agent("report_reviewer")
        )

        # Побудова графа
        graph = StateGraph(AgentState)
        graph.add_node("supervisor", supervisor_node)
        graph.add_node("data_loader", data_loader_node)
        graph.add_node("sales_analyst", sales_analyst_node)
        graph.add_node("anomaly_detector", anomaly_detector_node)
        graph.add_node("report_reviewer", report_reviewer_node)

        graph.add_edge(START, "supervisor")
        for member in members:
            graph.add_edge(member, "supervisor")

        graph.add_conditional_edges(
            "supervisor",
            lambda state: state["next_agent"],
            {**{m: m for m in members}, "FINISH": END}
        )

        # Компіляція з асинхронним збереженням стану
        saver = AsyncSqliteSaver(db_conn)
        app = graph.compile(checkpointer=saver)

        # ── 3. ДЕМОНСТРАЦІЯ ───────────────────────────────────────────────
        import sys
        sys.stdout.reconfigure(encoding='utf-8')
        
        user_request = "Проаналізуй продажі за 2026-Q1 в регіоні West: завантаж дані, обчисли виручку, вияви аномалії та підготуй дашборд 'Q1_West_Report' у форматі PDF."
        
        # 1. INPUT GUARDRAIL
        if not input_guardrail(user_request):
            print("Виявлено підозрілий запит (Prompt Injection). Роботу зупинено.")
            sys.exit()

        config = {"configurable": {"thread_id": "mas_session_async_001"}}
        
        print(f"\n[USER]: {user_request}\n")
        print("-" * 50)
        
        # 2. Асинхронне виконання графа із зупинкою на HITL
        async for event in app.astream({"messages": [HumanMessage(content=user_request)]}, config):
            if "__interrupt__" in event:
                interrupt_data = event["__interrupt__"][0].value
                print(f"\nГРАФ ПРИЗУПИНЕНО!")
                print(f"Повідомлення: {interrupt_data['message']}")
                break
            
            for key, value in event.items():
                if key != "supervisor":
                    msg = value["messages"][-1].content
                    print(f"[{key.upper()}]: {msg}")

        # 3. Імітація схвалення оператором
        print("\n[Оператор]: Схвалює публікацію дашборду...")
        
        final_output = ""
        async for event in app.astream(Command(resume={"approved": True}), config):
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

if __name__ == "__main__":
    asyncio.run(main())