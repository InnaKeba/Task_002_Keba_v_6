import os
import asyncio
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# ── 1. Ініціалізація LLM для CrewAI ────────────────────────────────────
llm = LLM(
    model="openrouter/google/gemini-2.5-flash", 
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

async def main():
    # ── 2. Ініціалізація клієнта MCP ───────────────────────────────────
    client = MultiServerMCPClient({
        "mcp": {
            "command": "python",
            "args": ["mcp_server.py"],
            "transport": "stdio"
        }
    })
    mcp_tools = await client.get_tools()

    # Функція-помічник для виклику конкретного MCP інструмента
    def invoke_mcp_tool(name: str, args: dict):
        mcp_tool = next(t for t in mcp_tools if name in t.name)
        return mcp_tool.invoke(args)

    # ── 3. Створення нативних інструментів CrewAI ──────────────────────
    @tool("load_sales_data")
    def load_sales_data_tool(period: str, region: str) -> str:
        """Завантаження даних продажів (період, регіон)."""
        return invoke_mcp_tool("load_sales_data", {"period": period, "region": region})

    @tool("calculate_kpis")
    def calculate_kpis_tool(period: str, metric: str, group_by: str) -> str:
        """Розрахунок ключових показників (metric, group_by)."""
        return invoke_mcp_tool("calculate_kpis", {"period": period, "metric": metric, "group_by": group_by})

    @tool("detect_sales_anomalies")
    def detect_sales_anomalies_tool(period: str, metric: str) -> str:
        """Виявлення аномалій у продажах (period, metric)."""
        return invoke_mcp_tool("detect_sales_anomalies", {"period": period, "metric": metric})

    @tool("export_dashboard")
    def export_dashboard_tool(report_name: str, format: str) -> str:
        """Експорт або публікація dashboard."""
        return invoke_mcp_tool("export_dashboard", {"report_name": report_name, "format": format})

    # ── 4. Створення агентів ───────────────────────────────────────────
    data_loader = Agent(
        role='Data Loader',
        goal='Завантажити точні сирі дані продажів за вказаний період та регіон',
        backstory='Ти експерт з баз даних. Твоя єдина задача — використовувати інструменти для вивантаження даних і передавати їх далі.',
        tools=[load_sales_data_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    sales_analyst = Agent(
        role='Sales Analyst',
        goal='Розрахувати ключові показники (KPI) на основі наданих сирих даних',
        backstory='Досвідчений фінансовий аналітик. Ти перетворюєш масиви даних на зрозумілі бізнес-метрики.',
        tools=[calculate_kpis_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    anomaly_detector = Agent(
        role='Anomaly Detector',
        goal='Виявити аномалії або нестандартні патерни у фінансових показниках',
        backstory='Спеціаліст з машинного навчання. Твоє завдання — перевірити дані на наявність відхилень.',
        tools=[detect_sales_anomalies_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    report_reviewer = Agent(
        role='Report Reviewer',
        goal='Перевірити метрики та експортувати фінальний дашборд',
        backstory='Ти фінальний рев\'юер звітів. Твоя мета — підготувати звіт та ініціювати його експорт.',
        tools=[export_dashboard_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # ── 5. Завдання (Tasks) ─────────────────────────────────────────
    task1 = Task(
        description='Завантаж дані продажів за період "2026-Q1" для регіону "West". Використовуй інструмент load_sales_data.',
        expected_output='JSON з сирими даними про продажі (revenue, orders, avg_check).',
        agent=data_loader
    )

    task2 = Task(
        description='Використовуючи дані від Data Loader, розрахуй KPI (metric="revenue", group_by="region"). Використовуй інструмент calculate_kpis.',
        expected_output='Текстовий звіт про KPI.',
        agent=sales_analyst
    )

    task3 = Task(
        description='Перевір розраховані метрики на наявність аномалій (period="2026-Q1", metric="revenue"). Використовуй інструмент detect_sales_anomalies.',
        expected_output='Короткий звіт про знайдені аномалії або їх відсутність.',
        agent=anomaly_detector
    )

    task4 = Task(
        description='Використовуючи результати попередніх завдань, підготуй фінальний висновок та експортуй дашборд (report_name="Q1_West_Report", format="PDF").',
        expected_output='Статус експорту дашборду та фінальний висновок.',
        agent=report_reviewer,
        human_input=True 
    )

    # ── 6. Команда (Crew) ───────────────────────────────────────────
    sales_crew = Crew(
        agents=[data_loader, sales_analyst, anomaly_detector, report_reviewer],
        tasks=[task1, task2, task3, task4],
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=True
    )

    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("🚀 Запуск CrewAI MAS (через протокол MCP)...")
    print("-" * 50)
    
    result = await sales_crew.kickoff_async()
    
    print("\n==================================================")
    print("Фінальний результат CREWAI:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())