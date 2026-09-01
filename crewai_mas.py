import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from mcp_server import load_sales_data, calculate_kpis, detect_sales_anomalies

load_dotenv()

# ── 1. Ініціалізація LLM для CrewAI ────────────────────────────────────
llm = LLM(
    model="openrouter/google/gemini-2.5-flash", 
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# ── 2. Створення CrewAI інструментів ───────────────────────────────────

@tool("Load Sales Data")
def crew_load_sales_data(period: str, region: str) -> str:
    """Завантаження даних продажів (період, регіон)."""
    return load_sales_data(period, region)

@tool("Calculate KPIs")
def crew_calculate_kpis(period: str, metric: str, group_by: str) -> str:
    """Розрахунок ключових показників (наприклад, metric='revenue', group_by='region')."""
    return calculate_kpis(period, metric, group_by)

@tool("Detect Sales Anomalies")
def crew_detect_sales_anomalies(period: str, metric: str) -> str:
    """Виявлення аномалій у продажах (наприклад, period='2026-Q1', metric='revenue')."""
    return detect_sales_anomalies(period, metric)


# ── 3. Створення агентів ────────────────────────────────────────────────

data_loader = Agent(
    role='Data Loader',
    goal='Завантажити точні сирі дані продажів за вказаний період та регіон',
    backstory='Ти експерт з баз даних. Твоя єдина задача — використовувати інструменти для вивантаження даних і передавати їх далі.',
    tools=[crew_load_sales_data],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

sales_analyst = Agent(
    role='Sales Analyst',
    goal='Розрахувати ключові показники (KPI) на основі наданих сирих даних',
    backstory='Досвідчений фінансовий аналітик. Ти перетворюєш масиви даних на зрозумілі бізнес-метрики.',
    tools=[crew_calculate_kpis],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

anomaly_detector = Agent(
    role='Anomaly Detector',
    goal='Виявити аномалії або нестандартні патерни у фінансових показниках',
    backstory='Спеціаліст з машинного навчання. Твоє завдання — перевірити дані на наявність відхилень.',
    tools=[crew_detect_sales_anomalies],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# ── 4. Tasks ─────────────────────────────────────────
task1 = Task(
    description='Завантаж дані продажів за період "2026-Q1" для регіону "West". Використовуй інструмент Load Sales Data.',
    expected_output='JSON з сирими даними про продажі (revenue, orders, avg_check).',
    agent=data_loader
)

task2 = Task(
    description='Використовуючи дані від Data Loader, розрахуй KPI (metric="revenue", group_by="region"). Використовуй інструмент Calculate KPIs.',
    expected_output='Текстовий звіт про KPI.',
    agent=sales_analyst
)

task3 = Task(
    description='Перевір розраховані метрики на наявність аномалій (period="2026-Q1", metric="revenue"). Використовуй інструмент Detect Sales Anomalies.',
    expected_output='Короткий звіт про знайдені аномалії або їх відсутність.',
    agent=anomaly_detector
)

# ── 5. Crew ─────────────────────────────────────────
sales_crew = Crew(
    agents=[data_loader, sales_analyst, anomaly_detector],
    tasks=[task1, task2, task3],
    process=Process.hierarchical,
    manager_llm=llm,
    verbose=True
)

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("Запуск CrewAI MAS...")
    print("-" * 50)
    
    result = sales_crew.kickoff()
    
    print("\n==================================================")
    print("Фінальний результат CREWAI:")
    print(result)