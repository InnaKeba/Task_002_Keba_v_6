from mcp.server.fastmcp import FastMCP
import json
import os

mcp = FastMCP("sales_dashboard_mcp")

# Mock-дані для аналітики продажів
SALES_DATA = {
    ("2026-Q1", "West"): {
        "revenue": 245000,
        "orders": 980,
        "avg_check": 250.0
    }
}

ANOMALIES = {
    ("2026-Q1", "revenue"): [
        {"month": "2026-02", "issue": "Unexpected spike in revenue"},
        {"month": "2026-03", "issue": "Drop in conversion rate"}
    ]
}

STATE_FILE = "mcp_state.json"

def get_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"kpis_calculated": False, "anomalies_detected": False}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

@mcp.tool()
def load_sales_data(period: str, region: str) -> str:
    """Завантаження даних продажів."""
    # Скидаємо Sequencing Guardrail при старті нового процесу
    save_state({"kpis_calculated": False, "anomalies_detected": False})
    
    data = SALES_DATA.get((period, region))
    if not data:
        return json.dumps({"error": f"No sales data for {period} and {region}"})
    return json.dumps({"period": period, "region": region, **data})

@mcp.tool()
def calculate_kpis(period: str, metric: str, group_by: str) -> str:
    """Розрахунок ключових показників."""
    state = get_state()
    state["kpis_calculated"] = True
    save_state(state)
    
    return json.dumps({
        "period": period,
        "metric": metric,
        "group_by": group_by,
        "result": "KPIs calculated."
    })

@mcp.tool()
def detect_sales_anomalies(period: str, metric: str) -> str:
    """Виявлення аномалій у продажах."""
    state = get_state()
    state["anomalies_detected"] = True
    save_state(state)
    
    anomalies = ANOMALIES.get((period, metric), [])
    return json.dumps({
        "period": period,
        "metric": metric,
        "anomalies": anomalies
    })

@mcp.tool()
def export_dashboard(report_name: str, format: str) -> str:
    """Експорт або публікація dashboard (РИЗИКОВИЙ ІНСТРУМЕНТ - HITL!)."""
    state = get_state()
    
    # SEQUENCING GUARDRAIL
    if not (state.get("kpis_calculated") and state.get("anomalies_detected")):
        return json.dumps({"error": "SEQUENCING GUARDRAIL VIOLATION: Розрахуйте KPI та виявіть аномалії перед експортом!"})
    
    return json.dumps({
        "report_name": report_name,
        "format": format,
        "status": "pending_human_approval",
        "message": f"Dashboard '{report_name}' is ready for export in {format} format."
    })

if __name__ == "__main__":
    mcp.run(transport="stdio")