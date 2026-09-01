from mcp.server.fastmcp import FastMCP
import json

mcp = FastMCP("sales_dashboard_mcp")

# Mock-дані для аналітики продажів
SALES_DATA = {
    ("2026-Q1", "West"): {
        "revenue": 245000,
        "orders": 980,
        "avg_check": 250.0
    },
    ("2026-Q1", "Central"): {
        "revenue": 310000,
        "orders": 1105,
        "avg_check": 280.5
    }
}

ANOMALIES = {
    ("2026-Q1", "revenue"): [
        {"month": "2026-02", "issue": "Unexpected spike in revenue"},
        {"month": "2026-03", "issue": "Drop in conversion rate"}
    ]
}

@mcp.tool()
def load_sales_data(period: str, region: str) -> str:
    """Завантаження даних продажів."""
    data = SALES_DATA.get((period, region))
    if not data:
        return json.dumps({"error": f"No sales data for {period} and {region}"})
    return json.dumps({
        "period": period,
        "region": region,
        **data
    })

@mcp.tool()
def calculate_kpis(period: str, metric: str, group_by: str) -> str:
    """Розрахунок ключових показників (revenue, avg_check тощо)."""
    # Імітація успішного розрахунку
    return json.dumps({
        "period": period,
        "metric": metric,
        "group_by": group_by,
        "result": f"Calculated {metric} grouped by {group_by}. Trend is stable."
    })

@mcp.tool()
def detect_sales_anomalies(period: str, metric: str) -> str:
    """Виявлення аномалій у продажах."""
    anomalies = ANOMALIES.get((period, metric), [])
    if not anomalies:
        return json.dumps({"message": "No anomalies detected."})
    return json.dumps({
        "period": period,
        "metric": metric,
        "anomalies": anomalies
    })

@mcp.tool()
def export_dashboard(report_name: str, format: str) -> str:
    """Експорт або публікація dashboard (РИЗИКОВИЙ ІНСТРУМЕНТ - HITL!)."""
    return json.dumps({
        "report_name": report_name,
        "format": format,
        "status": "pending_human_approval",
        "message": f"Dashboard '{report_name}' is ready for export in {format} format."
    })

if __name__ == "__main__":
    mcp.run(transport="stdio")