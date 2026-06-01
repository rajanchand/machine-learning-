"""
report_generator.py — Generate downloadable HTML reports with all results.
"""

import base64
import io
import datetime
from src.utils import get_logger

logger = get_logger("report_generator")


def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64-encoded PNG string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return encoded


def generate_html_report(
    title: str = "Network Anomaly Detection — Evaluation Report",
    dataset_info: dict = None,
    model_results: dict = None,
    comparison_table_html: str = "",
    figures: dict = None,
    ablation_table_html: str = "",
):
    """
    Generate a self-contained HTML report.

    Args:
        title: Report title
        dataset_info: dict with dataset metadata
        model_results: dict mapping model_name → metrics_dict
        comparison_table_html: pre-rendered HTML table from pandas
        figures: dict mapping figure_name → base64-encoded PNG
        ablation_table_html: pre-rendered HTML table
    Returns:
        HTML string
    """
    dataset_info = dataset_info or {}
    model_results = model_results or {}
    figures = figures or {}

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build figure HTML
    figures_html = ""
    for name, b64_img in figures.items():
        figures_html += f"""
        <div class="figure">
            <h3>{name}</h3>
            <img src="data:image/png;base64,{b64_img}" alt="{name}" />
        </div>
        """

    # Build model results HTML
    results_html = ""
    for model_name, metrics in model_results.items():
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items())
        results_html += f"""
        <h3>{model_name}</h3>
        <table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>{rows}</tbody></table>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0f0f1a; color: #e0e0e0;
            padding: 40px; line-height: 1.6;
        }}
        h1 {{ color: #7c3aed; margin-bottom: 8px; font-size: 28px; }}
        h2 {{ color: #a78bfa; margin: 30px 0 15px; border-bottom: 1px solid #2d2d44; padding-bottom: 8px; }}
        h3 {{ color: #c4b5fd; margin: 20px 0 10px; }}
        .meta {{ color: #888; font-size: 14px; margin-bottom: 30px; }}
        table {{
            border-collapse: collapse; width: 100%; margin: 15px 0;
            background: #1a1a2e; border-radius: 8px; overflow: hidden;
        }}
        th, td {{ padding: 10px 16px; text-align: left; border-bottom: 1px solid #2d2d44; }}
        th {{ background: #16162a; color: #a78bfa; font-weight: 600; }}
        tr:hover {{ background: #1e1e38; }}
        .figure {{ margin: 25px 0; text-align: center; }}
        .figure img {{ max-width: 100%; border-radius: 8px; border: 1px solid #2d2d44; }}
        .section {{ margin: 30px 0; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="meta">Generated on {now} | Master's Dissertation — ML-based Network Anomaly Detection</p>

    <div class="section">
        <h2>1. Dataset Information</h2>
        <table>
            <tbody>
                {"".join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in dataset_info.items())}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>2. Model Evaluation Results</h2>
        {results_html}
    </div>

    <div class="section">
        <h2>3. Model Comparison</h2>
        {comparison_table_html}
    </div>

    <div class="section">
        <h2>4. Visualisations</h2>
        {figures_html}
    </div>

    <div class="section">
        <h2>5. Ablation Study</h2>
        {ablation_table_html if ablation_table_html else "<p>No ablation data available.</p>"}
    </div>
</body>
</html>"""

    logger.info("HTML report generated successfully")
    return html
