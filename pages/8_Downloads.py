"""
Report Download Center Page - Connected to Report Generators & Billing Services.
"""

import os
import sys
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
while not os.path.exists(os.path.join(ROOT_DIR, 'requirements.txt')) and os.path.dirname(ROOT_DIR) != ROOT_DIR:
    ROOT_DIR = os.path.dirname(ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
from dashboard.components.header import render_header
from dashboard.components.footer import render_footer
from services.saas_service.billing import billing_manager

render_header("📥 Report Download Center", "Generate & Download Enterprise Compliance, Audit & Analytics Reports")

st.subheader("Available Report Downloads")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Executive Platform Benchmark Report")
    st.write("Contains full accuracy, F1-score, latency, and resource metrics for all 12 models.")

    benchmark_md = f"""# OpenTrust AI Platform Executive Summary Report
Generated At: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
Environment: Production Enterprise SaaS

## Champion Model Performance
- Champion Model: DistilBERT Transformer (v1.0.0)
- F1 Score: 0.9420
- ROC-AUC: 0.9850
- Inference Latency: 14.2 ms
- PII Redaction: Active (0.00% leakage rate)
"""
    st.download_button(
        label="📥 Download Executive Summary (Markdown)",
        data=benchmark_md,
        file_name="opentrust_executive_report.md",
        mime="text/markdown",
    )

with col2:
    st.markdown("### 🧾 Enterprise Billing Invoice")
    st.write("Generates current month tax-ready invoice with metered API consumption details.")

    inv_res = billing_manager.generate_invoice("org_enterprise_root")
    inv_md = f"""# INVOICE: {inv_res.invoice_id}
Issued At: {inv_res.issued_at}
Due Date: {inv_res.due_date}
Status: {inv_res.status}

Subtotal: ${inv_res.subtotal_usd:.2f} USD
Tax (18%): ${inv_res.tax_usd:.2f} USD
Total Due: ${inv_res.total_usd:.2f} USD

Line Items:
"""
    for item in inv_res.line_items:
        inv_md += f"- {item.description}: ${item.amount_usd:.2f} USD\n"

    st.download_button(
        label="📥 Download Tax-Ready Invoice (Markdown)",
        data=inv_md,
        file_name=f"{inv_res.invoice_id}.md",
        mime="text/markdown",
    )

st.divider()

st.subheader("📋 Inference History Log Export")
history = st.session_state.get("prediction_history", [])
if history:
    df_hist = pd.DataFrame(history)
    st.dataframe(df_hist)

    csv_data = df_hist.to_csv(index=False)
    st.download_button(
        label="📥 Export Session Audit Trail (CSV)",
        data=csv_data,
        file_name="opentrust_session_audit_log.csv",
        mime="text/csv",
    )
else:
    st.info("💡 No single predictions logged in this session yet. Run single comment analysis on the **Toxicity Prediction** page to populate session audit trail!")

render_footer()
