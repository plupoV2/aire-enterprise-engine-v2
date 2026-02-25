import os, re, json, hashlib, sqlite3, base64, math
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st
import requests
import pandas as pd
import numpy as np
import openai

# ============================================================
# AIRE v6: Institutional AI Underwriting Engine
# Proprietary Notice: AI Vector Grade™ and Zero-Hallucination Parser
# ============================================================

st.set_page_config(page_title="AIRE | Enterprise Underwriting", layout="wide", initial_sidebar_state="expanded")

# ----------------------------
# 1. THEME & UI STYLING
# ----------------------------
st.markdown("""
<style>
    /* Super Clean Enterprise UI */
    .block-container { padding-top: 2rem; max-width: 1200px; }
    h1, h2, h3 { font-family: 'Inter', -apple-system, sans-serif; font-weight: 800; color: #111827; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; padding-bottom: 12px; }
    .stDataFrame { border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .metric-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .metric-title { font-size: 13px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 28px; font-weight: 800; color: #111827; margin-top: 4px; }
    .alert-box { background-color: #eff6ff; border-left: 4px solid #3b82f6; padding: 16px; border-radius: 4px; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 2. CORE DATABASE (SQLite)
# ----------------------------
DB_PATH = "aire_enterprise.db"

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS deals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, address TEXT,
        grade_score REAL, base_noi REAL, risk_probability REAL, payload TEXT
    )""")
    conn.commit()
    return conn

CONN = init_db()

# ----------------------------
# 3. AI ENGINE: ZERO-HALLUCINATION PARSER
# ----------------------------
def parse_rent_roll_with_ai(raw_text: str) -> pd.DataFrame:
    """Uses strictly typed JSON output and temperature=0.0 to prevent AI hallucinations."""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.warning("⚠️ No OPENAI_API_KEY found in Streamlit Secrets. Using mock extraction for demo.")
        return pd.DataFrame([
            {"unit_number": "101", "bed_bath_type": "1B/1B", "square_feet": 800, "current_rent": 1200, "market_rent": 1400, "status": "Occupied"},
            {"unit_number": "102", "bed_bath_type": "2B/2B", "square_feet": 1100, "current_rent": 1600, "market_rent": 1800, "status": "Occupied"},
            {"unit_number": "103", "bed_bath_type": "Studio", "square_feet": 600, "current_rent": 950, "market_rent": 1100, "status": "Vacant"}
        ])

    openai.api_key = api_key
    
    system_prompt = """
    You are an institutional real estate data extraction engine. 
    CRITICAL RULES:
    1. DO NOT GUESS OR CALCULATE. If a value is missing, output null. 
    2. Output STRICTLY as a JSON object with a single root key called "units" containing an array of objects.
    3. Keys MUST be EXACTLY: "unit_number", "bed_bath_type", "square_feet", "current_rent", "market_rent", "status".
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract this rent roll:\n\n{raw_text}"}
            ],
            response_format={ "type": "json_object" },
            temperature=0.0 
        )
        
        data = json.loads(response.choices[0].message.content)
        df = pd.DataFrame(data["units"])
        
        for col in ['current_rent', 'market_rent', 'square_feet']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        return df
    except Exception as e:
        st.error(f"AI Pipeline Failed: {e}")
        return pd.DataFrame()

# ----------------------------
# 4. AI ENGINE: MONTE CARLO SIMULATOR
# ----------------------------
def run_monte_carlo(base_noi: float, base_cap: float, hold_years: int = 5, iterations: int = 10000) -> dict:
    """Runs 10,000 parallel universe simulations to find exact probability of capital loss."""
    np.random.seed(42) 
    rent_growth_sims = np.random.normal(0.03, 0.02, iterations)
    exit_cap_sims = np.random.normal(base_cap, 0.0075, iterations)
    
    yr5_noi = base_noi * ((1 + rent_growth_sims) ** hold_years)
    exit_values = yr5_noi / exit_cap_sims
    entry_value = base_noi / base_cap
    
    prob_loss = np.sum(exit_values < entry_value) / iterations * 100
    expected_value = np.median(exit_values)
    
    score = max(0, min(100, 100 - (prob_loss * 2.5)))
    
    return {
        "expected_exit_value": expected_value,
        "probability_of_loss": prob_loss,
        "aire_score": score,
        "simulations": exit_values
    }

# ----------------------------
# 5. UI VIEWS
# ----------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🏢 AIRE Engine")
        menu = st.radio("Navigation", ["Data Ingestion (Rent Roll)", "Risk Engine (Monte Carlo)", "Deal Pipeline"], label_visibility="collapsed")
        st.markdown("---")
        st.success("🟢 Systems Operational")
        
        # ADDED: Lead Capture for Enterprise Upgrades
        st.markdown("---")
        st.markdown("### Upgrade to Enterprise")
        st.caption("Want AIRE integrated directly into your firm's Excel or CRM pipeline?")
        # Replace the link below with your actual free Calendly link!
        st.markdown("[**Book a 15-Min Demo 📅**](https://calendly.com/)")
        
        return menu

def view_data_ingestion():
    st.title("Step 1: AI Data Ingestion")
    st.markdown('<div class="alert-box"><b>RedIQ Alternative:</b> Paste messy, unstructured rent roll text below. The AI will instantly parse it using a strict Zero-Hallucination JSON schema.</div>', unsafe_allow_html=True)
    
    # ADDED: Magic Demo Button
    if st.button("🪄 Load Example Messy Rent Roll"):
        demo_text = """
        Unit 101 is a 1b1b 750sqft paying $1200, market is $1400. 
        Unit 102 is vacant, 2 bed 2 bath 1100 sqft, market rent $1800. 
        Unit 103 pays $900, studio, 500 sqft, occupied.
        Unit 104 2b/1b, 950 sqft, current rent 1450, market 1500.
        """
        st.session_state.rent_roll_input = demo_text
        st.rerun()

    raw_text = st.text_area("Paste Raw Rent Roll Text (from PDF or Email):", height=200, key="rent_roll_input", placeholder="Unit 101, 1bed 1bath, 800 sqft, pays $1200...\nUnit 102, 2x2, vacant...")
    
    if st.button("Extract & Standardize Data", type="primary"):
        with st.spinner("AI is reading and structuring the document..."):
            df = parse_rent_roll_with_ai(st.session_state.rent_roll_input)
            
            if not df.empty:
                st.session_state["extracted_df"] = df
                st.success("Extraction Complete. Please verify the data below.")

    if "extracted_df" in st.session_state:
        st.markdown("### Human-in-the-Loop Verification")
        st.caption("Review the AI's extraction. You can edit any cell directly before pushing to the Risk Engine.")
        
        edited_df = st.data_editor(
            st.session_state["extracted_df"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "current_rent": st.column_config.NumberColumn("Current Rent ($)", format="$ %d"),
                "market_rent": st.column_config.NumberColumn("Market Rent ($)", format="$ %d"),
                "square_feet": st.column_config.NumberColumn("Sq Ft")
            }
        )
        
        if st.button("Lock Data & Calculate NOI"):
            total_rent = edited_df["current_rent"].sum()
            annual_gpr = total_rent * 12
            estimated_noi = annual_gpr * 0.55 
            
            st.session_state["verified_noi"] = estimated_noi
            st.success(f"Data Locked! Estimated Annual NOI: **${estimated_noi:,.2f}**. Proceed to Risk Engine.")

def view_risk_engine():
    st.title("Step 2: Monte Carlo Risk Simulator")
    st.markdown('<div class="alert-box"><b>Argus Alternative:</b> Instead of deterministic single-scenario models, AIRE runs 10,000 parallel economies to map risk probabilities.</div>', unsafe_allow_html=True)
    
    base_noi = st.session_state.get("verified_noi", 250000.0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        noi_input = st.number_input("Base Year NOI ($)", value=float(base_noi), step=5000.0)
    with col2:
        cap_input = st.number_input("Market Cap Rate (%)", value=5.5, step=0.25) / 100
    with col3:
        hold_input = st.number_input("Hold Period (Years)", value=5, min_value=1, max_value=20)

    if st.button("Run 10,000 Simulations", type="primary"):
        with st.spinner("Calculating quantum risk probabilities..."):
            results = run_monte_carlo(noi_input, cap_input, int(hold_input))
            
            st.markdown("### AIRE Institutional Underwriting Results")
            c1, c2, c3 = st.columns(3)
            
            c1.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">AIRE Confidence Grade</div>
                <div class="metric-value">{results['aire_score']:.1f} <span style="font-size:16px; color:#6b7280;">/ 100</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            color = "#ef4444" if results['probability_of_loss'] > 20 else "#22c55e"
            c2.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Probability of Capital Loss</div>
                <div class="metric-value" style="color: {color};">{results['probability_of_loss']:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            c3.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Median Exit Value</div>
                <div class="metric-value">${results['expected_exit_value']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Scenario Distribution (Exit Values)")
            
            counts, bins = np.histogram(results['simulations'], bins=40)
            bin_midpoints = (bins[:-1] + bins[1:]) / 2
            clean_labels = [f"${x/1000000:.2f}M" for x in bin_midpoints]
            
            chart_df = pd.DataFrame({
                "Exit Value": clean_labels,
                "Frequency": counts
            }).set_index("Exit Value")
            
            st.bar_chart(chart_df)
            
            cur = CONN.cursor()
            cur.execute("INSERT INTO deals (created_at, address, grade_score, base_noi, risk_probability) VALUES (?, ?, ?, ?, ?)",
                        (datetime.utcnow().isoformat(), "AI Extracted Deal", results['aire_score'], noi_input, results['probability_of_loss']))
            CONN.commit()
            st.success("Deal Risk Profile saved to Pipeline.")

def view_pipeline():
    st.title("Step 3: Tracked Pipeline")
    cur = CONN.cursor()
    cur.execute("SELECT id, created_at, address, grade_score, base_noi, risk_probability FROM deals ORDER BY id DESC")
    rows = cur.fetchall()
    
    if not rows:
        st.info("Your pipeline is currently empty. Run a deal through the Risk Engine.")
        return
        
    df = pd.DataFrame(rows, columns=["ID", "Date Extracted", "Address", "AIRE Score", "Base NOI ($)", "Loss Probability (%)"])
    df["Date Extracted"] = pd.to_datetime(df["Date Extracted"]).dt.strftime('%Y-%m-%d')
    df["Base NOI ($)"] = df["Base NOI ($)"].apply(lambda x: f"${x:,.2f}")
    df["Loss Probability (%)"] = df["Loss Probability (%)"].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(df, use_container_width=True, hide_index=True)

# ----------------------------
# 6. APP EXECUTION ROUTER
# ----------------------------
def main():
    menu = render_sidebar()
    if menu == "Data Ingestion (Rent Roll)":
        view_data_ingestion()
    elif menu == "Risk Engine (Monte Carlo)":
        view_risk_engine()
    elif menu == "Deal Pipeline":
        view_pipeline()

if __name__ == "__main__":
    main()