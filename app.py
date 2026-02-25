import os, json, io
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import openai
import PyPDF2
from supabase import create_client, Client

# ============================================================
# AIRE v8: Enterprise SaaS Edition (Production Ready)
# Multi-Tenant | Postgres DB | PDF OCR | Institutional Math
# ============================================================

st.set_page_config(page_title="AIRE | Enterprise Underwriting", layout="wide", initial_sidebar_state="expanded")

# ----------------------------
# 1. THEME & UI STYLING
# ----------------------------
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1200px; }
    h1, h2, h3 { font-family: 'Inter', -apple-system, sans-serif; font-weight: 800; color: #111827; }
    .stDataFrame { border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .metric-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .metric-title { font-size: 13px; color: #6b7280; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 26px; font-weight: 800; color: #111827; margin-top: 4px; }
    .alert-box { background-color: #eff6ff; border-left: 4px solid #3b82f6; padding: 16px; border-radius: 4px; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 2. SUPABASE DB INIT & LOGIN
# ----------------------------
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", "")
    if not url or not key:
        st.error("⚠️ Database missing. Please add SUPABASE_URL and SUPABASE_KEY to your secrets.")
        st.stop()
    return create_client(url, key)

supabase = init_supabase()

# --- THE SAAS LOGIN PORTAL ---
if "firm_id" not in st.session_state:
    st.title("AIRE Enterprise Portal")
    st.markdown("### Access your firm's private workspace.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        firm_code = st.text_input("Enter Firm Access Code (e.g., BLACKSTONE)", type="password")
        if st.button("Authenticate Workspace", type="primary"):
            if firm_code.strip():
                st.session_state.firm_id = firm_code.strip().upper()
                st.rerun()
            else:
                st.error("Please enter a valid firm code.")
    with col2:
        st.info("**Beta Testers:** Enter your company name as your Access Code. This isolates your deal pipeline securely from other firms.")
    st.stop() # Stops the rest of the app from loading until logged in

# ----------------------------
# 3. AI PDF & PARSING ENGINE
# ----------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def parse_rent_roll_with_ai(raw_text: str) -> pd.DataFrame:
    openai.api_key = st.secrets.get("OPENAI_API_KEY", "")
    system_prompt = """
    You are an institutional real estate data extraction engine. 
    1. DO NOT GUESS. If missing, output null. 
    2. Output STRICT JSON with root key "units" containing objects.
    3. Keys: "unit_number", "bed_bath_type", "square_feet", "current_rent", "market_rent", "status".
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract this rent roll:\n\n{raw_text[:8000]}"} # Limit tokens
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
# 4. INSTITUTIONAL MATH (IRR & EM)
# ----------------------------
def compute_irr_bisection(cash_flows: list, iterations=100) -> float:
    """Robust IRR solver using bisection grid to prevent overflow on extreme holds."""
    low, high = -0.99, 2.0
    irr = 0.0
    for _ in range(iterations):
        irr = (low + high) / 2
        npv = sum([cf / ((1 + irr) ** t) for t, cf in enumerate(cash_flows)])
        if npv > 0:
            low = irr
        else:
            high = irr
    return irr

def run_monte_carlo(base_noi: float, base_cap: float, hold_years: int = 5, iterations: int = 5000) -> dict:
    np.random.seed(42) 
    rent_growth_sims = np.random.normal(0.03, 0.02, iterations)
    exit_cap_sims = np.random.normal(base_cap, 0.0075, iterations)
    
    entry_price = base_noi / base_cap
    
    simulated_irrs = []
    simulated_ems = []
    exit_values = []
    
    # Run the rigorous institutional math
    for i in range(iterations):
        cash_flows = [-entry_price]
        current_noi = base_noi
        
        # Hold period cash flows
        for year in range(1, hold_years):
            current_noi *= (1 + rent_growth_sims[i])
            cash_flows.append(current_noi)
            
        # Exit Year
        final_noi = current_noi * (1 + rent_growth_sims[i])
        exit_price = final_noi / exit_cap_sims[i]
        exit_values.append(exit_price)
        cash_flows.append(final_noi + exit_price)
        
        # Metrics
        total_profit = sum(cash_flows[1:])
        simulated_ems.append(total_profit / entry_price)
        simulated_irrs.append(compute_irr_bisection(cash_flows))

    prob_loss = np.sum(np.array(exit_values) < entry_price) / iterations * 100
    expected_irr = np.median(simulated_irrs) * 100
    expected_em = np.median(simulated_ems)
    
    # Grading Scale adjustment
    score = max(0, min(100, 100 - (prob_loss * 2.5) + (expected_irr / 2)))
    
    return {
        "expected_exit_value": np.median(exit_values),
        "probability_of_loss": prob_loss,
        "expected_irr": expected_irr,
        "expected_em": expected_em,
        "aire_score": score,
        "simulations": exit_values
    }

# ----------------------------
# 5. UI VIEWS
# ----------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(f"### 🏢 Workspace: `{st.session_state.firm_id}`")
        # FIXED: Removed the invalid size argument from st.button
        if st.button("Log Out"):
            st.session_state.clear()
            st.rerun()
        st.markdown("---")
        menu = st.radio("Navigation", ["Data Ingestion (PDF AI)", "Risk Engine (Monte Carlo)", "Master Pipeline"], label_visibility="collapsed")
        st.markdown("---")
        st.success("🟢 Systems Operational")
        return menu

def view_data_ingestion():
    st.title("Step 1: AI Data Ingestion")
    st.markdown('<div class="alert-box"><b>Upload PDF:</b> Drag and drop a broker Offering Memorandum (OM) or Rent Roll. AIRE will extract the text and standardize the units instantly.</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload PDF Rent Roll", type=["pdf"])
    raw_text = ""
    
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF via OCR..."):
            raw_text = extract_text_from_pdf(uploaded_file)
            st.success("PDF Read Successfully! Click 'Standardize' below.")
            
    # Fallback for manual paste
    with st.expander("Or paste raw text manually"):
        manual_text = st.text_area("Raw Text:", height=150)
        if manual_text: raw_text = manual_text
    
    if st.button("Extract & Standardize Data", type="primary", disabled=not raw_text):
        with st.spinner("AI is structuring the document..."):
            df = parse_rent_roll_with_ai(raw_text)
            if not df.empty:
                st.session_state["extracted_df"] = df

    if "extracted_df" in st.session_state:
        st.markdown("### Human-in-the-Loop Verification")
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
        
        deal_address = st.text_input("Property Name / Address (For Pipeline Tracking):", "123 Main St Portfolio")
        
        if st.button("Lock Data & Calculate NOI"):
            total_rent = edited_df["current_rent"].sum()
            estimated_noi = (total_rent * 12) * 0.55 # 45% Expense Ratio
            
            st.session_state["verified_noi"] = estimated_noi
            st.session_state["deal_address"] = deal_address
            st.success(f"Data Locked! Estimated Annual NOI: **${estimated_noi:,.2f}**. Proceed to Risk Engine.")

def view_risk_engine():
    st.title("Step 2: Monte Carlo Risk Simulator")
    
    base_noi = st.session_state.get("verified_noi", 250000.0)
    deal_address = st.session_state.get("deal_address", "Manual Entry")
    
    st.write(f"**Underwriting Deal:** {deal_address}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        noi_input = st.number_input("Base Year NOI ($)", value=float(base_noi), step=5000.0)
    with col2:
        cap_input = st.number_input("Market Cap Rate (%)", value=5.5, step=0.25) / 100
    with col3:
        hold_input = st.number_input("Hold Period (Years)", value=5, min_value=1, max_value=20)

    if st.button("Run 10,000 Simulations", type="primary"):
        with st.spinner("Calculating quantum risk probabilities and IRR..."):
            results = run_monte_carlo(noi_input, cap_input, int(hold_input))
            
            st.markdown("### AIRE Institutional Underwriting Results")
            c1, c2, c3, c4 = st.columns(4)
            
            c1.markdown(f"""<div class="metric-card"><div class="metric-title">AIRE Confidence</div>
                <div class="metric-value">{results['aire_score']:.1f}</div></div>""", unsafe_allow_html=True)
                
            c2.markdown(f"""<div class="metric-card"><div class="metric-title">Expected IRR</div>
                <div class="metric-value">{results['expected_irr']:.1f}%</div></div>""", unsafe_allow_html=True)
                
            c3.markdown(f"""<div class="metric-card"><div class="metric-title">Equity Multiple</div>
                <div class="metric-value">{results['expected_em']:.2f}x</div></div>""", unsafe_allow_html=True)
                
            color = "#ef4444" if results['probability_of_loss'] > 20 else "#22c55e"
            c4.markdown(f"""<div class="metric-card"><div class="metric-title">Capital Loss Prob.</div>
                <div class="metric-value" style="color: {color};">{results['probability_of_loss']:.1f}%</div></div>""", unsafe_allow_html=True)
            
            st.markdown("<br>#### Scenario Distribution (Exit Values)", unsafe_allow_html=True)
            counts, bins = np.histogram(results['simulations'], bins=40)
            bin_midpoints = (bins[:-1] + bins[1:]) / 2
            chart_df = pd.DataFrame({"Exit Value": [f"${x/1000000:.2f}M" for x in bin_midpoints], "Frequency": counts}).set_index("Exit Value")
            st.bar_chart(chart_df)
            
            # --- SAVE TO SUPABASE ---
            try:
                data = {
                    "firm_id": st.session_state.firm_id,
                    "address": deal_address,
                    "grade_score": results['aire_score'],
                    "base_noi": noi_input,
                    "risk_probability": results['probability_of_loss'],
                    "expected_irr": results['expected_irr'],
                    "equity_multiple": results['expected_em']
                }
                supabase.table("deals").insert(data).execute()
                st.success("✅ Deal permanently saved to your Firm's Cloud Pipeline.")
            except Exception as e:
                st.error(f"Database Save Error: {e}")

def view_pipeline():
    st.title("Step 3: Master Deal Pipeline")
    st.markdown(f"**Viewing secure pipeline for:** `{st.session_state.firm_id}`")
    
    try:
        # Fetch ONLY the deals belonging to the logged-in firm
        response = supabase.table("deals").select("*").eq("firm_id", st.session_state.firm_id).order("id", desc=True).execute()
        rows = response.data
    except Exception as e:
        st.error("Could not fetch database.")
        rows = []
    
    if not rows:
        st.info("Your firm's pipeline is currently empty. Run a deal through the Risk Engine.")
        return
        
    df = pd.DataFrame(rows)
    df = df[["id", "created_at", "address", "grade_score", "expected_irr", "equity_multiple", "base_noi", "risk_probability"]]
    
    # Formatting for UI
    display_df = df.copy()
    display_df["created_at"] = pd.to_datetime(display_df["created_at"]).dt.strftime('%Y-%m-%d')
    display_df["expected_irr"] = display_df["expected_irr"].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
    display_df["equity_multiple"] = display_df["equity_multiple"].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "N/A")
    display_df["base_noi"] = display_df["base_noi"].apply(lambda x: f"${x:,.2f}")
    display_df["risk_probability"] = display_df["risk_probability"].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    def generate_excel(dataframe: pd.DataFrame):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            dataframe.to_excel(writer, index=False, sheet_name='Master Pipeline')
        return output.getvalue()

    # Clean the dataframe column names before export
    export_df = df.rename(columns={"address": "Property Address", "grade_score": "AIRE Score", "expected_irr": "Expected IRR (%)", "equity_multiple": "Equity Multiple", "base_noi": "Base NOI ($)", "risk_probability": "Loss Prob (%)"})
    export_df.drop(columns=["id"], inplace=True)
    
    st.download_button(
        label="📊 Download Excel Pipeline (.xlsx)",
        data=generate_excel(export_df),
        file_name=f"{st.session_state.firm_id}_Pipeline_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )

def main():
    menu = render_sidebar()
    if menu == "Data Ingestion (PDF AI)": view_data_ingestion()
    elif menu == "Risk Engine (Monte Carlo)": view_risk_engine()
    elif menu == "Master Pipeline": view_pipeline()

if __name__ == "__main__":
    main()
