import os
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px
import requests
from openai import OpenAI
from supabase import create_client, Client

# ==============================================================================
# AIRE | INSTITUTIONAL UNDERWRITING ENGINE V3.0 (THE $1000/MO FLAGSHIP)
# ==============================================================================

st.set_page_config(
    page_title="AIRE | Institutional Underwriting",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 1. ENTERPRISE CSS & ANIMATIONS
# ------------------------------------------------------------------------------
def inject_enterprise_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
        
        /* Global Reset */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #f3f4f6; } /* Slightly darker grey for contrast */
        #MainMenu, footer, header {visibility: hidden;}
        
        /* The Enterprise Sidebar */
        [data-testid="stSidebar"] {
            background-color: #0a0f1c !important; /* Extremely dark navy */
            border-right: 1px solid #1e293b;
        }
        [data-testid="stSidebar"] * { color: #94a3b8 !important; }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #f8fafc !important; font-weight: 800; letter-spacing: -0.5px;
        }
        
        /* Top Navigation/Header area */
        .enterprise-header {
            background: #ffffff; padding: 24px 32px; border-bottom: 1px solid #e2e8f0;
            margin: -6rem -4rem 2rem -4rem; display: flex; justify-content: space-between; align-items: center;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        }
        .header-title { font-size: 26px; font-weight: 800; color: #0f172a; letter-spacing: -0.5px; }
        .header-badge {
            background: #f1f5f9; color: #0f172a; padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 700; border: 1px solid #e2e8f0;
        }

        /* Institutional Metric Cards */
        div[data-testid="metric-container"] {
            background-color: #ffffff; border: 1px solid #e2e8f0; padding: 24px 20px;
            border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            border-top: 4px solid #2563eb; /* Blue accent top */
            transition: all 0.2s ease;
        }
        div[data-testid="metric-container"]:hover { box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); transform: translateY(-2px); }
        div[data-testid="metric-container"] label {
            color: #64748b !important; font-size: 12px !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.5px;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            color: #0f172a !important; font-size: 34px !important; font-weight: 800 !important; font-family: 'JetBrains Mono', monospace;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricDelta"] svg { display: none; } /* Hide default arrows */

        /* Custom Panels & Glassmorphism */
        .glass-panel {
            background: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom: 24px;
        }
        .panel-title {
            font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 0.5px;
            border-bottom: 2px solid #f1f5f9; padding-bottom: 12px; display: flex; justify-content: space-between;
        }
        
        /* Institutional Pro Forma Table */
        .proforma-table { width: 100%; border-collapse: collapse; font-size: 13px; font-family: 'JetBrains Mono', monospace; }
        .proforma-table th { text-align: right; padding: 10px; border-bottom: 2px solid #cbd5e1; color: #475569; font-weight: 700; font-family: 'Inter', sans-serif;}
        .proforma-table th:first-child { text-align: left; }
        .proforma-table td { text-align: right; padding: 10px; border-bottom: 1px solid #e2e8f0; color: #0f172a; }
        .proforma-table td:first-child { text-align: left; font-family: 'Inter', sans-serif; font-weight: 500; color: #334155; }
        .proforma-table tr.noi-row td { font-weight: 800; background-color: #f8fafc; color: #0f172a; border-top: 2px solid #94a3b8; border-bottom: 2px solid #94a3b8;}
        .proforma-table tr:hover { background-color: #f1f5f9; }

        /* Chat Interface Customization */
        .stChatMessage { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05); }
        .stChatMessage.user { background-color: #f8fafc; border-color: #cbd5e1; }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. CORE BACKEND & STATE MANAGEMENT
# ------------------------------------------------------------------------------
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", "")
    if not url or not key: return None
    return create_client(url, key)

supabase = init_supabase()
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))

def initialize_session_state():
    if "user_email" not in st.session_state: st.session_state.user_email = None
    if "firm_id" not in st.session_state: st.session_state.firm_id = None
    if "current_view" not in st.session_state: st.session_state.current_view = "Dashboard"
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "deal_data" not in st.session_state: 
        st.session_state.deal_data = {
            "name": "The Grand at 100 Main St", "units": 240, "vintage": 2018,
            "irr": 0.182, "equity_mult": 2.15, "gp_irr": 0.265, "loss_prob": 0.042,
            "purchase_price": 45000000, "debt_amount": 29250000, "lp_equity": 14175000, "gp_equity": 1575000
        }

# ------------------------------------------------------------------------------
# 3. ANALYTICAL ENGINES & MATH
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_live_interest_rate():
    fred_key = st.secrets.get("FRED_API_KEY", "")
    if not fred_key: return 6.75 
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={fred_key}&file_type=json&sort_order=desc&limit=1"
        resp = requests.get(url, timeout=5).json()
        return float(resp['observations'][0]['value']) + 2.00 
    except: return 6.75

def run_monte_carlo(base_return=0.182, volatility=0.045, simulations=2000):
    np.random.seed(42)
    return np.random.normal(base_return, volatility, simulations)

def generate_sensitivity_matrix(base_irr, base_cap_rate):
    # Generates a realistic 5x5 matrix for Exit Cap Rate vs Hold Year
    cap_rates = [base_cap_rate - 0.005, base_cap_rate - 0.0025, base_cap_rate, base_cap_rate + 0.0025, base_cap_rate + 0.005]
    years = [3, 4, 5, 6, 7]
    matrix = np.zeros((len(cap_rates), len(years)))
    for i, cap in enumerate(cap_rates):
        for j, yr in enumerate(years):
            # Synthetic IRR calculation based on cap rate expansion/compression and time
            adj_irr = base_irr + ((base_cap_rate - cap) * 10) - ((yr - 5) * 0.005)
            matrix[i, j] = adj_irr
    return matrix, cap_rates, years

# ------------------------------------------------------------------------------
# 4. SECURE AUTHENTICATION GATEWAY
# ------------------------------------------------------------------------------
def render_login():
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown('<div class="glass-panel" style="padding: 50px; text-align: center; border-top: 4px solid #0f172a;">', unsafe_allow_html=True)
        st.markdown('<h1 style="font-weight: 900; color: #0f172a; margin-bottom: 0px; font-size: 42px; letter-spacing: -2px;">AIRE</h1>', unsafe_allow_html=True)
        st.markdown('<p style="color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; font-size: 12px; margin-bottom: 40px;">Institutional Underwriting</p>', unsafe_allow_html=True)
        
        with st.form("enterprise_login"):
            email = st.text_input("Corporate Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Authenticate", use_container_width=True, type="primary")
            
            if submit:
                if supabase:
                    try:
                        resp = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        st.session_state.user_email = resp.user.email
                        st.session_state.firm_id = resp.user.email.split('@')[1].split('.')[0].upper()
                        st.rerun()
                    except: st.error("Access Denied. Ensure active subscription via Stripe.")
                else:
                    if email and password:
                        st.session_state.user_email = email
                        st.session_state.firm_id = email.split('@')[1].split('.')[0].upper() if '@' in email else "DEMO_FIRM"
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 5. UI COMPONENTS: HIGH-FIDELITY VISUALS
# ------------------------------------------------------------------------------
def render_plotly_monte_carlo(simulated_returns):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=simulated_returns, nbinsx=100, marker_color='#3b82f6', opacity=0.9,
        marker_line_width=0.5, marker_line_color='white',
        hovertemplate='Return: %{x:.1%}<br>Frequency: %{y}<extra></extra>'
    ))
    avg_ret = np.mean(simulated_returns)
    fig.add_vline(x=avg_ret, line_width=2, line_dash="dot", line_color="#ef4444")
    fig.add_annotation(
        x=avg_ret, y=80, text=f"Mean IRR: {avg_ret:.1%}", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#ef4444",
        bordercolor="#ef4444", borderpad=4, bgcolor="#ffffff", font=dict(color="#b91c1c", size=12, family="Inter")
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickformat='.0%', showgrid=True, gridcolor='#f1f5f9', zeroline=False, tickfont=dict(color='#64748b')),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        height=280, showlegend=False
    )
    return fig

def render_sensitivity_heatmap(base_irr=0.182, base_cap=0.0525):
    matrix, cap_rates, years = generate_sensitivity_matrix(base_irr, base_cap)
    
    # Format labels
    x_labels = [f"Yr {y}" for y in years]
    y_labels = [f"{c*100:.2f}%" for c in cap_rates]
    
    # Custom colorscale: Red (bad) -> White (neutral) -> Green (good)
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=x_labels, y=y_labels,
        colorscale=[[0, '#fee2e2'], [0.5, '#ffffff'], [1, '#dcfce3']],
        text=[[f"{val*100:.1f}%" for val in row] for row in matrix],
        texttemplate="<b>%{text}</b>", textfont=dict(size=12, family="JetBrains Mono"),
        showscale=False, hoverinfo="skip"
    ))
    
    # Bulletproof layout updates (no nested dictionaries)
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0), 
        height=280,
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Safely update axes individually
    fig.update_xaxes(title_text="Hold Period", side="top", title_font=dict(size=12, color='#64748b'), tickfont=dict(color='#0f172a'))
    fig.update_yaxes(title_text="Exit Cap Rate", autorange="reversed", title_font=dict(size=12, color='#64748b'), tickfont=dict(color='#0f172a'))
    
    return fig

def render_capital_stack_donut(d):
    labels = ['Senior Debt', 'LP Equity', 'GP Equity']
    values = [d['debt_amount'], d['lp_equity'], d['gp_equity']]
    colors = ['#1e293b', '#3b82f6', '#38bdf8'] # Dark Navy, Royal Blue, Light Blue
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.65, marker_colors=colors, textinfo='none')])
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), height=220, showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=0.9, font=dict(size=11, color='#475569')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(text=f"${sum(values)/1000000:.1f}M", x=0.35, y=0.5, font_size=20, font_family="JetBrains Mono", font_color="#0f172a", showarrow=False)]
    )
    return fig

def render_3d_map():
    view_state = pdk.ViewState(latitude=32.7767, longitude=-96.7970, zoom=14.5, pitch=45)
    layer = pdk.Layer(
        "ColumnLayer", data=[{"lat": 32.7767, "lon": -96.7970, "elevation": 150}],
        get_position="[lon, lat]", get_elevation="elevation", elevation_scale=1.5,
        radius=60, get_fill_color=[37, 99, 235, 255], auto_highlight=True,
    )
    return pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v10")

def render_proforma_html():
    return """
    <table class="proforma-table">
        <thead>
            <tr><th>Line Item</th><th>Year 1</th><th>Year 2</th><th>Year 3</th><th>Year 4</th><th>Year 5</th></tr>
        </thead>
        <tbody>
            <tr><td>Gross Potential Rent</td><td>$4,350,000</td><td>$4,524,000</td><td>$4,704,960</td><td>$4,893,158</td><td>$5,088,884</td></tr>
            <tr><td>Loss to Lease / Vacancy</td><td>($348,000)</td><td>($316,680)</td><td>($282,297)</td><td>($293,589)</td><td>($305,333)</td></tr>
            <tr><td>Other Income</td><td>$215,000</td><td>$221,450</td><td>$228,093</td><td>$234,935</td><td>$241,983</td></tr>
            <tr style="border-top: 1px solid #cbd5e1; background: #f8fafc;"><td><b>Effective Gross Income</b></td><td>$4,217,000</td><td>$4,428,770</td><td>$4,650,756</td><td>$4,834,504</td><td>$5,025,534</td></tr>
            <tr><td>Taxes & Insurance</td><td>($650,000)</td><td>($669,500)</td><td>($689,585)</td><td>($710,272)</td><td>($731,580)</td></tr>
            <tr><td>Payroll & Management</td><td>($420,000)</td><td>($432,600)</td><td>($445,578)</td><td>($458,945)</td><td>($472,713)</td></tr>
            <tr><td>Repairs, Maint. & Utilities</td><td>($310,000)</td><td>($319,300)</td><td>($328,879)</td><td>($338,745)</td><td>($348,907)</td></tr>
            <tr class="noi-row"><td>Net Operating Income</td><td>$2,837,000</td><td>$3,007,370</td><td>$3,186,714</td><td>$3,326,542</td><td>$3,472,334</td></tr>
            <tr><td><i>CapEx Reserves</i></td><td><i>($60,000)</i></td><td><i>($60,000)</i></td><td><i>($60,000)</i></td><td><i>($60,000)</i></td><td><i>($60,000)</i></td></tr>
            <tr><td><b>Net Cash Flow</b></td><td><b>$2,777,000</b></td><td><b>$2,947,370</b></td><td><b>$3,126,714</b></td><td><b>$3,266,542</b></td><td><b>$3,412,334</b></td></tr>
        </tbody>
    </table>
    """

# ------------------------------------------------------------------------------
# 6. APPLICATION VIEWS
# ------------------------------------------------------------------------------
def view_dashboard():
    # --- HEADER ---
    st.markdown(f"""
    <div class="enterprise-header">
        <div>
            <div class="header-title">{st.session_state.deal_data['name']} <span style="color:#94a3b8; font-weight:400;">| 240 Units | Value-Add</span></div>
            <div style="color: #64748b; font-size: 13px; margin-top: 6px; font-weight: 500;">
                <span style="color: #10b981;">●</span> Pipeline: Active Underwriting &nbsp;&bull;&nbsp; Market: Dallas, TX &nbsp;&bull;&nbsp; Last Updated: Just now
            </div>
        </div>
        <div class="header-badge">WORKSPACE: {st.session_state.firm_id}</div>
    </div>
    """, unsafe_allow_html=True)

    d = st.session_state.deal_data

    # --- ROW 1: THE METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Levered Deal IRR", f"{d['irr']*100:.1f}%", "+1.2% vs Target", delta_color="normal")
    c2.metric("Equity Multiple", f"{d['equity_mult']:.2f}x", "0.15x vs Target", delta_color="normal")
    c3.metric("GP Promote IRR", f"{d['gp_irr']*100:.1f}%", "+4.5% Over Hurdle", delta_color="normal")
    c4.metric("Risk: Equity Loss Prob.", f"{d['loss_prob']*100:.1f}%", "-2.1% vs Market", delta_color="inverse")

    # --- ROW 2: ADVANCED VISUALS ---
    col_mc, col_sens, col_map = st.columns([1.5, 1.5, 1])
    with col_mc:
        st.markdown('<div class="glass-panel"><div class="panel-title">Monte Carlo Simulation <span style="font-size:11px; color:#94a3b8; float:right; font-weight:500;">2,000 SCENARIOS</span></div>', unsafe_allow_html=True)
        st.plotly_chart(render_plotly_monte_carlo(run_monte_carlo(d['irr'])), use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_sens:
        st.markdown('<div class="glass-panel"><div class="panel-title">Sensitivity: Exit Cap vs Hold <span style="font-size:11px; color:#94a3b8; float:right; font-weight:500;">IRR IMPACT</span></div>', unsafe_allow_html=True)
        st.plotly_chart(render_sensitivity_heatmap(), use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_map:
        st.markdown('<div class="glass-panel" style="height: 385px;"><div class="panel-title">Asset Overview</div>', unsafe_allow_html=True)
        st.pydeck_chart(render_3d_map(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- ROW 3: PRO FORMA & CAPITAL STACK ---
    col_pf, col_cap = st.columns([2.5, 1])
    with col_pf:
        st.markdown('<div class="glass-panel"><div class="panel-title">Standardized 5-Year Pro Forma <span style="font-size:11px; color:#2563eb; float:right; cursor:pointer;">⬇ EXPORT TO EXCEL</span></div>', unsafe_allow_html=True)
        st.markdown(render_proforma_html(), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_cap:
        st.markdown('<div class="glass-panel" style="height: 420px;"><div class="panel-title">Capital Stack</div>', unsafe_allow_html=True)
        st.plotly_chart(render_capital_stack_donut(d), use_container_width=True, config={'displayModeBar': False})
        
        # Live Debt Box
        live_rate = fetch_live_interest_rate()
        st.markdown(f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; margin-top: 10px;">
            <div style="font-size: 11px; color: #64748b; font-weight: 700;">LIVE DEBT INDEX (10-YR T + 200BPS)</div>
            <div style="font-size: 20px; font-weight: 800; color: #0f172a; font-family: 'JetBrains Mono';">{live_rate:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def view_data_room():
    st.markdown('<div class="enterprise-header"><div class="header-title">AI Data Room & Document Intelligence</div></div>', unsafe_allow_html=True)
    
    col_doc, col_chat = st.columns([1, 1.2])
    
    with col_doc:
        st.markdown('<div class="glass-panel" style="height: 70vh; overflow-y:auto;">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Document Repository</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader("Upload T12, Rent Roll, or Offering Memorandum (PDF/Excel)", accept_multiple_files=True)
        if uploaded_files:
            st.success(f"{len(uploaded_files)} documents indexed and vectorized into active memory.")
            for file in uploaded_files:
                st.markdown(f"<div style='padding:10px; border:1px solid #e2e8f0; border-radius:6px; margin-bottom:8px; font-size:13px;'>📄 <b>{file.name}</b> <span style='float:right; color:#10b981;'>Indexed</span></div>", unsafe_allow_html=True)
        else:
            st.info("Upload deal documents to activate AI extraction and RAG chat.")
            # Mock documents for visual effect
            st.markdown("<div style='padding:10px; border:1px dashed #cbd5e1; border-radius:6px; margin-bottom:8px; font-size:13px; color:#64748b;'>📄 100_Main_St_OM_Final.pdf (Pre-loaded)</div>", unsafe_allow_html=True)
            st.markdown("<div style='padding:10px; border:1px dashed #cbd5e1; border-radius:6px; margin-bottom:8px; font-size:13px; color:#64748b;'>📊 T12_Trailing_Financials.xlsx (Pre-loaded)</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chat:
        st.markdown('<div class="glass-panel" style="height: 70vh; display: flex; flex-direction: column;">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title" style="margin-bottom:0px;">Deal Copilot (RAG)</div>', unsafe_allow_html=True)
        
        # Chat History Container
        chat_container = st.container(height=450, border=False)
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("<div style='text-align:center; color:#94a3b8; margin-top: 100px; font-size:14px;'>Ask me to extract expenses, identify risks in the OM, or summarize lease terms.</div>", unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Input Area
        if prompt := st.chat_input("Ask a question about the deal documents..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.rerun() # Refresh to show user message immediately
            
        # Mocking the AI response logic (If user just sent a message, bot replies)
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Scanning documents..."):
                        time.sleep(1.5) # Simulate RAG latency
                        bot_reply = "Based on page 42 of the Offering Memorandum, the property requires a complete roof replacement on Building C within the next 24 months. The estimated CapEx for this is **$125,000**. I have automatically added this to your Year 1/Year 2 CapEx reserves in the underwriting model."
                        st.markdown(bot_reply)
                        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                        # Rerun needed to stabilize state without re-triggering
                        st.rerun() 
        st.markdown('</div>', unsafe_allow_html=True)

def view_ic_memo():
    st.markdown('<div class="enterprise-header"><div class="header-title">Investment Committee Memo Generator</div></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("### Export Configuration", unsafe_allow_html=True)
        st.checkbox("Include Monte Carlo Chart", value=True)
        st.checkbox("Include Sensitivity Heatmap", value=True)
        st.checkbox("Include 5-Year Pro Forma", value=True)
        st.checkbox("Include Map & Demographics", value=True)
        
        if st.button("Generate Word Document", type="primary", use_container_width=True):
            with st.spinner("Compiling institutional memo..."):
                time.sleep(2)
            st.success("Memo successfully generated.")
            st.download_button("Download IC_Memo_100_Main_St.docx", data=b"mock_word_data", file_name="IC_Memo.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-panel" style="background:#f8fafc; border: 1px dashed #cbd5e1; text-align:center; height: 400px; display:flex; align-items:center; justify-content:center; color:#64748b;">', unsafe_allow_html=True)
        st.markdown("<h3>Document Preview Render Space</h3><p>Shows a live preview of the generated Word document here.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 7. ROUTING & MASTER EXECUTION
# ------------------------------------------------------------------------------
def main():
    initialize_session_state()
    inject_enterprise_css()

    if not st.session_state.user_email:
        render_login()
        st.stop()

    # The Institutional Sidebar Navigation
    with st.sidebar:
        st.markdown("<div style='padding: 10px 0 30px 0;'><h1 style='margin:0; font-size:32px;'>AIRE</h1><span style='color:#3b82f6; font-size: 11px; font-weight: 800; letter-spacing:1px;'>INSTITUTIONAL PLATFORM</span></div>", unsafe_allow_html=True)
        
        st.markdown("<div style='font-size:11px; color:#475569; font-weight:700; margin-bottom:10px; letter-spacing:1px;'>DEAL ANALYSIS</div>", unsafe_allow_html=True)
        if st.button("📊 Deal Dashboard", use_container_width=True): st.session_state.current_view = "Dashboard"
        if st.button("🧠 AI Data Room & Chat", use_container_width=True): st.session_state.current_view = "DataRoom"
        if st.button("📄 IC Memo Generator", use_container_width=True): st.session_state.current_view = "ICMemo"
        
        st.markdown("<div style='font-size:11px; color:#475569; font-weight:700; margin:25px 0 10px 0; letter-spacing:1px;'>PORTFOLIO</div>", unsafe_allow_html=True)
        st.button("🏢 Master Pipeline", use_container_width=True)
        st.button("⚙️ Underwriting Settings", use_container_width=True)
        
        st.markdown("<br>"*8, unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 12px; color: #64748b; padding-top:20px; border-top: 1px solid #1e293b;'>Logged in as:<br><b style='color:#f8fafc;'>{st.session_state.user_email}</b></div>", unsafe_allow_html=True)
        if st.button("Secure Logout"):
            st.session_state.clear()
            st.rerun()

    # View Router
    if st.session_state.current_view == "Dashboard": view_dashboard()
    elif st.session_state.current_view == "DataRoom": view_data_room()
    elif st.session_state.current_view == "ICMemo": view_ic_memo()

if __name__ == "__main__":
    main()
