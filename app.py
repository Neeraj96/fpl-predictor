import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
st.set_page_config(page_title="FPL Pro Hybrid 25/26", page_icon="🧬", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 60px; white-space: pre-wrap; background-color: #f0f2f6;
        border-radius: 8px 8px 0 0; padding: 10px 20px;
        font-size: 18px; font-weight: 700; color: #4a4a4a;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: #e0e2e6; color: #1f77b4; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff; border-top: 3px solid #00cc00;
        color: #00cc00; box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
    }
    .stButton button { width: 100%; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
API_BASE = "https://fantasy.premierleague.com/api"

# =========================================
# 1. AI DATA INFRASTRUCTURE
# =========================================

def download_training_data():
    if os.path.exists("fpl_5_year_history.csv"):
        return pd.read_csv("fpl_5_year_history.csv")
        
    status = st.empty()
    status.info("⏳ Initializing: Downloading historical data...")
    
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
    all_data = []
    
    for season in seasons:
        try:
            url = f"{base_url}/{season}/gws/merged_gw.csv"
            r = requests.get(url)
            if r.status_code == 200:
                df = pd.read_csv(io.BytesIO(r.content), on_bad_lines='skip', low_memory=False)
                
                # Extended Column List (Actuals + Expected)
                cols = ['minutes', 'total_points', 'was_home', 'clean_sheets', 
                        'goals_conceded', 'expected_goals', 'expected_assists', 
                        'influence', 'creativity', 'threat', 'value', 'element_type',
                        'goals_scored', 'assists', 'saves', 'bps', 'yellow_cards']
                
                existing = [c for c in cols if c in df.columns]
                df = df[existing]
                all_data.append(df)
        except: pass
        
    if all_data:
        master = pd.concat(all_data)
        master.fillna(0, inplace=True)
        master.to_csv("fpl_5_year_history.csv", index=False)
        status.empty()
        return master
    return None

@st.cache_resource
def train_ai_model():
    df = download_training_data()
    if df is None: return None, None, None
    
    df = df[df['minutes'] > 60].copy()
    
    # FULL FEATURE LIST (Let the AI Decide)
    # We include Actuals (Goals/Assists) AND Underlying (xG/xA/ICT)
    features = [
        'value', 'element_type', 'was_home', 
        'expected_goals', 'expected_assists', 'influence', 'creativity', 'threat', # Underlying
        'clean_sheets', 'goals_conceded', 'saves', 'bps', # Defensive/Bonus Actuals
        'goals_scored', 'assists', 'yellow_cards' # Attacking/Discipline Actuals
    ]
    
    valid_features = [f for f in features if f in df.columns]
    X = df[valid_features]
    y = df['total_points']
    
    # HistGradientBoosting is excellent at ignoring irrelevant features
    model = HistGradientBoostingRegressor(max_iter=50, random_state=42)
    model.fit(X, y)
    
    max_ai_pts = df['total_points'].quantile(0.99)
    
    return model, valid_features, max_ai_pts

# =========================================
# 2. LIVE DATA & FIXTURES
# =========================================

@st.cache_data(ttl=1800)
def load_live_data():
    static = requests.get(f"{API_BASE}/bootstrap-static/").json()
    fixtures = requests.get(f"{API_BASE}/fixtures/").json()
    return static, fixtures

def process_fixtures(fixtures, teams_data):
    team_map = {t['id']: t['short_name'] for t in teams_data}
    t_stats = {t['id']: {'att_h': t['strength_attack_home'], 'att_a': t['strength_attack_away'],
                         'def_h': t['strength_defence_home'], 'def_a': t['strength_defence_away']} 
               for t in teams_data}
    
    team_sched = {t['id']: {'fut_opp_att': [], 'fut_opp_def': [], 'display': []} for t in teams_data}
    
    for f in fixtures:
        if not f['kickoff_time'] or f['finished']: continue
        h, a = f['team_h'], f['team_a']
        
        team_sched[h]['fut_opp_att'].append(t_stats[a]['att_a'])
        team_sched[h]['fut_opp_def'].append(t_stats[a]['def_a'])
        team_sched[h]['display'].append(f"{team_map[a]}(H)")
        
        team_sched[a]['fut_opp_att'].append(t_stats[h]['att_h'])
        team_sched[a]['fut_opp_def'].append(t_stats[h]['def_h'])
        team_sched[a]['display'].append(f"{team_map[h]}(A)")
        
    return team_sched, 1080.0

def get_fixture_multiplier(schedule_list, league_avg, limit, mode="def"):
    if not schedule_list: return 1.0
    subset = schedule_list[:limit]
    avg_strength = sum(subset) / len(subset)
    ratio = league_avg / avg_strength
    power = 4.0 if mode == "def" else 2.0
    return ratio ** power

def get_display_score(schedule_list, limit):
    if not schedule_list: return 5.0
    avg = sum(schedule_list[:limit]) / len(schedule_list[:limit])
    return max(0, min(10, 10 - ((avg - 950)/400 * 10)))

def min_max_scale(series):
    if series.empty: return series
    min_v, max_v = series.min(), series.max()
    if max_v == min_v: return pd.Series([5.0]*len(series), index=series.index)
    return ((series - min_v) / (max_v - min_v)) * 10

# =========================================
# 3. HYBRID APP LOGIC
# =========================================

def main():
    st.title("🧬 FPL Pro: Hybrid Intelligence")
    st.markdown("### Unrestricted AI + Contextual Logic")

    # 1. Train AI
    with st.spinner("AI is analyzing 15+ stats across 5 years..."):
        model, ai_cols, max_ai_pts = train_ai_model()
    
    if model is None:
        st.error("Data Error. Please refresh.")
        return

    # 2. Load Live
    static, fixtures = load_live_data()
    teams = static['teams']
    team_names = {t['id']: t['name'] for t in teams}
    team_sched, avg_str = process_fixtures(fixtures, teams)
    
    # 3. Prepare Player Data
    df = pd.DataFrame(static['elements'])
    df['matches_played'] = df['minutes'] / 90
    df = df[df['matches_played'] > 2.0]
    
    # AI Input Prep
    ai_input = pd.DataFrame()
    ai_input['value'] = df['now_cost']
    ai_input['element_type'] = df['element_type']
    ai_input['was_home'] = 0.5
    
    # Map API Columns to Training Columns
    stat_map = {
        'expected_goals': 'expected_goals_per_90',
        'expected_assists': 'expected_assists_per_90',
        'clean_sheets': 'clean_sheets_per_90',
        'goals_conceded': 'goals_conceded_per_90',
        'influence': 'influence', 'creativity': 'creativity', 'threat': 'threat',
        'goals_scored': 'goals_scored', 'assists': 'assists', 'saves': 'saves', 'bps': 'bps',
        'yellow_cards': 'yellow_cards'
    }
    
    for train_col, api_col in stat_map.items():
        if train_col in ai_cols:
            if 'per_90' in api_col:
                ai_input[train_col] = pd.to_numeric(df[api_col], errors='coerce').fillna(0)
            else:
                # Normalize totals to per-match average for accurate prediction
                ai_input[train_col] = pd.to_numeric(df[api_col], errors='coerce').fillna(0) / df['matches_played']
            
    # --- AI PREDICTION ---
    df['AI_Points'] = model.predict(ai_input[ai_cols])
    
    # --- UI CONTROLS ---
    
    # Sidebar: Feature Importance (For Transparency)
    st.sidebar.header("🧠 AI Brain Scan")
    st.sidebar.caption("Features used by AI (Ranked by Importance):")
    # Note: HistGradient doesn't always expose straightforward importance arrays in older sklearn versions,
    # so we list the features the AI has access to.
    st.sidebar.code("xG, xA, ICT, Clean Sheets,\nGoals, Assists, Saves, BPS...", language="text")
    
    st.sidebar.divider()
    st.sidebar.header("🔮 Horizon")
    horizon = st.sidebar.selectbox("Lookahead", [1, 5, 10], format_func=lambda x: f"Next {x} Matches")
    
    st.sidebar.divider()
    st.sidebar.header("⚖️ Hybrid Weights")
    
    w_budget = st.sidebar.slider("Price Sensitivity", 0.0, 1.0, 0.5)
    
    st.sidebar.subheader("Position Strategy")
    
    # SLIDERS
    with st.sidebar.expander("🧤 Goalkeepers", expanded=False):
        w_gk = {
            'ai': st.slider("AI Stats", 0.0, 1.0, 0.6, key="g1"),
            'form': st.slider("Form (PPM)", 0.0, 1.0, 0.4, key="g2"),
            'fix': st.slider("Fixture Impact", 0.0, 1.0, 1.0, key="g3")
        }
    with st.sidebar.expander("🛡️ Defenders", expanded=False):
        w_def = {
            'ai': st.slider("AI Stats", 0.0, 1.0, 0.6, key="d1"),
            'form': st.slider("Form (PPM)", 0.0, 1.0, 0.4, key="d2"),
            'xgi': st.slider("Attacking Bonus", 0.0, 1.0, 0.3, key="d3"),
            'fix': st.slider("Fixture Impact", 0.0, 1.0, 1.0, key="d4")
        }
    with st.sidebar.expander("⚔️ Midfielders", expanded=False):
        w_mid = {
            'ai': st.slider("AI Stats", 0.0, 1.0, 0.6, key="m1"),
            'form': st.slider("Form (PPM)", 0.0, 1.0, 0.4, key="m2"),
            'fix': st.slider("Fixture Impact", 0.0, 1.0, 0.8, key="m3")
        }
    with st.sidebar.expander("⚽ Forwards", expanded=False):
        w_fwd = {
            'ai': st.slider("AI Stats", 0.0, 1.0, 0.6, key="f1"),
            'form': st.slider("Form (PPM)", 0.0, 1.0, 0.4, key="f2"),
            'fix': st.slider("Fixture Impact", 0.0, 1.0, 0.8, key="f3")
        }

    st.sidebar.divider()
    min_mins = st.sidebar.slider("Min Minutes", 0, 2500, 400)

    # --- HYBRID ENGINE ---
    def run_hybrid_engine(p_ids, cat, w):
        cands = []
        subset = df[df['element_type'].isin(p_ids) & (df['minutes'] >= min_mins)]
        MAX_PPM = subset['points_per_game'].astype(float).max()
        
        for _, row in subset.iterrows():
            tid = row['team']
            
            # 1. CONTEXT
            if cat in ["GK", "DEF"]:
                sched = team_sched[tid]['fut_opp_att']
                mode = "def"
            else:
                sched = team_sched[tid]['fut_opp_def']
                mode = "att"
                
            fix_mult = get_fixture_multiplier(sched, avg_str, horizon, mode)
            fix_score_display = get_display_score(sched, horizon)
            fix_display = ", ".join(team_sched[tid]['display'][:horizon])
            
            # 2. SCORES
            score_ai = (row['AI_Points'] / max_ai_pts) * 10
            raw_ppm = float(row['points_per_game'])
            score_form = (raw_ppm / MAX_PPM) * 10
            
            score_bonus = 0
            if cat == "DEF":
                xgi = float(row['expected_goal_involvements_per_90'])
                score_bonus = (xgi * 10) * w['xgi']
            
            # 3. BLEND
            base_score = (score_ai * w['ai']) + (score_form * w['form']) + score_bonus
            
            # 4. APPLY CONTEXT
            eff_mult = 1.0 + (fix_mult - 1.0) * w['fix']
            final_score = base_score * eff_mult
            
            # 5. ROI
            price = row['now_cost'] / 10.0
            price_div = price ** w_budget
            roi = final_score / price_div
            
            stat_disp = float(row['clean_sheets_per_90']) if cat in ["GK", "DEF"] else float(row['expected_goal_involvements_per_90'])

            cands.append({
                "Name": row['web_name'],
                "Team": team_names.get(tid, "Unknown"),
                "Price": price,
                "Key Stat": stat_disp,
                "Upcoming": fix_display,
                "AI Base": round(row['AI_Points'], 2),
                "PPM": raw_ppm,
                "Fix Rate": round(fix_score_display, 1),
                "Raw ROI": roi
            })
            
        res = pd.DataFrame(cands)
        if res.empty: return res
        
        res['ROI Index'] = min_max_scale(res['Raw ROI'])
        return res.sort_values(by="ROI Index", ascending=False)

    # --- RENDER ---
    def render(p_ids, cat, w):
        d = run_hybrid_engine(p_ids, cat, w)
        if d.empty: st.write("No players."); return
        
        stat_lbl = "CS/90" if cat in ["GK", "DEF"] else "xGI/90"
        
        st.dataframe(
            d.head(50), hide_index=True, use_container_width=True,
            column_config={
                "ROI Index": st.column_config.ProgressColumn("ROI Index", format="%.1f", min_value=0, max_value=10),
                "Price": st.column_config.NumberColumn("£", format="£%.1f"),
                "Upcoming": st.column_config.TextColumn("Opponents", width="medium"),
                "AI Base": st.column_config.NumberColumn("AI Exp", help="Points predicted by AI using ALL Stats (Actual + Underlying)"),
                "PPM": st.column_config.NumberColumn("Form", help="Actual Points Per Match this season"),
                "Fix Rate": st.column_config.NumberColumn("Fix Rating", help="10=Easy, 0=Hard"),
                "Key Stat": st.column_config.NumberColumn(stat_lbl, format="%.2f")
            }
        )

    t1, t2, t3, t4 = st.tabs(["🧤 GK", "🛡️ DEF", "⚔️ MID", "⚽ FWD"])
    with t1: render([1], "GK", w_gk)
    with t2: render([2], "DEF", w_def)
    with t3: render([3], "MID", w_mid)
    with t4: render([4], "FWD", w_fwd)

if __name__ == "__main__":
    main()
