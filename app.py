import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
st.set_page_config(page_title="FPL Pro Hybrid 25/26", page_icon="🧬", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 60px; font-weight: 700; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { border-top: 3px solid #00cc00; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
API_BASE = "https://fantasy.premierleague.com/api"

# =========================================
# 1. DATA INFRASTRUCTURE
# =========================================

@st.cache_data(persist="disk") 
def load_training_data():
    """Loads historical data. Checks local file first, then downloads."""
    if os.path.exists("fpl_5_year_history.csv"):
        return pd.read_csv("fpl_5_year_history.csv")
    
    # Fallback Downloader
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
    all_data = []
    
    for season in seasons:
        try:
            url = f"{base_url}/{season}/gws/merged_gw.csv"
            r = requests.get(url)
            if r.status_code == 200:
                # on_bad_lines='skip' handles the 24-25 corruption issue
                df = pd.read_csv(io.BytesIO(r.content), on_bad_lines='skip', low_memory=False)
                
                # We fetch EVERYTHING, but select features later
                cols = ['minutes', 'total_points', 'was_home', 'clean_sheets', 
                        'goals_conceded', 'expected_goals_conceded', # Crucial for Def Logic
                        'expected_goals', 'expected_assists', 
                        'influence', 'creativity', 'threat', 'value', 'element_type']
                
                existing = [c for c in cols if c in df.columns]
                all_data.append(df[existing])
        except: pass
        
    if all_data:
        return pd.concat(all_data).fillna(0)
    return None

@st.cache_resource
def get_trained_model():
    df = load_training_data()
    if df is None: return None, None, None, None
    
    # Filter Starters (>60 mins) to train AI on "Real Matches"
    df = df[df['minutes'] > 60].copy()
    
    # --- FEATURES (Predictive Only) ---
    # We removed Result Stats (Goals/Assists/CS) to prevent data leakage.
    # The AI must predict points based on UNDERLYING stats.
    features = [
        'minutes', 'element_type', 'was_home', 
        'expected_goals', 'expected_assists', 
        'expected_goals_conceded', # AI learns Defense Quality from this
        'influence', 'creativity', 'threat',
        'goals_conceded' # Kept as context for defensive weakness
    ]
    
    valid_features = [f for f in features if f in df.columns]
    
    # RandomForest for Transparency & Accuracy balance
    model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(df[valid_features], df['total_points'])
    
    # Feature Importance
    importance_df = pd.DataFrame({
        'Stat': valid_features,
        'Weight': model.feature_importances_
    }).sort_values(by='Weight', ascending=False)
    importance_df['Weight'] = (importance_df['Weight'] / importance_df['Weight'].sum()) * 100
    
    max_ai_pts = df['total_points'].quantile(0.99)
    
    return model, valid_features, max_ai_pts, importance_df

# =========================================
# 2. LIVE FIXTURE ENGINE
# =========================================

@st.cache_data(ttl=1800)
def get_live_data():
    static = requests.get(f"{API_BASE}/bootstrap-static/").json()
    fixtures = requests.get(f"{API_BASE}/fixtures/").json()
    return static, fixtures

def process_fixtures(fixtures, teams_data):
    team_map = {t['id']: t['short_name'] for t in teams_data}
    # Team Stats
    t_stats = {t['id']: {'att_h': t['strength_attack_home'], 'att_a': t['strength_attack_away'],
                         'def_h': t['strength_defence_home'], 'def_a': t['strength_defence_away']} 
               for t in teams_data}
    
    # Schedule Containers
    team_sched = {t['id']: {'fut_opp_att': [], 'fut_opp_def': [], 'display': []} for t in teams_data}
    
    for f in fixtures:
        if not f['kickoff_time'] or f['finished']: continue
        h, a = f['team_h'], f['team_a']
        
        # Store FUTURE Opponent Strength
        
        # Home Team faces Away Team
        # We store Away Team's ATTACK strength (for Home Def) and DEFENSE strength (for Home Att)
        team_sched[h]['fut_opp_att'].append(t_stats[a]['att_a'])
        team_sched[h]['fut_opp_def'].append(t_stats[a]['def_a'])
        team_sched[h]['display'].append(f"{team_map[a]}(H)")
        
        # Away Team faces Home Team
        team_sched[a]['fut_opp_att'].append(t_stats[h]['att_h'])
        team_sched[a]['fut_opp_def'].append(t_stats[h]['def_h'])
        team_sched[a]['display'].append(f"{team_map[h]}(A)")
        
    return team_sched, 1080.0 # Avg Strength

def calculate_aggressive_multiplier(schedule_list, league_avg, limit, intensity_weight, mode="def"):
    """
    Calculates Fixture Impact.
    mode="def": Defenders need Easy Attacks. Hard Attacks punish heavily.
    mode="att": Attackers need Easy Defenses.
    """
    if not schedule_list: return 1.0
    subset = schedule_list[:limit]
    avg_strength = sum(subset) / len(subset)
    
    # Ratio: League Avg / Opponent Strength
    # If Opponent is Strong (1300), Ratio < 1.0 (Penalty)
    ratio = league_avg / avg_strength
    
    # Base Power: Defenders (4.0) suffer more from hard games (CS Wipeout) than Attackers (2.0)
    base_power = 4.0 if mode == "def" else 2.0
    
    # Final Power scales with User Slider
    final_power = base_power * (intensity_weight * 2.0)
    
    return ratio ** final_power

def get_display_score(schedule_list, limit):
    if not schedule_list: return 5.0
    avg = sum(schedule_list[:limit]) / len(schedule_list[:limit])
    # Normalize 1350->0, 950->10
    return max(0, min(10, 10 - ((avg - 950)/400 * 10)))

def min_max_scale(series):
    if series.empty: return series
    min_v, max_v = series.min(), series.max()
    if max_v == min_v: return pd.Series([5.0]*len(series), index=series.index)
    return ((series - min_v) / (max_v - min_v)) * 10

# =========================================
# 3. MAIN APP LOGIC
# =========================================

def main():
    st.title("🧬 FPL Pro: Hybrid Intelligence")
    
    # 1. Load & Train AI
    with st.spinner("Initializing AI & Data..."):
        model, ai_cols, max_ai_pts, feature_weights = get_trained_model()
    
    if model is None:
        st.warning("⚠️ Downloading Data... (Wait 30s)")
        return

    # 2. Load Live Data
    static, fixtures = get_live_data()
    teams = static['teams']
    team_names = {t['id']: t['name'] for t in teams}
    team_sched, avg_str = process_fixtures(fixtures, teams)
    
    # 3. Prepare Players
    df = pd.DataFrame(static['elements'])
    df['matches_played'] = df['minutes'] / 90
    df = df[df['matches_played'] > 2.0]
    
    # AI Input Prep
    ai_input = pd.DataFrame()
    ai_input['element_type'] = df['element_type']
    ai_input['was_home'] = 0.5 # Neutral
    
    # Map Live API Stats to AI Training Features
    stat_map = {
        'minutes': 'minutes',
        'expected_goals': 'expected_goals_per_90',
        'expected_assists': 'expected_assists_per_90',
        'expected_goals_conceded': 'expected_goals_conceded_per_90', # Correct Mapping
        'goals_conceded': 'goals_conceded_per_90',
        'influence': 'influence', 'creativity': 'creativity', 'threat': 'threat'
    }
    
    for train_col, api_col in stat_map.items():
        if train_col in ai_cols:
            if 'per_90' in api_col:
                ai_input[train_col] = pd.to_numeric(df[api_col], errors='coerce').fillna(0)
            else:
                ai_input[train_col] = pd.to_numeric(df[api_col], errors='coerce').fillna(0) / df['matches_played']
            
    # 4. AI Prediction (Pure Stats)
    df['AI_Points'] = model.predict(ai_input[ai_cols])
    
    # --- UI ---
    
    st.sidebar.header("🧠 AI Brain")
    if feature_weights is not None:
        st.sidebar.caption("Top 5 Predictive Factors (AI Decided):")
        st.sidebar.dataframe(feature_weights.head(5).style.format({"Weight": "{:.1f}%"}), hide_index=True)
    
    st.sidebar.divider()
    st.sidebar.header("🔮 Horizon")
    horizon = st.sidebar.selectbox("Lookahead", [1, 5, 10], format_func=lambda x: f"Next {x} Matches")
    
    st.sidebar.divider()
    st.sidebar.header("⚖️ Weights")
    
    w_budget = st.sidebar.slider("Price Sensitivity", 0.0, 1.0, 0.5)
    
    with st.sidebar.expander("🧤 GK Settings", expanded=False):
        w_gk = {'ai': st.slider("AI (xGC/Stats)", 0.0, 1.0, 0.6, key="g1"), 'form': st.slider("Form (PPM)", 0.0, 1.0, 0.4, key="g2"), 'fix': st.slider("Fixture Impact", 0.0, 1.0, 1.0, key="g3")}
    with st.sidebar.expander("🛡️ DEF Settings", expanded=False):
        w_def = {'ai': st.slider("AI (xGC/Stats)", 0.0, 1.0, 0.6, key="d1"), 'form': st.slider("Form (PPM)", 0.0, 1.0, 0.4, key="d2"), 'xgi': st.slider("Att Bonus", 0.0, 1.0, 0.3, key="d3"), 'fix': st.slider("Fixture Impact", 0.0, 1.0, 1.0, key="d4")}
    with st.sidebar.expander("⚔️ MID Settings", expanded=False):
        w_mid = {'ai': st.slider("AI (xG/xA/Stats)", 0.0, 1.0, 0.6, key="m1"), 'form': st.slider("Form (PPM)", 0.0, 1.0, 0.4, key="m2"), 'fix': st.slider("Fixture Impact", 0.0, 1.0, 0.8, key="m3")}
    with st.sidebar.expander("⚽ FWD Settings", expanded=False):
        w_fwd = {'ai': st.slider("AI (xG/xA/Stats)", 0.0, 1.0, 0.6, key="f1"), 'form': st.slider("Form (PPM)", 0.0, 1.0, 0.4, key="f2"), 'fix': st.slider("Fixture Impact", 0.0, 1.0, 0.8, key="f3")}

    st.sidebar.divider()
    min_mins = st.sidebar.slider("Min Minutes", 0, 2500, 400)

    # --- HYBRID ENGINE ---
    def run_engine(p_ids, cat, w):
        cands = []
        subset = df[df['element_type'].isin(p_ids) & (df['minutes'] >= min_mins)]
        MAX_PPM = subset['points_per_game'].astype(float).max()
        
        for _, row in subset.iterrows():
            tid = row['team']
            
            # 1. CONTEXT ENGINE
            if cat in ["GK", "DEF"]:
                # Defenders vs Opponent Attack
                sched = team_sched[tid]['fut_opp_att']
                mode = "def"
                stat_disp = float(row['expected_goals_conceded_per_90']) # Display xGC
            else:
                # Attackers vs Opponent Defense
                sched = team_sched[tid]['fut_opp_def']
                mode = "att"
                stat_disp = float(row['expected_goal_involvements_per_90']) # Display xGI
                
            # Calculate Multipliers
            eff_mult = calculate_aggressive_multiplier(sched, avg_str, horizon, w['fix'], mode)
            fix_score_display = get_display_score(sched, horizon)
            fix_display = ", ".join(team_sched[tid]['display'][:horizon])
            
            # 2. HYBRID CALCULATION
            score_ai = (row['AI_Points'] / max_ai_pts) * 10
            raw_ppm = float(row['points_per_game'])
            score_form = (raw_ppm / MAX_PPM) * 10
            
            score_bonus = 0
            if cat == "DEF":
                xgi = float(row['expected_goal_involvements_per_90'])
                score_bonus = (xgi * 10) * w['xgi']
            
            # The Blend
            base_score = (score_ai * w['ai']) + (score_form * w['form']) + score_bonus
            
            # Apply Context
            final_score = base_score * eff_mult
            
            # 3. ROI
            price = row['now_cost'] / 10.0
            price_div = price ** w_budget
            roi = final_score / price_div
            
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
        d = run_engine(p_ids, cat, w)
        if d.empty: st.write("No players."); return
        
        if cat in ["GK", "DEF"]:
            stat_lbl = "xGC/90"
            stat_tip = "Exp. Goals Conceded (Lower is Better). Used by AI."
        else:
            stat_lbl = "xGI/90"
            stat_tip = "Exp. Goal Involvement (Higher is Better). Used by AI."
        
        st.dataframe(
            d.head(50), hide_index=True, use_container_width=True,
            column_config={
                "ROI Index": st.column_config.ProgressColumn("ROI Index", format="%.1f", min_value=0, max_value=10),
                "Price": st.column_config.NumberColumn("£", format="£%.1f"),
                "Upcoming": st.column_config.TextColumn("Opponents", width="medium"),
                "AI Base": st.column_config.NumberColumn("AI Exp", help="Points predicted by AI using Underlying Stats"),
                "PPM": st.column_config.NumberColumn("Form", help="Actual Points Per Match"),
                "Fix Rate": st.column_config.NumberColumn("Fix Rating", help="10=Easy, 0=Hard"),
                "Key Stat": st.column_config.NumberColumn(stat_lbl, format="%.2f", help=stat_tip)
            }
        )

    t1, t2, t3, t4 = st.tabs(["🧤 GK", "🛡️ DEF", "⚔️ MID", "⚽ FWD"])
    with t1: render([1], "GK", w_gk)
    with t2: render([2], "DEF", w_def)
    with t3: render([3], "MID", w_mid)
    with t4: render([4], "FWD", w_fwd)

if __name__ == "__main__":
    main()
