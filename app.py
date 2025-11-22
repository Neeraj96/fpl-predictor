import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import os
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURATION ---
st.set_page_config(page_title="FPL Prediction App powered by AI", page_icon="🤖", layout="wide")

# --- CUSTOM CSS (BIG TABS & UI) ---
st.markdown("""
<style>
    /* Make the Tabs look like big buttons/boxes */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px 25px;
        font-size: 20px; /* Bigger Font */
        font-weight: 700; /* Bold */
        color: #31333F;
        border: 1px solid #d6d6d6;
        flex-grow: 1; /* Stretch to fill width */
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e6e6e6;
        color: #000000;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff;
        border: 2px solid #00cc00; /* Green highlight */
        color: #00cc00;
        border-bottom: 2px solid #00cc00; /* Box style */
    }
    
    /* Metric Alignment */
    div[data-testid="stMetricValue"] { font-size: 18px; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
API_BASE = "https://fantasy.premierleague.com/api"

# =========================================
# 1. OPTIMIZED DATA LOADER
# =========================================

@st.cache_data(persist="disk") 
def load_training_data():
    """Loads historical data efficiently and guarantees schema safety."""
    if os.path.exists("fpl_5_year_history.csv"):
        df = pd.read_csv("fpl_5_year_history.csv")
    else:
        # Fallback Downloader
        seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
        base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
        all_data = []
        
        # Minimal columns to save memory/bandwidth
        cols_needed = [
            'minutes', 'total_points', 'was_home', 'clean_sheets', 
            'goals_conceded', 'expected_goals', 'expected_assists', 
            'expected_goals_conceded', 'influence', 'creativity', 'threat', 
            'value', 'element_type', 'position', 
            'goals_scored', 'assists', 'saves', 'bps', 'yellow_cards'
        ]

        for season in seasons:
            try:
                url = f"{base_url}/{season}/gws/merged_gw.csv"
                r = requests.get(url)
                if r.status_code == 200:
                    temp = pd.read_csv(io.BytesIO(r.content), on_bad_lines='skip', low_memory=False)
                    existing = [c for c in cols_needed if c in temp.columns]
                    all_data.append(temp[existing])
            except: pass
            
        if all_data:
            df = pd.concat(all_data, ignore_index=True).fillna(0)
        else:
            return None

    # --- CRITICAL FIX: ENSURE ELEMENT_TYPE EXISTS ---
    if 'element_type' not in df.columns:
        if 'position' in df.columns:
            pos_map = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4, 'GKP': 1}
            df['element_type'] = df['position'].map(pos_map).fillna(3)
        else:
            df['element_type'] = 3
            
    df['element_type'] = pd.to_numeric(df['element_type'], errors='coerce').fillna(3).astype(int)
    
    return df

@st.cache_resource
def train_dual_models():
    df = load_training_data()
    if df is None: return None, None, None, None, None, None
    
    if 'minutes' in df.columns:
        df = df[df['minutes'] > 60].copy()
    
    # Split views
    mask_def = df['element_type'].isin([1, 2])
    mask_att = df['element_type'].isin([3, 4])
    
    df_def = df.loc[mask_def].copy()
    df_att = df.loc[mask_att].copy()
    
    if 'expected_goals_conceded' in df_def.columns:
        df_def = df_def[df_def['expected_goals_conceded'] > 0]

    # --- MODEL 1: DEFENSIVE ---
    feats_def = ['minutes', 'was_home', 'element_type', 'expected_goals_conceded', 
                 'influence', 'threat', 'creativity', 'expected_goals', 'expected_assists', 'yellow_cards']
    valid_def = [f for f in feats_def if f in df_def.columns]
    
    model_def = None
    if len(df_def) > 50:
        model_def = RandomForestRegressor(n_estimators=40, max_depth=10, n_jobs=-1, random_state=42)
        model_def.fit(df_def[valid_def], df_def['total_points'])

    # --- MODEL 2: ATTACKING ---
    feats_att = ['minutes', 'was_home', 'element_type', 'expected_goals', 'expected_assists', 
                 'threat', 'creativity', 'yellow_cards']
    valid_att = [f for f in feats_att if f in df_att.columns]
    
    model_att = None
    if len(df_att) > 50:
        model_att = RandomForestRegressor(n_estimators=40, max_depth=10, n_jobs=-1, random_state=42)
        model_att.fit(df_att[valid_att], df_att['total_points'])
    
    max_pts = df['total_points'].quantile(0.99) if 'total_points' in df.columns else 15.0
    
    return model_def, valid_def, model_att, valid_att, max_pts

# =========================================
# 2. OPTIMIZED LIVE DATA & FIXTURES
# =========================================

@st.cache_data(ttl=1800)
def get_live_data():
    static = requests.get(f"{API_BASE}/bootstrap-static/").json()
    fixtures = requests.get(f"{API_BASE}/fixtures/").json()
    return static, fixtures

def process_fixture_lookups(fixtures, teams_data, horizon):
    team_map = {t['id']: t['short_name'] for t in teams_data}
    t_stats = {t['id']: {'att_h': t['strength_attack_home'], 'att_a': t['strength_attack_away'],
                         'def_h': t['strength_defence_home'], 'def_a': t['strength_defence_away']} 
               for t in teams_data}
    
    sched = {t['id']: {'att': [], 'def': [], 'txt': []} for t in teams_data}
    future_fix = [f for f in fixtures if not f['finished'] and f['kickoff_time']]
    
    for f in future_fix:
        h, a = f['team_h'], f['team_a']
        
        if len(sched[h]['att']) < horizon:
            sched[h]['att'].append(t_stats[a]['att_a'])
            sched[h]['def'].append(t_stats[a]['def_a'])
            sched[h]['txt'].append(f"{team_map[a]}(H)")
            
        if len(sched[a]['att']) < horizon:
            sched[a]['att'].append(t_stats[h]['att_h'])
            sched[a]['def'].append(t_stats[h]['def_h'])
            sched[a]['txt'].append(f"{team_map[h]}(A)")
            
    results = {}
    LEAGUE_AVG = 1080.0
    
    for tid, data in sched.items():
        count = len(data['att'])
        if count > 0:
            avg_att = sum(data['att']) / count
            avg_def = sum(data['def']) / count
            
            ratio_def = LEAGUE_AVG / avg_att
            ratio_att = LEAGUE_AVG / avg_def
            
            raw_diff = (avg_att + avg_def) / 2
            vis_score = max(0, min(10, 10 - ((raw_diff - 950)/400 * 10)))
            display = ", ".join(data['txt'])
        else:
            ratio_def, ratio_att, vis_score = 1.0, 1.0, 5.0
            display = "-"
            
        results[tid] = {
            'mult_def': ratio_def,
            'mult_att': ratio_att,
            'vis_score': vis_score,
            'display': display
        }
        
    return results

def calc_vectorized_metrics(df, team_sched_map):
    fix_mult_def_list = []
    fix_mult_att_list = []
    fix_score_list = []
    display_list = []
    
    for tid in df['team']:
        data = team_sched_map.get(tid, {'mult_def': 1.0, 'mult_att': 1.0, 'vis_score': 5.0, 'display': '-'})
        fix_mult_def_list.append(data['mult_def'])
        fix_mult_att_list.append(data['mult_att'])
        fix_score_list.append(data['vis_score'])
        display_list.append(data['display'])
        
    df['fix_mult_def'] = fix_mult_def_list
    df['fix_mult_att'] = fix_mult_att_list
    df['fix_display_score'] = fix_score_list
    df['upcoming'] = display_list
    return df

# =========================================
# 3. MAIN APP
# =========================================

def main():
    st.title("🤖 FPL Prediction App powered by AI")
    
    # 1. Train
    with st.spinner("Training AI Models..."):
        model_def, feat_def, model_att, feat_att, max_ai_pts = train_dual_models()
    
    if model_def is None:
        st.warning("⚠️ Downloading Data... (Wait 30s)")
        return

    # 2. Load Live
    static, fixtures = get_live_data()
    teams = static['teams']
    
    # 3. Prep Dataframe (Vectorized)
    df = pd.DataFrame(static['elements'])
    df['matches_played'] = df['minutes'] / 90
    df = df[df['matches_played'] > 2.0]
    
    # Add Player Team Names (Short)
    team_name_map = {t['id']: t['short_name'] for t in teams}
    df['team_short'] = df['team'].map(team_name_map)
    
    # --- ADD STATUS ICONS ---
    # Mapping: 'a'=Available, 'd'=Doubtful, 'i'/'s'/'u'/'n'=Unavailable
    status_map = {
        'a': '✅', 
        'd': '⚠️', 
        'i': '❌', 
        's': '❌', 
        'u': '❌', 
        'n': '❌'
    }
    # Create a new visual name column: "✅ E.Haaland"
    df['status_icon'] = df['status'].map(status_map).fillna('❓')
    df['player_display'] = df['status_icon'] + " " + df['web_name']
    
    # Feature Prep
    df['was_home'] = 0.5
    cols_to_norm = [
        ('expected_goals', 'expected_goals_per_90'),
        ('expected_assists', 'expected_assists_per_90'),
        ('expected_goals_conceded', 'expected_goals_conceded_per_90'),
        ('clean_sheets', 'clean_sheets_per_90'),
        ('goals_conceded', 'goals_conceded_per_90')
    ]
    for new, old in cols_to_norm:
        df[new] = pd.to_numeric(df[old], errors='coerce').fillna(0)
        
    cols_to_div = ['influence', 'creativity', 'threat', 'yellow_cards']
    for c in cols_to_div:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0) / df['matches_played']
        
    # 4. AI Predict
    if model_def:
        df['pred_def'] = model_def.predict(df[feat_def])
    else: df['pred_def'] = 0
    
    if model_att:
        df['pred_att'] = model_att.predict(df[feat_att])
    else: df['pred_att'] = 0
    
    df['AI_Points'] = np.where(df['element_type'].isin([1, 2]), df['pred_def'], df['pred_att'])
    
    # --- UI ---
    
    st.sidebar.header("🔮 Horizon")
    horizon = st.sidebar.selectbox("Lookahead", [1, 5, 10], format_func=lambda x: f"Next {x} Matches")
    
    # 5. Calculate Fixture Metrics
    team_sched_map = process_fixture_lookups(fixtures, teams, horizon)
    df = calc_vectorized_metrics(df, team_sched_map)
    
    st.sidebar.divider()
    st.sidebar.header("⚖️ Weights")
    w_budget = st.sidebar.slider("Price Sensitivity", 0.0, 1.0, 0.0)
    
    ws = {}
    # Set All Defaults to 0.5
    with st.sidebar.expander("🧤 GK Settings", expanded=False):
        ws['GK'] = {'ai': st.slider("AI Stats", 0., 1., 0.5, key="g1"), 'xgc': st.slider("Manual xGC", 0., 1., 0.5, key="g2"), 'form': st.slider("Form", 0., 1., 0.5, key="g3"), 'fix': st.slider("Fixtures", 0., 1., 0.5, key="g4")}
    with st.sidebar.expander("🛡️ DEF Settings", expanded=False):
        ws['DEF'] = {'ai': st.slider("AI Stats", 0., 1., 0.5, key="d1"), 'xgc': st.slider("Manual xGC", 0., 1., 0.5, key="d2"), 'form': st.slider("Form", 0., 1., 0.5, key="d3"), 'fix': st.slider("Fixtures", 0., 1., 0.5, key="d4")}
    with st.sidebar.expander("⚔️ MID Settings", expanded=False):
        ws['MID'] = {'ai': st.slider("AI Stats", 0., 1., 0.5, key="m1"), 'form': st.slider("Form", 0., 1., 0.5, key="m2"), 'fix': st.slider("Fixtures", 0., 1., 0.5, key="m3")}
    with st.sidebar.expander("⚽ FWD Settings", expanded=False):
        ws['FWD'] = {'ai': st.slider("AI Stats", 0., 1., 0.5, key="f1"), 'form': st.slider("Form", 0., 1., 0.5, key="f2"), 'fix': st.slider("Fixtures", 0., 1., 0.5, key="f3")}

    st.sidebar.divider()
    min_mins = st.sidebar.slider("Min Minutes", 0, 2500, 400)

    # --- ENGINE ---
    def render_table(pos_ids, cat, weights):
        sub = df[df['element_type'].isin(pos_ids) & (df['minutes'] >= min_mins)].copy()
        if sub.empty: st.write("No players."); return
        
        max_ppm = sub['points_per_game'].astype(float).max() or 1.0
        
        sub['norm_ai'] = (sub['AI_Points'] / max_ai_pts) * 10
        sub['norm_form'] = (sub['points_per_game'].astype(float) / max_ppm) * 10
        
        if cat in ['GK', 'DEF']:
            sub['norm_xgc'] = (2.5 - sub['expected_goals_conceded']).clip(lower=0) * 5
            
            sum_weights = weights['ai'] + weights['xgc'] + weights['form']
            if sum_weights == 0: sub['base'] = 10.0
            else:
                sub['base'] = (sub['norm_ai'] * weights['ai']) + \
                              (sub['norm_xgc'] * weights['xgc']) + \
                              (sub['norm_form'] * weights['form'])
            
            power = 4.0 * (weights['fix'] * 2.0)
            sub['eff_fix'] = sub['fix_mult_def'] ** power
            stat_col = 'expected_goals_conceded'
            stat_fmt = "xGC/90"
        else:
            sum_weights = weights['ai'] + weights['form']
            if sum_weights == 0: sub['base'] = 10.0
            else:
                sub['base'] = (sub['norm_ai'] * weights['ai']) + \
                              (sub['norm_form'] * weights['form'])
            
            power = 2.0 * (weights['fix'] * 2.0)
            sub['eff_fix'] = sub['fix_mult_att'] ** power
            stat_col = 'expected_goal_involvements_per_90'
            stat_fmt = "xGI/90"
            
        sub['final_score'] = sub['base'] * sub['eff_fix']
        sub['price'] = sub['now_cost'] / 10.0
        sub['roi_raw'] = sub['final_score'] / (sub['price'] ** w_budget)
        sub['ROI Index'] = (sub['roi_raw'] / sub['roi_raw'].max()) * 10
        
        display = sub[['ROI Index', 'player_display', 'team_short', 'price', 'upcoming', 'points_per_game', 'fix_display_score', stat_col]].sort_values(by='ROI Index', ascending=False).head(50)
        
        st.dataframe(
            display, hide_index=True, use_container_width=True,
            column_config={
                "ROI Index": st.column_config.ProgressColumn("ROI Index", format="%.1f", min_value=0, max_value=10),
                "player_display": st.column_config.TextColumn("Player", width="medium"),
                "team_short": st.column_config.TextColumn("Team", width="small"),
                "price": st.column_config.NumberColumn("£", format="£%.1f"),
                "upcoming": st.column_config.TextColumn("Opponents", width="medium"),
                "points_per_game": st.column_config.NumberColumn("Form", format="%.1f"),
                "fix_display_score": st.column_config.NumberColumn("Fixture Rating", help="10=Easy"),
                stat_col: st.column_config.NumberColumn(stat_fmt, format="%.2f")
            }
        )

    t1, t2, t3, t4 = st.tabs(["🧤 GOALKEEPERS", "🛡️ DEFENDERS", "⚔️ MIDFIELDERS", "⚽ FORWARDS"])
    with t1: render_table([1], "GK", ws['GK'])
    with t2: render_table([2], "DEF", ws['DEF'])
    with t3: render_table([3], "MID", ws['MID'])
    with t4: render_table([4], "FWD", ws['FWD'])

if __name__ == "__main__":
    main()
