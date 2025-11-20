import streamlit as st
import requests
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="FPL Pro Predictor 25/26", page_icon="⚽", layout="wide")

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

# --- SESSION STATE ---
if 'page' not in st.session_state: st.session_state.page = 0
def reset_page(): st.session_state.page = 0

# --- CONSTANTS ---
API_BASE = "https://fantasy.premierleague.com/api"

# --- DATA LOADING ---
@st.cache_data(ttl=1800)
def load_data():
    try:
        bootstrap = requests.get(f"{API_BASE}/bootstrap-static/").json()
    except:
        st.error("API Error: Could not fetch static data.")
        return None, None
    try:
        fixtures = requests.get(f"{API_BASE}/fixtures/").json()
    except:
        st.error("API Error: Could not fetch fixtures.")
        return bootstrap, None
    return bootstrap, fixtures

# --- FIXTURE ENGINE ---
def process_fixtures(fixtures, teams_data):
    team_map = {t['id']: t['short_name'] for t in teams_data}
    
    # 1. Calculate Raw Strengths
    t_stats = {}
    for t in teams_data:
        t_stats[t['id']] = {
            'att_h': t['strength_attack_home'],
            'att_a': t['strength_attack_away'],
            'def_h': t['strength_defence_home'],
            'def_a': t['strength_defence_away']
        }
        
    team_sched = {t['id']: {'opp_att': [], 'opp_def': [], 'display': []} for t in teams_data}

    for f in fixtures:
        if not f['kickoff_time'] or f['finished']: continue
        h, a = f['team_h'], f['team_a']
        
        # Store specific strength of the opponent they are facing
        # Home Team faces Away Att/Def
        team_sched[h]['opp_att'].append(t_stats[a]['att_a'])
        team_sched[h]['opp_def'].append(t_stats[a]['def_a'])
        team_sched[h]['display'].append(f"{team_map[a]}(H)")
        
        # Away Team faces Home Att/Def
        team_sched[a]['opp_att'].append(t_stats[h]['att_h'])
        team_sched[a]['opp_def'].append(t_stats[h]['def_h'])
        team_sched[a]['display'].append(f"{team_map[h]}(A)")

    return team_sched

def get_fixture_score(schedule_list, limit=None, opponent_type="def"):
    # Returns a 0-10 score where 10 is easiest, 0 is hardest
    if not schedule_list: return 5.0, "-"
    
    subset = schedule_list[:limit] if limit else schedule_list
    
    # Average Strength of Opponent
    avg_strength = sum(subset) / len(subset)
    
    # Normalize (League Avg ~1100. Max ~1350. Min ~1000)
    # We want Low Strength to be High Score (Easy)
    # Formula: 1350 (Hard) -> 0, 1000 (Easy) -> 10
    score = 10 - ((avg_strength - 1000) / 350 * 10)
    score = max(0, min(10, score)) # Clamp
    
    return score

def min_max_scale(series):
    if series.empty: return series
    min_v, max_v = series.min(), series.max()
    if max_v == min_v: return pd.Series([5.0]*len(series), index=series.index)
    return ((series - min_v) / (max_v - min_v)) * 10

# --- MAIN APP ---
def main():
    st.title("🧠 FPL Pro Predictor: ROI Engine")
    st.markdown("### User-Controlled Weighted Model")

    data, fixtures = load_data()
    if not data or not fixtures: return

    teams = data['teams']
    team_names = {t['id']: t['name'] for t in teams}
    team_schedule = process_fixtures(fixtures, teams)
    
    # Calculate dynamic maxes for normalization
    # We need to know what "Good" looks like to scale to 0-10
    all_elements = pd.DataFrame(data['elements'])
    MAX_PPM = all_elements['points_per_game'].astype(float).max()
    MAX_CS = all_elements['clean_sheets_per_90'].astype(float).max()
    MAX_XGI = all_elements['expected_goal_involvements_per_90'].astype(float).max()
    
    # --- SIDEBAR ---
    st.sidebar.header("🔮 Prediction Horizon")
    horizon_option = st.sidebar.selectbox(
        "Analyze next:", [1, 5, 10], 
        format_func=lambda x: f"{x} Fixture{'s' if x > 1 else ''}", on_change=reset_page
    )
    
    st.sidebar.divider()
    st.sidebar.header("⚖️ Impact Weights")
    
    # DIRECT PRICE CONTROL
    w_budget = st.sidebar.slider(
        "Price Impact", 0.0, 1.0, 0.5, 
        help="0.0 = Ignore Price (Best Players). 1.0 = Full Price Sensitivity (Best Value).",
        key="price_weight", on_change=reset_page
    )
    
    st.sidebar.divider()
    st.sidebar.subheader("Stat Importance")

    # 1. GOALKEEPERS
    with st.sidebar.expander("🧤 Goalkeepers", expanded=False):
        w_cs_gk = st.slider("Clean Sheet Ability", 0.0, 1.0, 1.0, key="gk_cs", on_change=reset_page)
        w_ppm_gk = st.slider("Form (PPM)", 0.0, 1.0, 1.0, key="gk_ppm", on_change=reset_page)
        w_fix_gk = st.slider("Fixture Ease", 0.0, 1.0, 1.0, key="gk_fix", on_change=reset_page)
        gk_weights = {'cs': w_cs_gk, 'ppm': w_ppm_gk, 'fix': w_fix_gk}

    # 2. DEFENDERS
    with st.sidebar.expander("🛡️ Defenders", expanded=False):
        w_cs_def = st.slider("Clean Sheet Ability", 0.0, 1.0, 1.0, key="def_cs", on_change=reset_page)
        w_xgi_def = st.slider("Attacking Threat", 0.0, 1.0, 1.0, key="def_xgi", on_change=reset_page)
        w_ppm_def = st.slider("Form (PPM)", 0.0, 1.0, 1.0, key="def_ppm", on_change=reset_page)
        w_fix_def = st.slider("Fixture Ease", 0.0, 1.0, 1.0, key="def_fix", on_change=reset_page)
        def_weights = {'cs': w_cs_def, 'xgi': w_xgi_def, 'ppm': w_ppm_def, 'fix': w_fix_def}

    # 3. MIDFIELDERS
    with st.sidebar.expander("⚔️ Midfielders", expanded=False):
        w_xgi_mid = st.slider("Goal/Assist Threat", 0.0, 1.0, 1.0, key="mid_xgi", on_change=reset_page)
        w_ppm_mid = st.slider("Form (PPM)", 0.0, 1.0, 1.0, key="mid_ppm", on_change=reset_page)
        w_fix_mid = st.slider("Fixture Ease", 0.0, 1.0, 1.0, key="mid_fix", on_change=reset_page)
        mid_weights = {'xgi': w_xgi_mid, 'ppm': w_ppm_mid, 'fix': w_fix_mid}

    # 4. FORWARDS
    with st.sidebar.expander("⚽ Forwards", expanded=False):
        w_xgi_fwd = st.slider("Goal/Assist Threat", 0.0, 1.0, 1.0, key="fwd_xgi", on_change=reset_page)
        w_ppm_fwd = st.slider("Form (PPM)", 0.0, 1.0, 1.0, key="fwd_ppm", on_change=reset_page)
        w_fix_fwd = st.slider("Fixture Ease", 0.0, 1.0, 1.0, key="fwd_fix", on_change=reset_page)
        fwd_weights = {'xgi': w_xgi_fwd, 'ppm': w_ppm_fwd, 'fix': w_fix_fwd}

    st.sidebar.divider()
    min_minutes = st.sidebar.slider("Min. Minutes Played", 0, 2000, 250, key="min_mins", on_change=reset_page)

    # --- ANALYSIS ENGINE ---
    def run_analysis(player_type_ids, pos_category, weights):
        candidates = []

        for p in data['elements']:
            if p['element_type'] not in player_type_ids: continue
            if p['minutes'] < min_minutes: continue
            tid = p['team']
            
            # FIXTURE SCORES (0-10 Scale)
            # GK/DEF care about Opponent Attack Strength
            # MID/FWD care about Opponent Defense Strength
            if pos_category in ["GK", "DEF"]:
                opp_sched = team_schedule[tid]['opp_att']
            else:
                opp_sched = team_schedule[tid]['opp_def']
            
            # Retrieve normalized fixture score (10=Easy, 0=Hard)
            fixture_score = get_fixture_score(opp_sched, limit=horizon_option)
            display_fixtures = ", ".join(team_schedule[tid]['display'][:horizon_option])

            try:
                # RAW STATS
                price = p['now_cost'] / 10.0
                if price <= 0: price = 4.0
                
                raw_ppm = float(p['points_per_game'])
                raw_cs = float(p['clean_sheets_per_90'])
                raw_xgi = float(p.get('expected_goal_involvements_per_90', 0))

                # NORMALIZE TO 0-10 (Relative to League Best)
                # This ensures that a weight of "1.0" on PPM means the same as "1.0" on xGI
                score_ppm = (raw_ppm / MAX_PPM) * 10
                score_cs = (raw_cs / MAX_CS) * 10 if MAX_CS > 0 else 0
                score_xgi = (raw_xgi / MAX_XGI) * 10 if MAX_XGI > 0 else 0
                score_fix = fixture_score # Already 0-10

                # DIRECT WEIGHTED SUM
                total_score = 0
                
                if pos_category == "GK":
                    total_score = (score_cs * weights['cs']) + \
                                  (score_ppm * weights['ppm']) + \
                                  (score_fix * weights['fix'])
                                  
                elif pos_category == "DEF":
                    total_score = (score_cs * weights['cs']) + \
                                  (score_xgi * weights['xgi']) + \
                                  (score_ppm * weights['ppm']) + \
                                  (score_fix * weights['fix'])
                                  
                else: # MID/FWD
                    total_score = (score_xgi * weights['xgi']) + \
                                  (score_ppm * weights['ppm']) + \
                                  (score_fix * weights['fix'])
                
                # PRICE ADJUSTMENT (Power Law)
                # If w_budget is 0, divisor is 1.0 (Score unchanged)
                # If w_budget is 1, divisor is Price (Score / Price)
                price_divisor = price ** w_budget
                roi_index = total_score / price_divisor
                
                stat_disp = raw_cs if pos_category in ["GK", "DEF"] else raw_xgi

                candidates.append({
                    "Name": p['web_name'],
                    "Team": team_names[tid],
                    "Price": price,
                    "Key Stat": stat_disp,
                    "Upcoming Fixtures": display_fixtures,
                    "PPM": raw_ppm,
                    "Fix Score": round(fixture_score, 1),
                    "Raw Score": roi_index # For sorting before display
                })
            except: continue

        df = pd.DataFrame(candidates)
        if df.empty: return df
        
        # Final 0-10 Scaling for the UI Bar
        df['ROI Index'] = min_max_scale(df['Raw Score'])
        
        df = df.sort_values(by="ROI Index", ascending=False)
        return df[["ROI Index", "Name", "Team", "Price", "Key Stat", "Upcoming Fixtures", "PPM", "Fix Score"]]

    # --- RENDER ---
    def render_tab(p_ids, pos_cat, weights):
        df = run_analysis(p_ids, pos_cat, weights)
        if df.empty: st.warning("No players found."); return

        items_per_page = 50
        total_pages = max(1, (len(df) + items_per_page - 1) // items_per_page)
        if st.session_state.page >= total_pages: st.session_state.page = total_pages - 1
        start, end = st.session_state.page * items_per_page, (st.session_state.page + 1) * items_per_page
        
        stat_label = "CS/90" if pos_cat in ["GK", "DEF"] else "xGI/90"
        
        st.dataframe(
            df.iloc[start:end], hide_index=True, use_container_width=True,
            column_config={
                "ROI Index": st.column_config.ProgressColumn("ROI Index", format="%.1f", min_value=0, max_value=10),
                "Price": st.column_config.NumberColumn("£", format="£%.1f"),
                "Key Stat": st.column_config.NumberColumn(stat_label, format="%.2f"),
                "Upcoming Fixtures": st.column_config.TextColumn("Opponents", width="medium"),
                "PPM": st.column_config.NumberColumn("Pts/G", format="%.1f"),
                "Fix Score": st.column_config.NumberColumn("Fix Rating", help="0-10 Scale. 10 = Easiest Games. 0 = Hardest Games."),
            }
        )
        c1, _, c3 = st.columns([1, 2, 1])
        if c1.button("⬅️ Previous", key=f"p_{pos_cat}"): st.session_state.page -= 1; st.rerun()
        if c3.button("Next ➡️", key=f"n_{pos_cat}"): st.session_state.page += 1; st.rerun()

    tab_gk, tab_def, tab_mid, tab_fwd = st.tabs(["🧤 GOALKEEPERS", "🛡️ DEFENDERS", "⚔️ MIDFIELDERS", "⚽ FORWARDS"])
    with tab_gk: render_tab([1], "GK", gk_weights)
    with tab_def: render_tab([2], "DEF", def_weights)
    with tab_mid: render_tab([3], "MID", mid_weights)
    with tab_fwd: render_tab([4], "FWD", fwd_weights)

if __name__ == "__main__":
    main()
