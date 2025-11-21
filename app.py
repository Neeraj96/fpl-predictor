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
    
    t_stats = {}
    for t in teams_data:
        t_stats[t['id']] = {
            'att_h': t['strength_attack_home'],
            'att_a': t['strength_attack_away'],
            'def_h': t['strength_defence_home'],
            'def_a': t['strength_defence_away']
        }
        
    team_sched = {t['id']: {'past_opp_att': [], 'past_opp_def': [], 'fut_opp_att': [], 'fut_opp_def': [], 'display': []} for t in teams_data}

    for f in fixtures:
        if not f['kickoff_time']: continue
        h, a = f['team_h'], f['team_a']
        
        # Opponent Stats
        h_opp_att = t_stats[a]['att_a']
        h_opp_def = t_stats[a]['def_a']
        a_opp_att = t_stats[h]['att_h']
        a_opp_def = t_stats[h]['def_h']

        if f['finished']:
            # PAST: For calculating Resistance
            team_sched[h]['past_opp_att'].append(h_opp_att)
            team_sched[h]['past_opp_def'].append(h_opp_def)
            team_sched[a]['past_opp_att'].append(a_opp_att)
            team_sched[a]['past_opp_def'].append(a_opp_def)
        else:
            # FUTURE: For calculating Opportunity
            team_sched[h]['fut_opp_att'].append(h_opp_att)
            team_sched[h]['fut_opp_def'].append(h_opp_def)
            team_sched[h]['display'].append(f"{team_map[a]}(H)")
            
            team_sched[a]['fut_opp_att'].append(a_opp_att)
            team_sched[a]['fut_opp_def'].append(a_opp_def)
            team_sched[a]['display'].append(f"{team_map[h]}(A)")

    return team_sched

def get_fixture_score(schedule_list, limit=None):
    # Returns 0-10 (10 = Easy)
    if not schedule_list: return 5.0
    subset = schedule_list[:limit] if limit else schedule_list
    avg_strength = sum(subset) / len(subset)
    score = 10 - ((avg_strength - 1000) / 350 * 10) # Normalize 1000-1350 to 10-0
    return max(0, min(10, score))

def get_resistance_factor(past_strengths):
    # Returns multiplier (High = Played Hard Teams = Boost)
    if not past_strengths: return 1.0
    avg_strength = sum(past_strengths) / len(past_strengths)
    return avg_strength / 1080.0 # ~League Avg

def min_max_scale(series):
    if series.empty: return series
    min_v, max_v = series.min(), series.max()
    if max_v == min_v: return pd.Series([5.0]*len(series), index=series.index)
    return ((series - min_v) / (max_v - min_v)) * 10

# --- MAIN APP ---
def main():
    st.title("🧠 FPL Pro Predictor: ROI Engine")
    st.markdown("### Advanced Weighted Model (xGC Integrated)")

    data, fixtures = load_data()
    if not data or not fixtures: return

    teams = data['teams']
    team_names = {t['id']: t['name'] for t in teams}
    team_schedule = process_fixtures(fixtures, teams)
    
    # Calculate Team Defensive Strength (0-10) for the composite metric
    # We need this because a player on Man City has higher CS potential than one on Ipswich
    # even if their personal stats are identical.
    team_conceded = {t['id']: t['strength_defence_home'] + t['strength_defence_away'] for t in teams}
    max_str = max(team_conceded.values()) if team_conceded else 1
    min_str = min(team_conceded.values()) if team_conceded else 1
    # Invert: High Strength ID (1350) = Good Defense
    team_def_strength = {k: ((v - min_str) / (max_str - min_str)) * 10 for k,v in team_conceded.items()}

    # Normalization Constants
    all_elements = pd.DataFrame(data['elements'])
    MAX_PPM = all_elements['points_per_game'].astype(float).max()
    MAX_XGI = all_elements['expected_goal_involvements_per_90'].astype(float).max()
    
    # --- SIDEBAR ---
    st.sidebar.header("🔮 Prediction Horizon")
    horizon_option = st.sidebar.selectbox(
        "Analyze next:", [1, 5, 10], 
        format_func=lambda x: f"{x} Fixture{'s' if x > 1 else ''}", on_change=reset_page
    )
    
    st.sidebar.divider()
    st.sidebar.header("⚖️ Impact Weights")
    
    w_budget = st.sidebar.slider("Price Impact", 0.0, 1.0, 0.5, key="price_weight", on_change=reset_page)
    
    st.sidebar.divider()
    st.sidebar.subheader("Stat Importance")

    # 1. GOALKEEPERS
    with st.sidebar.expander("🧤 Goalkeepers", expanded=False):
        w_cs_gk = st.slider("Defensive Solidity (CS + xGC)", 0.0, 1.0, 1.0, key="gk_cs", on_change=reset_page)
        w_ppm_gk = st.slider("Form (PPM)", 0.0, 1.0, 1.0, key="gk_ppm", on_change=reset_page)
        w_fix_gk = st.slider("Fixture Ease", 0.0, 1.0, 1.0, key="gk_fix", on_change=reset_page)
        gk_weights = {'cs': w_cs_gk, 'ppm': w_ppm_gk, 'fix': w_fix_gk}

    # 2. DEFENDERS
    with st.sidebar.expander("🛡️ Defenders", expanded=False):
        w_cs_def = st.slider("Defensive Solidity (CS + xGC)", 0.0, 1.0, 1.0, key="def_cs", on_change=reset_page)
        w_xgi_def = st.slider("Attacking Threat (xGI)", 0.0, 1.0, 1.0, key="def_xgi", on_change=reset_page)
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
            
            # --- CONTEXT ENGINE ---
            if pos_category in ["GK", "DEF"]:
                fut_sched = team_schedule[tid]['opp_att']
                past_sched = team_schedule[tid]['past_opp_att']
            else:
                fut_sched = team_schedule[tid]['opp_def']
                past_sched = team_schedule[tid]['past_opp_def']
                
            fixture_score = get_fixture_score(fut_sched, limit=horizon_option)
            resistance_mult = get_resistance_factor(past_sched)
            
            display_fixtures = ", ".join(team_schedule[tid]['display'][:horizon_option])

            try:
                price = p['now_cost'] / 10.0
                price = 4.0 if price <= 0 else price
                
                raw_ppm = float(p['points_per_game'])
                
                # --- COMPOSITE STATS ---
                
                if pos_category in ["GK", "DEF"]:
                    # 1. DEFENSIVE SOLIDITY COMPOSITE
                    raw_cs = float(p['clean_sheets_per_90'])
                    raw_xgc = float(p.get('expected_goals_conceded_per_90', 0))
                    
                    # Invert xGC: Lower is better. 
                    # A value of 2.5 is terrible (score 0). A value of 0.5 is elite (score 2.0).
                    inv_xgc = max(0, 2.5 - raw_xgc)
                    
                    # Blend: CS (Result) + xGC (Underlying) + Team Strength (Structure)
                    # This creates a robust metric that rewards good performances even if CS was lost
                    comp_def_val = (raw_cs * 20) + (inv_xgc * 3) + (team_def_strength[tid] * 0.5)
                    
                    # Apply Past Context: Did they have low xGC against City? Reward them.
                    adj_def = comp_def_val * resistance_mult
                    
                    # Normalize (Find Max later or use rough estimate for 0-10 scale)
                    # We'll assume a perfect score is around 25.0
                    score_def = min(10, (adj_def / 25.0) * 10)
                    
                    # Attacking Threat for Def
                    raw_xgi = float(p.get('expected_goal_involvements_per_90', 0))
                    adj_xgi = raw_xgi * resistance_mult
                    score_xgi = (adj_xgi / MAX_XGI) * 10 if MAX_XGI > 0 else 0
                    
                    stat_disp = raw_cs # Show CS as primary visual
                    
                else:
                    # ATTACKING COMPOSITE
                    raw_xgi = float(p.get('expected_goal_involvements_per_90', 0))
                    adj_xgi = raw_xgi * resistance_mult
                    score_xgi = (adj_xgi / MAX_XGI) * 10 if MAX_XGI > 0 else 0
                    stat_disp = raw_xgi

                # Common Stats
                adj_ppm = raw_ppm * resistance_mult
                score_ppm = (adj_ppm / MAX_PPM) * 10
                score_fix = fixture_score

                # --- WEIGHTED SUM ---
                total_score = 0
                
                if pos_category == "GK":
                    total_score = (score_def * weights['cs']) + \
                                  (score_ppm * weights['ppm']) + \
                                  (score_fix * weights['fix'])
                                  
                elif pos_category == "DEF":
                    total_score = (score_def * weights['cs']) + \
                                  (score_xgi * weights['xgi']) + \
                                  (score_ppm * weights['ppm']) + \
                                  (score_fix * weights['fix'])
                                  
                else: # MID/FWD
                    total_score = (score_xgi * weights['xgi']) + \
                                  (score_ppm * weights['ppm']) + \
                                  (score_fix * weights['fix'])
                
                # Price Adjustment
                price_divisor = price ** w_budget
                roi_index = total_score / price_divisor

                candidates.append({
                    "Name": p['web_name'],
                    "Team": team_names[tid],
                    "Price": price,
                    "Key Stat": stat_disp,
                    "Upcoming Fixtures": display_fixtures,
                    "PPM": raw_ppm,
                    "Fix Score": round(fixture_score, 1),
                    "Past Diff": round(resistance_mult, 2),
                    "Raw Score": roi_index
                })
            except: continue

        df = pd.DataFrame(candidates)
        if df.empty: return df
        
        df['ROI Index'] = min_max_scale(df['Raw Score'])
        df = df.sort_values(by="ROI Index", ascending=False)
        
        return df[["ROI Index", "Name", "Team", "Price", "Key Stat", "Upcoming Fixtures", "PPM", "Fix Score", "Past Diff"]]

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
                "Fix Score": st.column_config.NumberColumn("Fix Rating", help="10 = Easy. 0 = Hard."),
                "Past Diff": st.column_config.NumberColumn("Past Ctx", help=">1.0: Good stats against Hard teams (Boosted). <1.0: Good stats against Easy teams (Reduced)."),
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
