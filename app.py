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

# --- ADVANCED FIXTURE ENGINE ---
def process_fixtures_with_context(fixtures, teams_data):
    """
    Calculates distinct Attacking and Defensive favourability scores based on
    Home/Away splits and Opponent Strength.
    """
    team_map = {t['id']: t['short_name'] for t in teams_data}
    
    # Extract Team Strengths (Inverted logic: High Strength = Hard Opponent)
    # We normalize roughly around 1000-1350 range provided by API
    t_stats = {}
    for t in teams_data:
        t_stats[t['id']] = {
            'att_h': t['strength_attack_home'],
            'att_a': t['strength_attack_away'],
            'def_h': t['strength_defence_home'],
            'def_a': t['strength_defence_away']
        }
    
    # Initialize Schedule Containers
    # 'att_sched': How easy is it to ATTACK against this schedule?
    # 'def_sched': How easy is it to DEFEND against this schedule?
    team_sched = {t['id']: {'past_att': [], 'fut_att': [], 'past_def': [], 'fut_def': []} for t in teams_data}

    avg_strength = 1100.0 # Approximate league average strength metric

    for f in fixtures:
        if not f['kickoff_time']: continue
        h, a = f['team_h'], f['team_a']
        
        # --- CALCULATE FAVOURABILITY ---
        # Score > 1.0 = Easy, Score < 1.0 = Hard
        
        # 1. FOR HOME TEAM (h)
        # To Attack: We face Away Defense. Lower Opp Def = Easier.
        # Logic: Avg / Opp_Def
        h_att_fav = (avg_strength / t_stats[a]['def_a']) * 1.15 # +15% Home Boost
        # To Defend: We face Away Attack. Lower Opp Att = Easier.
        h_def_fav = (avg_strength / t_stats[a]['att_a']) * 1.10 # +10% Home Boost

        # 2. FOR AWAY TEAM (a)
        # To Attack: We face Home Defense.
        a_att_fav = (avg_strength / t_stats[h]['def_h']) * 0.9 # -10% Away Penalty
        # To Defend: We face Home Attack.
        a_def_fav = (avg_strength / t_stats[h]['att_h']) * 0.85 # -15% Away Penalty

        # Map to Display Strings
        h_disp = f"{team_map[a]}(H)"
        a_disp = f"{team_map[h]}(A)"

        # Objects
        h_att_obj = {'score': h_att_fav, 'display': h_disp}
        h_def_obj = {'score': h_def_fav, 'display': h_disp}
        
        a_att_obj = {'score': a_att_fav, 'display': a_disp}
        a_def_obj = {'score': a_def_fav, 'display': a_disp}

        if f['finished']:
            team_sched[h]['past_att'].append(h_att_obj)
            team_sched[h]['past_def'].append(h_def_obj)
            team_sched[a]['past_att'].append(a_att_obj)
            team_sched[a]['past_def'].append(a_def_obj)
        else:
            team_sched[h]['fut_att'].append(h_att_obj)
            team_sched[h]['fut_def'].append(h_def_obj)
            team_sched[a]['fut_att'].append(a_att_obj)
            team_sched[a]['fut_def'].append(a_def_obj)

    return team_sched

def get_aggregated_data(schedule_list, limit=None):
    # Normalizes the multiplier back to a 0-10 scale for the UI/Logic
    # Base Multiplier 1.0 -> equates to roughly 5.0 on 0-10 scale
    if not schedule_list: return 5.0, "-"
    
    subset = schedule_list[:limit] if limit else schedule_list
    avg_mult = sum(item['score'] for item in subset) / len(subset)
    
    # Convert Multiplier (0.6 to 1.4) to Score (2.0 to 8.0)
    # Formula: (Multiplier - 0.5) * 10
    # 1.0 -> 5.0
    # 1.3 -> 8.0 (Easy)
    # 0.7 -> 2.0 (Hard)
    score = max(1.0, min(10.0, (avg_mult - 0.5) * 10))
    
    display_str = ", ".join([item['display'] for item in subset])
    return score, display_str

def min_max_scale(series):
    if series.empty: return series
    min_v, max_v = series.min(), series.max()
    if max_v == min_v: return pd.Series([5.0]*len(series), index=series.index)
    return ((series - min_v) / (max_v - min_v)) * 10

# --- MAIN APP ---
def main():
    st.title("🧠 FPL Pro Predictor: ROI Engine")
    st.markdown("### Context-Aware Model (Stats + Home/Away Splits)")

    data, fixtures = load_data()
    if not data or not fixtures: return

    teams = data['teams']
    team_names = {t['id']: t['name'] for t in teams}
    
    # Run Advanced Fixture Engine
    team_schedule = process_fixtures_with_context(fixtures, teams)
    
    # Team Defense Strength (for static CS calc)
    team_conceded = {t['id']: t['strength_defence_home'] + t['strength_defence_away'] for t in teams}
    max_str = max(team_conceded.values()) if team_conceded else 1
    team_def_strength = {k: 10 - ((v/max_str)*10) + 5 for k,v in team_conceded.items()}

    # --- SIDEBAR ---
    st.sidebar.header("🔮 Prediction Horizon")
    horizon_option = st.sidebar.selectbox(
        "Predict for upcoming:", [1, 5, 10], 
        format_func=lambda x: f"Next {x} Fixture{'s' if x > 1 else ''}", on_change=reset_page
    )

    st.sidebar.divider()
    st.sidebar.header("⚖️ Model Weights")
    w_budget = st.sidebar.slider("Price Importance", 0.0, 1.0, 0.5, key="price_weight", on_change=reset_page)
    
    st.sidebar.divider()
    st.sidebar.subheader("Position Settings")

    # 1. GOALKEEPERS
    with st.sidebar.expander("🧤 Goalkeepers", expanded=False):
        w_cs_gk = st.slider("Clean Sheet Potential", 0.1, 1.0, 0.5, key="gk_cs", on_change=reset_page)
        w_ppm_gk = st.slider("Form (PPM)", 0.1, 1.0, 0.5, key="gk_ppm", on_change=reset_page)
        w_fix_gk = st.slider("Fixture Favourability", 0.1, 1.0, 0.5, key="gk_fix", on_change=reset_page)
        gk_weights = {'cs': w_cs_gk, 'ppm': w_ppm_gk, 'fix': w_fix_gk}

    # 2. DEFENDERS
    with st.sidebar.expander("🛡️ Defenders", expanded=False):
        w_cs_def = st.slider("Clean Sheet Potential", 0.1, 1.0, 0.5, key="def_cs", on_change=reset_page)
        w_xgi_def = st.slider("Attacking Threat (xGI)", 0.1, 1.0, 0.5, key="def_xgi", on_change=reset_page)
        w_ppm_def = st.slider("Form (PPM)", 0.1, 1.0, 0.5, key="def_ppm", on_change=reset_page)
        w_fix_def = st.slider("Fixture Favourability", 0.1, 1.0, 0.5, key="def_fix", on_change=reset_page)
        def_weights = {'cs': w_cs_def, 'xgi': w_xgi_def, 'ppm': w_ppm_def, 'fix': w_fix_def}

    # 3. MIDFIELDERS
    with st.sidebar.expander("⚔️ Midfielders", expanded=False):
        w_xgi_mid = st.slider("Total xGI Threat", 0.1, 1.0, 0.5, key="mid_xgi", on_change=reset_page)
        w_ppm_mid = st.slider("Form (PPM)", 0.1, 1.0, 0.5, key="mid_ppm", on_change=reset_page)
        w_fix_mid = st.slider("Fixture Favourability", 0.1, 1.0, 0.5, key="mid_fix", on_change=reset_page)
        mid_weights = {'xgi': w_xgi_mid, 'ppm': w_ppm_mid, 'fix': w_fix_mid}

    # 4. FORWARDS
    with st.sidebar.expander("⚽ Forwards", expanded=False):
        w_xgi_fwd = st.slider("Total xGI Threat", 0.1, 1.0, 0.5, key="fwd_xgi", on_change=reset_page)
        w_ppm_fwd = st.slider("Form (PPM)", 0.1, 1.0, 0.5, key="fwd_ppm", on_change=reset_page)
        w_fix_fwd = st.slider("Fixture Favourability", 0.1, 1.0, 0.5, key="fwd_fix", on_change=reset_page)
        fwd_weights = {'xgi': w_xgi_fwd, 'ppm': w_ppm_fwd, 'fix': w_fix_fwd}

    st.sidebar.divider()
    min_minutes = st.sidebar.slider("Min. Minutes Played", 0, 2000, 250, key="min_mins", on_change=reset_page)

    # --- ANALYSIS ENGINE ---
    def run_analysis(player_type_ids, pos_category, weights):
        candidates = []
        
        # FPL Point definitions
        if pos_category in ["GK", "DEF"]:
            pts_goal, pts_cs, pts_assist = 6, 4, 3
        elif pos_category == "MID":
            pts_goal, pts_cs, pts_assist = 5, 1, 3
        else:
            pts_goal, pts_cs, pts_assist = 4, 0, 3

        for p in data['elements']:
            if p['element_type'] not in player_type_ids: continue
            if p['minutes'] < min_minutes: continue
            tid = p['team']
            
            # --- SMART FIXTURE SELECTION ---
            # If player is GK/DEF, we care about DEFENSIVE schedule (Opponent Attack Strength)
            # If player is MID/FWD, we care about ATTACKING schedule (Opponent Def Weakness)
            if pos_category in ["GK", "DEF"]:
                past_sched = team_schedule[tid]['past_def']
                fut_sched = team_schedule[tid]['fut_def']
            else:
                past_sched = team_schedule[tid]['past_att']
                fut_sched = team_schedule[tid]['fut_att']
                
            past_score, _ = get_aggregated_data(past_sched)
            future_score, future_display = get_aggregated_data(fut_sched, limit=horizon_option)

            try:
                ppm = float(p['points_per_game'])
                price = p['now_cost'] / 10.0
                price = 4.0 if price <= 0 else price
                minutes = float(p['minutes']) if p['minutes'] > 0 else 1
                
                xG = float(p.get('expected_goals_per_90', 0))
                xA = float(p.get('expected_assists_per_90', 0))
                xGI = float(p.get('expected_goal_involvements_per_90', 0))
                CS_rate = float(p.get('clean_sheets_per_90', 0))
                saves = float(p.get('saves_per_90', 0))
                
                # Bonus Calculation
                total_bonus = float(p.get('bonus', 0))
                bonus_per_90 = (total_bonus / minutes) * 90
                bonus_score = min(10, bonus_per_90 * 8)
                internal_bps_weight = 0.4

                # --- ROI SCORES ---
                
                attack_potential = ((xG * pts_goal) + (xA * pts_assist)) * 1.5
                attack_score = min(10, attack_potential)

                team_factor = team_def_strength[tid] / 10.0 
                def_raw = (CS_rate * pts_cs) * team_factor
                if pos_category == "GK": def_raw += (saves / 3)
                def_score = min(10, def_raw * 2.0)

                # Calculation
                if pos_category == "GK":
                    base_score = (def_score * weights['cs']) + \
                                 (ppm * weights['ppm']) + \
                                 (future_score * weights['fix']) + \
                                 (bonus_score * internal_bps_weight)
                                 
                elif pos_category == "DEF":
                    base_score = (def_score * weights['cs']) + \
                                 (attack_score * weights['xgi']) + \
                                 (ppm * weights['ppm']) + \
                                 (future_score * weights['fix']) + \
                                 (bonus_score * internal_bps_weight)
                                 
                else: # MID/FWD
                    def_component = def_score if pos_category == "MID" else 0
                    base_score = (attack_score * weights['xgi']) + \
                                 (ppm * weights['ppm']) + \
                                 (future_score * weights['fix']) + \
                                 (def_component * 0.1) + \
                                 (bonus_score * internal_bps_weight)

                # Resistance (Using Context-Aware Past Score)
                resistance_factor = max(2.0, min(past_score, 5.0))
                raw_perf_metric = base_score / resistance_factor
                
                status_icon = "✅" if p['status'] == 'a' else ("⚠️" if p['status'] == 'd' else "❌")
                stat_disp = CS_rate if pos_category in ["GK", "DEF"] else xGI

                candidates.append({
                    "Name": f"{status_icon} {p['web_name']}",
                    "Team": team_names[tid],
                    "Price": price,
                    "Key Stat": stat_disp,
                    "Upcoming Fixtures": future_display,
                    "PPM": ppm,
                    "Future Fix": round(future_score, 2),
                    "Past Fix": round(past_score, 2),
                    "Raw_Metric": raw_perf_metric,
                })
            except: continue

        df = pd.DataFrame(candidates)
        if df.empty: return df

        df['Norm_Perf'] = min_max_scale(df['Raw_Metric'])
        df['Value_Metric'] = df['Raw_Metric'] / df['Price']
        df['Norm_Value'] = min_max_scale(df['Value_Metric'])
        df['ROI Index'] = (df['Norm_Perf'] * (1 - w_budget)) + (df['Norm_Value'] * w_budget)
        
        df = df.sort_values(by="ROI Index", ascending=False)
        return df[["ROI Index", "Name", "Team", "Price", "Key Stat", "Upcoming Fixtures", "PPM", "Future Fix", "Past Fix"]]

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
                "Future Fix": st.column_config.NumberColumn("Fut Fix", help="Higher = Easier (Adjusted for H/A & Team Strength)"),
                "Past Fix": st.column_config.NumberColumn("Past Fix", help="Higher = Easier (Adjusted for H/A & Team Strength)"),
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
