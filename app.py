import streamlit as st
import requests
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="FPL 10-GW Horizon", page_icon="🔭", layout="wide")

# --- CSS FOR WIDER TABLES ---
st.markdown("""
<style>
    .stDataFrame { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADING ---
@st.cache_data(ttl=600)
def load_data():
    base_url = "https://fantasy.premierleague.com/api/"
    try:
        static = requests.get(base_url + "bootstrap-static/").json()
        fixtures = requests.get(base_url + "fixtures/").json()
        return static, fixtures
    except:
        return None, None

def get_team_leakiness(static_data):
    """
    Calculates average xGC (Expected Goals Conceded) per 90 for each team's defenders.
    Used to predict Clean Sheet potential.
    """
    team_xgc = {t['id']: [] for t in static_data['teams']}
    for p in static_data['elements']:
        if p['element_type'] in [1, 2] and p['minutes'] > 450: # GKP & DEF
            try:
                xgc = float(p.get('expected_goals_conceded_per_90', 0))
                team_xgc[p['team']].append(xgc)
            except: continue
            
    team_strength = {}
    for t_id, values in team_xgc.items():
        # Default to 1.5 if no data, otherwise average
        team_strength[t_id] = sum(values) / len(values) if values else 1.5
    return team_strength

def get_10_game_schedule(static_data, fixture_data):
    """Maps Team ID to list of next 10 opponents."""
    teams = {t['id']: t['short_name'] for t in static_data['teams']}
    next_gw = next((e['id'] for e in static_data['events'] if e['is_next']), 1)
    
    schedule = {t_id: [] for t_id in teams}
    
    for f in fixture_data:
        if f['event'] and f['event'] >= next_gw:
            h, a = f['team_h'], f['team_a']
            
            # Add to Home Team
            if len(schedule[h]) < 10:
                schedule[h].append({"opp": teams[a], "diff": f['team_h_difficulty']})
            
            # Add to Away Team
            if len(schedule[a]) < 10:
                schedule[a].append({"opp": teams[h], "diff": f['team_a_difficulty']})
                
    return schedule

# --- 2. SCORING MODELS ---

def calc_attacker_score(p, schedule_map, w_xgi, w_form, w_fix):
    """Predicts points based on Total xGI + 10-Game Fixtures"""
    try:
        xgi = float(p.get('expected_goal_involvements', 0))
        form = float(p['form'])
        
        my_fixtures = schedule_map.get(p['team'], [])
        if not my_fixtures: return 0, "", 0
        
        # Calculate 10-game average difficulty
        avg_diff = sum(m['diff'] for m in my_fixtures) / len(my_fixtures)
        
        # Scoring Math
        # xGI: Top players have ~15. Score: 15 * 0.8 = 12
        s_xgi = xgi * 0.8
        # Fixtures: (5 - AvgDiff) * Multiplier. If AvgDiff is 2 (Easy), Score is 9.
        s_fix = (5.0 - avg_diff) * 3.0
        
        total = (s_xgi * w_xgi) + (form * w_form) + (s_fix * w_fix)
        
        # Format Schedule String
        sched_str = ", ".join([f"{m['opp']}({m['diff']})" for m in my_fixtures])
        return total, sched_str, avg_diff
    except: return 0, "", 0

def calc_defender_score(p, schedule_map, leakiness_map, w_cs, w_fix, w_xgi):
    """Predicts points based on Team Defense + 10-Game Fixtures + xGI"""
    try:
        xgi = float(p.get('expected_goal_involvements', 0))
        team_xgc = leakiness_map.get(p['team'], 1.5)
        
        my_fixtures = schedule_map.get(p['team'], [])
        if not my_fixtures: return 0, "", 0
        
        avg_diff = sum(m['diff'] for m in my_fixtures) / len(my_fixtures)
        
        # Scoring Math
        # Clean Sheet Potential: Invert Team xGC. Lower xGC = Higher Score.
        s_cs = (3.0 - team_xgc) * 5.0 
        # Fixtures are CRITICAL for defenders
        s_fix = (5.0 - avg_diff) * 4.0
        # Attacking Bonus
        s_att = xgi * 2.5
        
        total = (s_cs * w_cs) + (s_fix * w_fix) + (s_att * w_xgi)
        
        sched_str = ", ".join([f"{m['opp']}({m['diff']})" for m in my_fixtures])
        return total, sched_str, avg_diff
    except: return 0, "", 0

# --- 3. MAIN APP ---
def main():
    st.title("🔭 FPL 10-Week Horizon Predictor")
    st.markdown("Analyzing **Next 10 Fixtures** to find long-term hauls.")
    
    with st.spinner("Loading full season data..."):
        static, fixtures = load_data()
        if not static:
            st.error("API Error")
            return
        
        schedules = get_10_game_schedule(static, fixtures)
        leakiness = get_team_leakiness(static)
        teams = {t['id']: t['short_name'] for t in static['teams']}

    # --- SIDEBAR WEIGHTS ---
    st.sidebar.header("⚙️ Algorithm Weights")
    with st.sidebar.expander("Attacker Weights", expanded=True):
        aw_xgi = st.slider("Total xGI", 0.1, 1.0, 0.6)
        aw_fix = st.slider("Fixture Ease (10 Gms)", 0.1, 1.0, 0.3)
        aw_form = st.slider("Current Form", 0.1, 1.0, 0.2)
    
    with st.sidebar.expander("Defender Weights", expanded=True):
        dw_cs = st.slider("Team Clean Sheet Prob", 0.1, 1.0, 0.5)
        dw_fix = st.slider("Fixture Ease (10 Gms)", 0.1, 1.0, 0.4)
        dw_att = st.slider("Attacking Threat", 0.1, 1.0, 0.2)
        
    min_mins = st.sidebar.number_input("Min Minutes Played", 0, 3000, 500)

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["⚔️ Attackers", "🛡️ Defenders", "💎 Best Value (ROI)"])

    # DATA PROCESSING
    all_attackers = []
    all_defenders = []
    
    # Loop once through all players
    for p in static['elements']:
        if p['minutes'] < min_mins: continue
        
        price = p['now_cost'] / 10.0
        pos = p['element_type'] # 1=GKP, 2=DEF, 3=MID, 4=FWD
        
        # ATTACKERS
        if pos in [3, 4]:
            score, sched, avg_diff = calc_attacker_score(p, schedules, aw_xgi, aw_form, aw_fix)
            row = {
                "Name": p['web_name'], "Team": teams[p['team']], "Pos": "MID" if pos==3 else "FWD",
                "Price": price, "xGI": float(p['expected_goal_involvements']),
                "Avg Diff": avg_diff, "Next 10": sched, "Score": score
            }
            all_attackers.append(row)
            
        # DEFENDERS
        if pos in [1, 2]:
            score, sched, avg_diff = calc_defender_score(p, schedules, leakiness, dw_cs, dw_fix, dw_att)
            row = {
                "Name": p['web_name'], "Team": teams[p['team']], "Pos": "DEF" if pos==2 else "GKP",
                "Price": price, "Team xGC": round(leakiness.get(p['team'], 0), 2),
                "Avg Diff": avg_diff, "Next 10": sched, "Score": score
            }
            all_defenders.append(row)

    # --- TAB 1: ATTACKERS ---
    with tab1:
        # Price filter specific to this tab
        p_min, p_max = st.slider("Filter Price (£m)", 4.0, 15.0, (4.0, 15.0), key="p_att")
        
        df_att = pd.DataFrame(all_attackers)
        # Filter by price
        df_att = df_att[(df_att['Price'] >= p_min) & (df_att['Price'] <= p_max)]
        df_att = df_att.sort_values("Score", ascending=False).head(50)
        
        st.dataframe(df_att, hide_index=True, use_container_width=True, column_config={
            "Score": st.column_config.ProgressColumn("Explosion Index", format="%.1f", min_value=0, max_value=max(df_att['Score'])),
            "Price": st.column_config.NumberColumn("£", format="£%.1f"),
            "Avg Diff": st.column_config.NumberColumn("Diff (10gms)", format="%.2f", help="1=Easy, 5=Hard"),
            "Next 10": st.column_config.TextColumn("Next 10 Fixtures", width="large")
        })

    # --- TAB 2: DEFENDERS ---
    with tab2:
        # Price filter specific to this tab
        dp_min, dp_max = st.slider("Filter Price (£m)", 3.5, 10.0, (3.5, 9.0), key="p_def")
        
        df_def = pd.DataFrame(all_defenders)
        df_def = df_def[(df_def['Price'] >= dp_min) & (df_def['Price'] <= dp_max)]
        df_def = df_def.sort_values("Score", ascending=False).head(50)
        
        st.dataframe(df_def, hide_index=True, use_container_width=True, column_config={
            "Score": st.column_config.ProgressColumn("Clean Sheet Potential", format="%.1f", min_value=0, max_value=max(df_def['Score'])),
            "Price": st.column_config.NumberColumn("£", format="£%.1f"),
            "Team xGC": st.column_config.NumberColumn("Leakiness", format="%.2f", help="Lower = Better Defense"),
            "Next 10": st.column_config.TextColumn("Next 10 Fixtures", width="large")
        })

    # --- TAB 3: BEST VALUE (UNFILTERED) ---
    with tab3:
        st.markdown("### 💎 True Value (ROI)")
        st.info("This list compares **Predicted Points per £ Million**. It includes ALL players (Premiums & Budget) to find the true best assets.")
        
        # Combine lists
        all_players = all_attackers + all_defenders
        
        value_list = []
        for p in all_players:
            if p['Price'] > 0:
                # ROI Calculation: Score divided by Price
                roi = p['Score'] / p['Price']
                
                # We copy the dict to modify it for this specific view
                p_copy = p.copy()
                p_copy['Value Score'] = roi
                value_list.append(p_copy)
        
        df_val = pd.DataFrame(value_list).sort_values("Value Score", ascending=False).head(50)
        
        # Clean up columns for this view
        cols = ["Name", "Team", "Pos", "Price", "Value Score", "Score", "Next 10"]
        df_val = df_val[cols]
        
        st.dataframe(df_val, hide_index=True, use_container_width=True, column_config={
            "Value Score": st.column_config.ProgressColumn("Value (Points per £)", format="%.2f", min_value=0, max_value=max(df_val['Value Score'])),
            "Score": st.column_config.NumberColumn("Pred. Pts", format="%.1f"),
            "Price": st.column_config.NumberColumn("£", format="£%.1f"),
            "Next 10": st.column_config.TextColumn("Schedule", width="large")
        })

if __name__ == "__main__":
    main()
