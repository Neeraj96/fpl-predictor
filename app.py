import streamlit as st
import requests
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="FPL Schedule-Adjusted Model v8", page_icon="⚖️", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .stDataFrame { width: 100%; }
    .stProgress > div > div > div > div { 
        background-image: linear-gradient(to right, #a83232, #d19630, #37a849); 
    }
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
    """Calculates Team xGC per 90."""
    team_xgc = {t['id']: [] for t in static_data['teams']}
    for p in static_data['elements']:
        if p['element_type'] in [1, 2] and p['minutes'] > 450:
            try:
                xgc = float(p.get('expected_goals_conceded_per_90', 0))
                team_xgc[p['team']].append(xgc)
            except: continue
    
    team_strength = {}
    for t_id, values in team_xgc.items():
        team_strength[t_id] = sum(values) / len(values) if values else 1.5
    return team_strength

def get_schedule_analytics(static_data, fixture_data):
    """
    Analyzes BOTH Past and Future schedules.
    Returns:
    1. future_schedule: {team_id: [list of next 10]}
    2. past_difficulty: {team_id: average_difficulty_of_played_games}
    """
    teams = {t['id']: t['short_name'] for t in static_data['teams']}
    next_gw = next((e['id'] for e in static_data['events'] if e['is_next']), 1)
    
    future_schedule = {t_id: [] for t_id in teams}
    past_difficulty_sum = {t_id: [] for t_id in teams}
    
    for f in fixture_data:
        h, a = f['team_h'], f['team_a']
        h_diff, a_diff = f['team_h_difficulty'], f['team_a_difficulty']

        # PAST GAMES (Finished)
        if f['finished'] == True:
            past_difficulty_sum[h].append(h_diff)
            past_difficulty_sum[a].append(a_diff)
            
        # FUTURE GAMES (Next 10)
        if f['event'] and f['event'] >= next_gw:
            if len(future_schedule[h]) < 10: future_schedule[h].append({"opp": teams[a], "diff": h_diff})
            if len(future_schedule[a]) < 10: future_schedule[a].append({"opp": teams[h], "diff": a_diff})
            
    # Calculate Average Past Difficulty
    past_avg = {}
    for t_id, diffs in past_difficulty_sum.items():
        past_avg[t_id] = sum(diffs) / len(diffs) if diffs else 3.0
        
    return future_schedule, past_avg

# --- 2. SCORING ENGINES (ADJUSTED FOR PAST DIFFICULTY) ---

def calc_attacker_score(p, future_map, past_avg_map, w_ppm, w_xgi, w_form, w_fix):
    try:
        # 1. Base Stats
        ppm = float(p['points_per_game'])
        xgi = float(p.get('expected_goal_involvements', 0))
        form = float(p['form'])
        
        # 2. Schedule Analytics
        my_future = future_map.get(p['team'], [])
        past_diff = past_avg_map.get(p['team'], 3.0)
        
        if not my_future: return 0, "", 0, 0
        future_avg_diff = sum(m['diff'] for m in my_future) / len(my_future)
        
        # 3. THE ADJUSTMENT LOGIC (Crucial Step)
        # If past schedule was Hard (>3.0), we BOOST the historical stats.
        # If past schedule was Easy (<3.0), we DAMPEN the historical stats.
        # We use a gentler multiplier: (Past / 3.0)
        hardship_multiplier = past_diff / 3.0
        
        adj_ppm = ppm * hardship_multiplier
        adj_xgi = xgi * hardship_multiplier
        adj_form = form * hardship_multiplier
        
        # 4. Scoring (Using Adjusted Stats)
        s_ppm = adj_ppm * 2.0          
        s_xgi = adj_xgi * 0.8          
        s_form = adj_form * 1.0        
        
        # Fixture score is forward-looking, so it uses future difficulty
        s_fix = (5.0 - future_avg_diff) * 3.0 
        
        raw_score = (s_ppm * w_ppm) + (s_xgi * w_xgi) + (s_form * w_form) + (s_fix * w_fix)
        
        sched_str = ", ".join([f"{m['opp']}({m['diff']})" for m in my_future])
        return raw_score, sched_str, future_avg_diff, past_diff
    except: return 0, "", 0, 0

def calc_defender_score(p, future_map, past_avg_map, leakiness_map, w_ppm, w_cs, w_fix, w_att):
    try:
        ppm = float(p['points_per_game'])
        xgi = float(p.get('expected_goal_involvements', 0))
        team_xgc = leakiness_map.get(p['team'], 1.5)
        
        my_future = future_map.get(p['team'], [])
        past_diff = past_avg_map.get(p['team'], 3.0)
        
        if not my_future: return 0, "", 0, 0
        future_avg_diff = sum(m['diff'] for m in my_future) / len(my_future)
        
        # Adjustment Logic
        hardship_multiplier = past_diff / 3.0
        
        adj_ppm = ppm * hardship_multiplier
        adj_xgi = xgi * hardship_multiplier # Attacking return usually independent of team, but harder vs hard teams
        
        # Scoring
        s_ppm = adj_ppm * 2.0 
        s_cs = (3.0 - team_xgc) * 5.5 # Equivalence factor
        s_fix = (5.0 - future_avg_diff) * 3.5 
        s_att = adj_xgi * 2.5 
        
        raw_score = (s_ppm * w_ppm) + (s_cs * w_cs) + (s_fix * w_fix) + (s_att * w_att)
        
        sched_str = ", ".join([f"{m['opp']}({m['diff']})" for m in my_future])
        return raw_score, sched_str, future_avg_diff, past_diff
    except: return 0, "", 0, 0

# --- 3. MAIN APP ---
def main():
    st.title("⚖️ FPL Schedule-Adjusted Model v8")
    st.markdown("""
    **Logic:** If a player has high stats despite a **Hard Past Schedule**, their ROI Index is boosted.
    *(Hardship Multiplier = Past Avg Difficulty / 3.0)*
    """)
    
    with st.spinner("Analyzing Past & Future Schedules..."):
        static, fixtures = load_data()
        if not static: st.error("API Error"); return
        
        # Get both schedules
        future_map, past_map = get_schedule_analytics(static, fixtures)
        leakiness = get_team_leakiness(static)
        teams = {t['id']: t['short_name'] for t in static['teams']}

    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Model Calibration")
    with st.sidebar.expander("Adjust Weights", expanded=False):
        st.write("**Attackers**")
        aw_ppm = st.slider("PPM", 0.1, 1.0, 0.9)
        aw_xgi = st.slider("xGI", 0.1, 1.0, 0.7)
        aw_fix = st.slider("Future Fixtures", 0.1, 1.0, 0.4)
        aw_form = st.slider("Form", 0.1, 1.0, 0.3)
        st.write("**Defenders**")
        dw_ppm = st.slider("PPM ", 0.1, 1.0, 0.8)
        dw_cs = st.slider("Clean Sheet", 0.1, 1.0, 0.7)
        dw_fix = st.slider("Future Fixtures ", 0.1, 1.0, 0.4)
    
    min_mins = st.sidebar.number_input("Min Minutes", 0, 3000, 500)

    # --- CALCULATION LOOP ---
    all_players = []
    
    for p in static['elements']:
        if p['minutes'] < min_mins: continue
        
        price = p['now_cost'] / 10.0
        pos = p['element_type']
        raw_score = 0
        sched = ""
        past_diff = 3.0
        category = ""
        
        if pos in [3, 4]: # ATTACK
            raw_score, sched, f_diff, past_diff = calc_attacker_score(
                p, future_map, past_map, aw_ppm, aw_xgi, aw_form, aw_fix
            )
            category = "Attack"
        elif pos in [1, 2]: # DEFENSE
            raw_score, sched, f_diff, past_diff = calc_defender_score(
                p, future_map, past_map, leakiness, dw_ppm, dw_cs, dw_fix, 0.3
            )
            category = "Defense"
            
        if raw_score > 0:
            all_players.append({
                "Name": p['web_name'],
                "Team": teams[p['team']],
                "Pos": {1:"GKP", 2:"DEF", 3:"MID", 4:"FWD"}[pos],
                "Category": category,
                "Price": price,
                "PPM": float(p['points_per_game']),
                "Past Diff": past_diff,  # NEW METRIC
                "Raw Score": raw_score,
                "Schedule": sched
            })

    df_all = pd.DataFrame(all_players)

    # NORMALIZE TO 1-10 SCALE
    if not df_all.empty:
        max_raw = df_all['Raw Score'].max()
        df_all['ROI Index'] = 1.0 + ((df_all['Raw Score'] / max_raw) * 9.0)
    else: return

    # --- DISPLAY ---
    tab_exp, tab_att, tab_def, tab_val = st.tabs([
        "💥 Explosion Potential (Adjusted)", 
        "⚔️ Attackers", "🛡️ Defenders", "💎 Value Engine"
    ])

    # Helper function for display config
    def get_config():
        return {
            "ROI Index": st.column_config.ProgressColumn("Exp. ROI Index (1-10)", format="%.1f", min_value=1, max_value=10),
            "Past Diff": st.column_config.NumberColumn(
                "Past Schedule", 
                format="%.2f", 
                help="Avg difficulty of games played so far. Higher (>3.0) means they survived a hard run and Stats are boosted."
            ),
            "PPM": st.column_config.NumberColumn("PPM", format="%.1f"),
            "Price": st.column_config.NumberColumn("£", format="£%.1f"),
            "Schedule": st.column_config.TextColumn("Next 10 Fixtures", width="large")
        }

    with tab_exp:
        st.markdown("### 💥 Schedule-Adjusted Leaderboard")
        st.info("Players with **High 'Past Schedule' (>3.0)** scores have their ROI Index boosted because they performed well against tough teams.")
        
        df_show = df_all.sort_values("ROI Index", ascending=False).head(50)
        st.dataframe(df_show, hide_index=True, use_container_width=True, column_config=get_config())

    with tab_att:
        df_show = df_all[df_all['Category']=="Attack"].sort_values("ROI Index", ascending=False).head(50)
        st.dataframe(df_show, hide_index=True, use_container_width=True, column_config=get_config())

    with tab_def:
        df_show = df_all[df_all['Category']=="Defense"].sort_values("ROI Index", ascending=False).head(50)
        st.dataframe(df_show, hide_index=True, use_container_width=True, column_config=get_config())
        
    with tab_val:
        c1, c2 = st.columns(2)
        w_roi = c1.slider("Importance: ROI Index", 0, 100, 70)
        w_cost = c2.slider("Importance: Low Price", 0, 100, 30)
        
        df_val = df_all.copy()
        df_val['n_roi'] = (df_val['ROI Index'] - 1) / 9
        min_p, max_p = df_val['Price'].min(), df_val['Price'].max()
        df_val['n_price'] = 1 - ((df_val['Price'] - min_p) / (max_p - min_p))
        df_val['Val Score'] = (df_val['n_roi'] * w_roi) + (df_val['n_price'] * w_cost)
        
        df_val = df_val.sort_values("Val Score", ascending=False).head(50)
        
        st.dataframe(df_val, hide_index=True, use_container_width=True, column_config={
            "Val Score": st.column_config.ProgressColumn("Algorithm Score", format="%.0f"),
            "ROI Index": st.column_config.NumberColumn("ROI Index", format="%.1f"),
            "Past Diff": st.column_config.NumberColumn("Past Diff", format="%.2f"),
            "Price": st.column_config.NumberColumn("£", format="£%.1f"),
            "Schedule": st.column_config.TextColumn("Next 10", width="large")
        })

if __name__ == "__main__":
    main()
