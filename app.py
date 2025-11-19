import streamlit as st
import requests
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="FPL Pro Model v6 (PPM Integrated)", page_icon="⚽", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .stDataFrame { width: 100%; }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #37A158, #E8E135, #C43939); }
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

def get_10_game_schedule(static_data, fixture_data):
    """Maps Team ID to list of next 10 opponents."""
    teams = {t['id']: t['short_name'] for t in static_data['teams']}
    next_gw = next((e['id'] for e in static_data['events'] if e['is_next']), 1)
    schedule = {t_id: [] for t_id in teams}
    for f in fixture_data:
        if f['event'] and f['event'] >= next_gw:
            h, a = f['team_h'], f['team_a']
            if len(schedule[h]) < 10: schedule[h].append({"opp": teams[a], "diff": f['team_h_difficulty']})
            if len(schedule[a]) < 10: schedule[a].append({"opp": teams[h], "diff": f['team_a_difficulty']})
    return schedule

# --- 2. SCORING ENGINES (UPDATED WITH PPM) ---

def calc_attacker_score(p, schedule_map, w_ppm, w_xgi, w_form, w_fix):
    """
    New Attacker Model.
    Key Input: PPM (Points Per Match).
    """
    try:
        # Metrics
        ppm = float(p['points_per_game'])
        xgi = float(p.get('expected_goal_involvements', 0))
        form = float(p['form'])
        
        # Schedule
        my_fixtures = schedule_map.get(p['team'], [])
        if not my_fixtures: return 0, "", 0, 0
        avg_diff = sum(m['diff'] for m in my_fixtures) / len(my_fixtures)
        
        # --- SCORING MATH (Normalized to ~10.0 scale) ---
        
        # 1. PPM Score: A PPM of 9.0 is elite. We multiply by 1.2 to get ~10.8
        s_ppm = ppm * 1.5
        
        # 2. xGI Score: Total xGI of 15 is elite. We multiply by 0.7 to get ~10.5
        s_xgi = xgi * 0.7
        
        # 3. Fixture Score: (5 - Diff) * 3. 
        # Easy schedule (Diff 2) -> Score 9.0
        s_fix = (5.0 - avg_diff) * 3.0
        
        # 4. Form Score: Directly use form (usually 0-10)
        s_form = form

        # Weighted Sum
        total = (s_ppm * w_ppm) + (s_xgi * w_xgi) + (s_form * w_form) + (s_fix * w_fix)
        
        sched_str = ", ".join([f"{m['opp']}({m['diff']})" for m in my_fixtures])
        return total, sched_str, avg_diff, ppm
    except: return 0, "", 0, 0

def calc_defender_score(p, schedule_map, leakiness_map, w_ppm, w_cs, w_fix, w_att):
    """
    New Defender Model.
    Key Input: PPM (Captures bonus points and reliability).
    """
    try:
        ppm = float(p['points_per_game'])
        xgi = float(p.get('expected_goal_involvements', 0))
        team_xgc = leakiness_map.get(p['team'], 1.5)
        
        my_fixtures = schedule_map.get(p['team'], [])
        if not my_fixtures: return 0, "", 0, 0
        avg_diff = sum(m['diff'] for m in my_fixtures) / len(my_fixtures)
        
        # --- SCORING MATH ---
        
        # 1. PPM Score (Crucial for Trent/Gabriel/Saliba)
        s_ppm = ppm * 1.5 
        
        # 2. Clean Sheet Prob (Inverted Team xGC)
        s_cs = (3.0 - team_xgc) * 4.0
        
        # 3. Fixtures (High importance for CS)
        s_fix = (5.0 - avg_diff) * 3.5
        
        # 4. Attacking Bonus
        s_att = xgi * 2.0
        
        total = (s_ppm * w_ppm) + (s_cs * w_cs) + (s_fix * w_fix) + (s_att * w_att)
        
        sched_str = ", ".join([f"{m['opp']}({m['diff']})" for m in my_fixtures])
        return total, sched_str, avg_diff, ppm
    except: return 0, "", 0, 0

# --- 3. MAIN APP ---
def main():
    st.title("⚽ FPL Pro Model v6 (PPM Integrated)")
    
    with st.spinner("Fetching live season data..."):
        static, fixtures = load_data()
        if not static: st.error("API Error"); return
        schedules = get_10_game_schedule(static, fixtures)
        leakiness = get_team_leakiness(static)
        teams = {t['id']: t['short_name'] for t in static['teams']}

    # --- SIDEBAR SETTINGS ---
    st.sidebar.header("⚙️ Model Calibration")
    
    with st.sidebar.expander("⚔️ Attacker Weights", expanded=True):
        aw_ppm = st.slider("Points Per Match", 0.1, 1.0, 0.8, help="Impact of historical consistency (Haaland factor)")
        aw_xgi = st.slider("Total xGI", 0.1, 1.0, 0.5, help="Impact of underlying stats")
        aw_fix = st.slider("Fixture Ease (10 Gms)", 0.1, 1.0, 0.3)
        aw_form = st.slider("Current Form", 0.1, 1.0, 0.2)
    
    with st.sidebar.expander("🛡️ Defender Weights", expanded=False):
        dw_ppm = st.slider("Points Per Match", 0.1, 1.0, 0.7, key="dw1")
        dw_cs = st.slider("Team CS Potential", 0.1, 1.0, 0.5, key="dw2")
        dw_fix = st.slider("Fixture Ease", 0.1, 1.0, 0.4, key="dw3")
        dw_att = st.slider("Attacking Threat", 0.1, 1.0, 0.2, key="dw4")
        
    min_mins = st.sidebar.number_input("Min Minutes Played", 0, 3000, 500)

    # --- DATA PROCESSING LOOP ---
    all_players = []
    
    for p in static['elements']:
        if p['minutes'] < min_mins: continue
        
        price = p['now_cost'] / 10.0
        pos = p['element_type']
        
        score = 0
        sched = ""
        avg_diff = 0
        ppm = 0
        category = ""
        
        # Attackers
        if pos in [3, 4]:
            score, sched, avg_diff, ppm = calc_attacker_score(p, schedules, aw_ppm, aw_xgi, aw_form, aw_fix)
            category = "Attack"
            
        # Defenders
        elif pos in [1, 2]:
            score, sched, avg_diff, ppm = calc_defender_score(p, schedules, leakiness, dw_ppm, dw_cs, dw_fix, dw_att)
            category = "Defense"
            
        if score > 0:
            all_players.append({
                "Name": p['web_name'],
                "Team": teams[p['team']],
                "Pos": {1:"GKP", 2:"DEF", 3:"MID", 4:"FWD"}[pos],
                "Category": category,
                "Price": price,
                "PPM": ppm,
                "Exp. Points": score, # The "Explosion Index"
                "Schedule": sched
            })

    df_all = pd.DataFrame(all_players)

    # --- UI TABS ---
    tab_att, tab_def, tab_val = st.tabs(["⚔️ Attackers", "🛡️ Defenders", "💎 Value & ROI Engine"])

    # TAB 1: ATTACKERS
    with tab_att:
        df_att = df_all[df_all['Category'] == "Attack"].copy()
        df_att = df_att.sort_values("Exp. Points", ascending=False).head(50)
        st.dataframe(df_att, hide_index=True, use_container_width=True, column_config={
            "Exp. Points": st.column_config.ProgressColumn("Predicted Points", format="%.1f", min_value=0, max_value=max(df_att['Exp. Points'])),
            "PPM": st.column_config.NumberColumn("Pts/Match", format="%.1f"),
            "Price": st.column_config.NumberColumn("£", format="£%.1f"),
            "Schedule": st.column_config.TextColumn("Next 10 Fixtures", width="large")
        })

    # TAB 2: DEFENDERS
    with tab_def:
        df_def = df_all[df_all['Category'] == "Defense"].copy()
        df_def = df_def.sort_values("Exp. Points", ascending=False).head(50)
        st.dataframe(df_def, hide_index=True, use_container_width=True, column_config={
            "Exp. Points": st.column_config.ProgressColumn("Predicted Points", format="%.1f", min_value=0, max_value=max(df_def['Exp. Points'])),
            "PPM": st.column_config.NumberColumn("Pts/Match", format="%.1f"),
            "Price": st.column_config.NumberColumn("£", format="£%.1f"),
            "Schedule": st.column_config.TextColumn("Next 10 Fixtures", width="large")
        })

    # TAB 3: VALUE ENGINE
    with tab_val:
        st.markdown("### ⚖️ Weighted Decision Engine")
        st.info("""
        **Instructions:**
        - **Points Priority = 100:** Ranking ignores price. Top players (Haaland/Salah) will be #1.
        - **Points Priority = 50:** Finds the "Best Value" (ROI) players.
        - **Points Priority = 0:** Finds the cheapest playable assets.
        """)
        
        c1, c2 = st.columns(2)
        w_pts = c1.slider("Weight: Expected Points", 0, 100, 80, help="Set to 100 to see the absolute best players regardless of cost.")
        w_price = c2.slider("Weight: Low Price", 0, 100, 20, help="Set higher to find budget gems.")

        # --- NORMALIZATION LOGIC ---
        df_val = df_all.copy()
        
        # Normalize Points (0 to 1)
        max_pts = df_val['Exp. Points'].max()
        df_val['n_pts'] = df_val['Exp. Points'] / max_pts
        
        # Normalize Price (0 to 1) - INVERTED (Low price = 1)
        min_price = df_val['Price'].min()
        max_price = df_val['Price'].max()
        df_val['n_price'] = 1 - ((df_val['Price'] - min_price) / (max_price - min_price))
        
        # Final Score
        # If w_pts is 100, we effectively ignore the price column
        df_val['Value Index'] = (df_val['n_pts'] * w_pts) + (df_val['n_price'] * w_price)
        
        df_val = df_val.sort_values("Value Index", ascending=False).head(50)
        
        # Clean View
        cols = ["Name", "Team", "Pos", "Price", "PPM", "Value Index", "Exp. Points", "Schedule"]
        st.dataframe(df_val[cols], hide_index=True, use_container_width=True, column_config={
            "Value Index": st.column_config.ProgressColumn("Value Score", format="%.0f", min_value=0, max_value=max(df_val['Value Index'])),
            "Exp. Points": st.column_config.NumberColumn("Pred. Points", format="%.1f"),
            "PPM": st.column_config.NumberColumn("Pts/Match", format="%.1f"),
            "Price": st.column_config.NumberColumn("£", format="£%.1f"),
            "Schedule": st.column_config.TextColumn("Next 10 Opponents", width="large")
        })

if __name__ == "__main__":
    main()
