import streamlit as st
import requests
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="FPL 10-GW Ultimate Value", page_icon="💎", layout="wide")

# --- CSS FOR WIDER TABLES ---
st.markdown("""
<style>
    .stDataFrame { width: 100%; }
    div[data-testid="stExpander"] div[role="button"] p { font-size: 1.1rem; font-weight: bold; }
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
    """Calculates Team xGC (Expected Goals Conceded) per 90."""
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

# --- 2. CORE SCORING MODELS ---

def calc_attacker_score(p, schedule_map, w_xgi, w_form, w_fix):
    """Returns 'Explosion Index' for MIDs/FWDs"""
    try:
        xgi = float(p.get('expected_goal_involvements', 0))
        form = float(p['form'])
        my_fixtures = schedule_map.get(p['team'], [])
        if not my_fixtures: return 0, "", 0
        
        avg_diff = sum(m['diff'] for m in my_fixtures) / len(my_fixtures)
        
        # Scoring: xGI (Base ~15) + Fixtures (Inv Diff)
        s_xgi = xgi * 0.8
        s_fix = (5.0 - avg_diff) * 3.0
        
        total = (s_xgi * w_xgi) + (form * w_form) + (s_fix * w_fix)
        sched_str = ", ".join([f"{m['opp']}({m['diff']})" for m in my_fixtures])
        return total, sched_str, avg_diff
    except: return 0, "", 0

def calc_defender_score(p, schedule_map, leakiness_map, w_cs, w_fix, w_xgi):
    """Returns 'Clean Sheet Potential' for DEFs/GKPs"""
    try:
        xgi = float(p.get('expected_goal_involvements', 0))
        team_xgc = leakiness_map.get(p['team'], 1.5)
        my_fixtures = schedule_map.get(p['team'], [])
        if not my_fixtures: return 0, "", 0
        
        avg_diff = sum(m['diff'] for m in my_fixtures) / len(my_fixtures)
        
        # Scoring: Low Team xGC + Low Fixture Diff
        s_cs = (3.0 - team_xgc) * 5.0 
        s_fix = (5.0 - avg_diff) * 4.0
        s_att = xgi * 2.5
        
        total = (s_cs * w_cs) + (s_fix * w_fix) + (s_att * w_xgi)
        sched_str = ", ".join([f"{m['opp']}({m['diff']})" for m in my_fixtures])
        return total, sched_str, avg_diff
    except: return 0, "", 0

# --- 3. MAIN APP ---
def main():
    st.title("💎 FPL Ultimate Value Model")
    
    with st.spinner("Fetching live stats & schedules..."):
        static, fixtures = load_data()
        if not static:
            st.error("API Connection Error")
            return
        
        schedules = get_10_game_schedule(static, fixtures)
        leakiness = get_team_leakiness(static)
        teams = {t['id']: t['short_name'] for t in static['teams']}

    # --- GLOBAL SIDEBAR ---
    st.sidebar.header("⚙️ Global Settings")
    min_mins = st.sidebar.number_input("Min Minutes Played", 0, 3000, 500)

    # --- TABS ---
    tab_att, tab_def, tab_val = st.tabs(["⚔️ Attackers", "🛡️ Defenders", "💰 Best Value (Unified)"])

    # --- PROCESS DATA (ALL PLAYERS) ---
    # We calculate scores for everyone first, as the Value tab needs both lists.
    
    # ATTACKER WEIGHTS (Hidden in Expander to keep UI clean)
    with st.sidebar.expander("⚔️ Attacker Model Weights"):
        aw_xgi = st.slider("Total xGI", 0.1, 1.0, 0.6, key="aw1")
        aw_fix = st.slider("Fixture Ease", 0.1, 1.0, 0.3, key="aw2")
        aw_form = st.slider("Form", 0.1, 1.0, 0.2, key="aw3")

    # DEFENDER WEIGHTS
    with st.sidebar.expander("🛡️ Defender Model Weights"):
        dw_cs = st.slider("Clean Sheet Prob", 0.1, 1.0, 0.6, key="dw1")
        dw_fix = st.slider("Fixture Ease", 0.1, 1.0, 0.3, key="dw2")
        dw_att = st.slider("Attacking Threat", 0.1, 1.0, 0.2, key="dw3")

    all_players_data = []

    for p in static['elements']:
        if p['minutes'] < min_mins: continue
        
        price = p['now_cost'] / 10.0
        pos_type = p['element_type'] # 1=GKP, 2=DEF, 3=MID, 4=FWD
        
        # INIT VARIABLES
        exp_pts = 0
        sched = ""
        avg_diff = 0
        category = ""
        pos_label = ""

        # 1. ATTACKERS (MID/FWD)
        if pos_type in [3, 4]:
            exp_pts, sched, avg_diff = calc_attacker_score(p, schedules, aw_xgi, aw_form, aw_fix)
            category = "Attack"
            pos_label = "MID" if pos_type == 3 else "FWD"

        # 2. DEFENDERS (GKP/DEF)
        elif pos_type in [1, 2]:
            exp_pts, sched, avg_diff = calc_defender_score(p, schedules, leakiness, dw_cs, dw_fix, dw_att)
            category = "Defense"
            pos_label = "GKP" if pos_type == 1 else "DEF"

        # ADD TO MASTER LIST
        if exp_pts > 0:
            all_players_data.append({
                "Name": p['web_name'],
                "Team": teams[p['team']],
                "Pos": pos_label,
                "Category": category,
                "Price": price,
                "Exp. Points": exp_pts, # Unified Score
                "Avg Diff": avg_diff,
                "Schedule": sched
            })

    df_all = pd.DataFrame(all_players_data)

    # --- TAB 1: ATTACKERS ---
    with tab_att:
        df_att = df_all[df_all['Category'] == "Attack"].copy()
        df_att = df_att.sort_values("Exp. Points", ascending=False).head(50)
        
        st.dataframe(df_att, hide_index=True, use_container_width=True, column_config={
            "Exp. Points": st.column_config.ProgressColumn("Explosion Index", format="%.1f", min_value=0, max_value=max(df_att['Exp. Points'])),
            "Price": st.column_config.NumberColumn("£ Price", format="£%.1f"),
            "Schedule": st.column_config.TextColumn("Next 10 Fixtures", width="large")
        })

    # --- TAB 2: DEFENDERS ---
    with tab_def:
        df_def = df_all[df_all['Category'] == "Defense"].copy()
        df_def = df_def.sort_values("Exp. Points", ascending=False).head(50)
        
        st.dataframe(df_def, hide_index=True, use_container_width=True, column_config={
            "Exp. Points": st.column_config.ProgressColumn("CS Potential", format="%.1f", min_value=0, max_value=max(df_def['Exp. Points'])),
            "Price": st.column_config.NumberColumn("£ Price", format="£%.1f"),
            "Schedule": st.column_config.TextColumn("Next 10 Fixtures", width="large")
        })

    # --- TAB 3: UNIFIED VALUE (THE NEW LOGIC) ---
    with tab_val:
        st.markdown("### ⚖️ The ROI Engine")
        st.info("Adjust the sliders below to find your perfect transfer. Do you want raw points (Haaland) or budget gems (Semenyo)?")
        
        c1, c2 = st.columns(2)
        w_val_pts = c1.slider("Weight: Expected Points", 0, 100, 70, help="Higher = Prioritize best players (Haaland/Salah) regardless of price.")
        w_val_price = c2.slider("Weight: Low Price", 0, 100, 30, help="Higher = Prioritize cheapest players.")

        # NORMALIZE DATA FOR WEIGHTED CALCULATION
        df_val = df_all.copy()
        
        # 1. Get Min/Max for Normalization
        max_pts = df_val['Exp. Points'].max()
        min_price = df_val['Price'].min()
        max_price = df_val['Price'].max()
        
        # 2. Calculation Logic
        # Normalized Points (0 to 1)
        df_val['norm_pts'] = df_val['Exp. Points'] / max_pts
        
        # Normalized Price (0 to 1). NOTE: Inverted! 
        # Lowest Price (£4.0) gets score 1.0. Highest Price (£15.0) gets score 0.0.
        df_val['norm_price'] = 1 - ((df_val['Price'] - min_price) / (max_price - min_price))
        
        # 3. Final Weighted Value Score
        # If w_val_pts is 100, we only look at norm_pts.
        # If w_val_price is 100, we only look at norm_price.
        df_val['Value Index'] = (df_val['norm_pts'] * w_val_pts) + (df_val['norm_price'] * w_val_price)
        
        # Sort
        df_val = df_val.sort_values("Value Index", ascending=False).head(50)
        
        # Display
        cols = ["Name", "Team", "Pos", "Price", "Value Index", "Exp. Points", "Schedule"]
        st.dataframe(df_val[cols], hide_index=True, use_container_width=True, column_config={
            "Value Index": st.column_config.ProgressColumn(
                "Value Score", 
                help="Weighted combination of Price and Points",
                format="%.0f", 
                min_value=0, 
                max_value=max(df_val['Value Index'])
            ),
            "Exp. Points": st.column_config.NumberColumn("Pred. Points", format="%.1f"),
            "Price": st.column_config.NumberColumn("£ Price", format="£%.1f"),
            "Schedule": st.column_config.TextColumn("Next 10 Opponents", width="large")
        })

if __name__ == "__main__":
    main()
