import streamlit as st
import requests
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="FPL 10-GW Predictor", page_icon="🔮", layout="wide")

# --- 1. OPTIMIZED DATA LOADER ---
@st.cache_data(ttl=600)
def load_data():
    base_url = "https://fantasy.premierleague.com/api/"
    
    # Fetch 1: General Data (Players, Teams, Events)
    static = requests.get(base_url + "bootstrap-static/").json()
    
    # Fetch 2: Master Fixture List (All 380 matches)
    # We fetch this ONCE to avoid calling the API 600 times for individual players
    fixtures = requests.get(base_url + "fixtures/").json()
    
    return static, fixtures

# --- 2. LOGIC: PROCESS FIXTURES ---
def process_team_fixtures(static_data, fixture_data):
    """
    Creates a dictionary mapping every Team ID to their next 10 matches.
    Returns: { team_id: [{'opponent': 'ARS', 'diff': 4, 'home': True}, ...] }
    """
    teams = {t['id']: {'name': t['short_name'], 'strength': t['strength']} for t in static_data['teams']}
    
    # Find current Gameweek
    current_gw = next((e['id'] for e in static_data['events'] if e['is_current']), 1)
    next_gw = current_gw + 1
    
    # Initialize dict
    team_schedule = {t_id: [] for t_id in teams}
    
    # Loop through ALL fixtures and assign to teams
    for f in fixture_data:
        if f['event'] and f['event'] >= next_gw: # Only future games
            
            # Home Team's perspective
            h_team = f['team_h']
            a_team = f['team_a']
            h_diff = f['team_h_difficulty']
            a_diff = f['team_a_difficulty']
            
            # Add to Home Team Schedule
            if len(team_schedule[h_team]) < 10:
                team_schedule[h_team].append({
                    "opp": teams[a_team]['name'],
                    "diff": h_diff, # Difficulty for the home team
                    "loc": "(H)"
                })
                
            # Add to Away Team Schedule
            if len(team_schedule[a_team]) < 10:
                team_schedule[a_team].append({
                    "opp": teams[h_team]['name'],
                    "diff": a_diff, # Difficulty for the away team
                    "loc": "(A)"
                })
                
    return team_schedule

# --- 3. UI & CALCULATION ---
def main():
    st.title("🔮 FPL 10-Week Horizon Predictor")
    st.markdown("""
    This model analyzes the **next 10 Gameweeks** to find players with the best long-term potential.
    It balances **Player Stats (xGI)** with **Long-Term Schedule Difficulty**.
    """)

    # Load Data
    with st.spinner("Downloading full season schedule..."):
        static, fixtures = load_data()
        
    if not static or not fixtures:
        st.error("API Connection Failed")
        return

    # Process Schedule
    team_schedule = process_team_fixtures(static, fixtures)
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("⚙️ Prediction Model")
    
    # Weights
    w_form = st.sidebar.slider("Weight: Current Form", 0.0, 1.0, 0.3)
    w_xgi = st.sidebar.slider("Weight: Underlying Stats (xGI)", 0.0, 1.0, 0.5)
    w_fix = st.sidebar.slider("Weight: 10-Game Fixture Ease", 0.0, 1.0, 0.2)
    
    st.sidebar.divider()
    st.sidebar.header("🔎 Filters")
    min_mins = st.sidebar.number_input("Min Minutes Played", 0, 3000, 500)
    price_range = st.sidebar.slider("Price Range (£m)", 4.0, 15.0, (5.0, 13.0))
    pos_filter = st.sidebar.multiselect("Position", ["GKP", "DEF", "MID", "FWD"], ["MID", "FWD"])

    # --- 4. ANALYSIS ALGORITHM ---
    candidates = []
    
    for p in static['elements']:
        # 1. Basic Filters
        p_type_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
        pos = p_type_map[p['element_type']]
        
        if pos not in pos_filter: continue
        if p['minutes'] < min_mins: continue
        
        price = p['now_cost'] / 10.0
        if not (price_range[0] <= price <= price_range[1]): continue

        # 2. Get Stats
        try:
            xgi = float(p.get('expected_goal_involvements_per_90', 0))
            form = float(p['form'])
            ict = float(p['ict_index_rank']) if p['ict_index_rank'] else 0 # Lower is better usually, but we use raw ICT
        except:
            continue

        # 3. Get Next 10 Fixtures
        t_id = p['team']
        my_fixtures = team_schedule.get(t_id, [])
        
        if not my_fixtures: continue
        
        # Calculate Average Difficulty (Lower is better)
        total_diff = sum(f['diff'] for f in my_fixtures)
        avg_diff = total_diff / len(my_fixtures) if len(my_fixtures) > 0 else 5
        
        # Create Visual String for Table (e.g. "ARS(4), LIV(5)")
        # We only show first 5 in text to save space, user sees avg for 10
        fixture_str = ", ".join([f"{m['opp']}({m['diff']})" for m in my_fixtures[:5]]) + "..."

        # 4. THE PREDICTION FORMULA
        # We normalize stats to roughly 0-10 scale
        
        stat_score = (xgi * 10) # xGI of 0.8 -> 8.0
        form_score = form       # Form of 6.0 -> 6.0
        
        # Fixture Score: Invert difficulty. 
        # Avg Diff 2 (Easy) -> Score 8. Avg Diff 4 (Hard) -> Score 2.
        fixture_score = (5 - avg_diff) * 2.5 
        
        # Apply Weights
        final_score = (stat_score * w_xgi) + (form_score * w_form) + (fixture_score * w_fix)
        
        # Defensive Bonus: If Defender, add Clean Sheet potential based on fixtures
        if pos in ["DEF", "GKP"]:
            # Defenders rely more on fixtures than xGI
            final_score = final_score * 0.5 + (fixture_score * 0.8)

        candidates.append({
            "Player": p['web_name'],
            "Team": static['teams'][p['team']-1]['short_name'],
            "Pos": pos,
            "Price": price,
            "Form": form,
            "xGI/90": xgi,
            "10-Game Avg Diff": round(avg_diff, 2),
            "Next 5 Opponents (Diff)": fixture_str,
            "Explosion Prob": round(final_score, 2)
        })

    # --- 5. DISPLAY RESULTS ---
    if candidates:
        df = pd.DataFrame(candidates)
        df = df.sort_values(by="Explosion Prob", ascending=False).head(25)
        
        # Add a visual indicator for difficulty
        st.success(f"Analysis Complete. Showing top {len(df)} players for the next 10 weeks.")
        
        st.dataframe(
            df,
            column_config={
                "Explosion Prob": st.column_config.ProgressColumn(
                    "Explosion Probability",
                    format="%.2f",
                    min_value=0,
                    max_value=max(df["Explosion Prob"]),
                ),
                "Price": st.column_config.NumberColumn("£ Price", format="£%.1f"),
                "10-Game Avg Diff": st.column_config.NumberColumn(
                    "Fixture Difficulty (10 Gms)",
                    help="Average difficulty rating (1=Easy, 5=Hard) over next 10 games.",
                    format="%.2f"
                ),
                "Next 5 Opponents (Diff)": st.column_config.TextColumn(
                    "Upcoming Schedule",
                    width="large"
                )
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("No players matched your filters.")

if __name__ == "__main__":
    main()
