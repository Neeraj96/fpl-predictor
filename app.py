import streamlit as st
import requests
import pandas as pd

# --- CONFIGURATION ---
st.set_page_config(page_title="FPL Pro Predictor 25/26", page_icon="⚽", layout="wide")

# --- CONSTANTS ---
API_BASE = "https://fantasy.premierleague.com/api"

# --- CACHED DATA LOADER ---
@st.cache_data(ttl=600)
def load_data():
    # 1. Fetch Bootstrap Static (Players, Teams, Events)
    try:
        bootstrap = requests.get(f"{API_BASE}/bootstrap-static/").json()
    except:
        st.error("API Error: Could not fetch static data.")
        return None, None

    # 2. Determine Next Gameweek
    next_gw = None
    for event in bootstrap['events']:
        if event['is_next']:
            next_gw = event['id']
            break
    
    if not next_gw:
        st.warning("Season finished or no next gameweek found.")
        return bootstrap, {}

    # 3. Fetch Fixtures for Next Gameweek
    try:
        fixtures = requests.get(f"{API_BASE}/fixtures/?event={next_gw}").json()
    except:
        st.error("API Error: Could not fetch fixtures.")
        return bootstrap, {}

    # 4. Map Team ID to Next Opponent ID & Difficulty
    # Structure: { team_id: { 'opponent': opp_id, 'difficulty': diff_rating, 'is_home': bool } }
    fixture_map = {}
    for f in fixtures:
        # Home Team's data
        fixture_map[f['team_h']] = {
            'opponent': f['team_a'],
            'difficulty': f['team_h_difficulty'],
            'is_home': True
        }
        # Away Team's data
        fixture_map[f['team_a']] = {
            'opponent': f['team_h'],
            'difficulty': f['team_a_difficulty'],
            'is_home': False
        }

    return bootstrap, fixture_map

# --- TEAM STATS CALCULATOR ---
def calculate_team_stats(data):
    """
    Returns two dictionaries:
    1. defensive_weakness: How likely a team is to concede (based on xGC).
    2. defensive_strength: How good a team is at defending (inverse of weakness).
    """
    team_xgc = {t['id']: 0.0 for t in data['teams']}
    
    # Sum xGC for Defenders (Type 2) and GKs (Type 1) to judge defensive leakiness
    for p in data['elements']:
        if p['element_type'] in [1, 2]:
            try:
                xgc = float(p.get('expected_goals_conceded', 0))
                team_xgc[p['team']] += xgc
            except:
                continue
                
    # Normalize to 0-10 Scale
    # High Score = Very Weak Defense (Good to attack against)
    # Low Score = Strong Defense
    max_val = max(team_xgc.values()) if team_xgc else 1
    weakness_map = {k: (v / max_val) * 10.0 for k, v in team_xgc.items()}
    
    # Create Strength Map (Inverse of Weakness) for Clean Sheet prediction
    strength_map = {k: 10.0 - v for k, v in weakness_map.items()}

    return weakness_map, strength_map

# --- MAIN APP ---
def main():
    st.title("🧠 FPL AI Predictor (Enhanced Model)")
    st.markdown("""
    **Logic:** 
    1. **Form & Threat:** Uses `xGI` (Expected Goal Involvement) and recent `Form`.
    2. **Fixture Difficulty:** Analyzes the **actual upcoming opponent**.
    3. **Opponent Leakiness:** Checks how many Expected Goals (xGC) the opponent concedes.
    """)

    data, fixture_map = load_data()
    if not data:
        return

    # Mappings
    team_names = {t['id']: t['name'] for t in data['teams']}
    opp_weakness_map, team_strength_map = calculate_team_stats(data)

    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Prediction Weights")
    st.sidebar.info("Adjusted to equal weights by default.")
    
    # Defaulting to equal weights (0.5 each)
    w_form = st.sidebar.slider("Player Stats (Form/xGI)", 0.1, 1.0, 0.5)
    w_fix = st.sidebar.slider("Fixture Favorability", 0.1, 1.0, 0.5)
    
    st.sidebar.divider()
    st.sidebar.header("🔎 Player Filters")
    
    # Expanded positions to include Defenders
    pos_map = {"Goalkeeper": 1, "Defender": 2, "Midfielder": 3, "Forward": 4}
    position_filter = st.sidebar.multiselect(
        "Positions", 
        options=list(pos_map.keys()), 
        default=["Midfielder", "Forward", "Defender"]
    )
    
    min_price = st.sidebar.number_input("Min Price (£m)", 3.5, 15.0, 4.0)
    max_price = st.sidebar.number_input("Max Price (£m)", 4.0, 15.0, 15.0)

    # --- ANALYSIS ---
    if st.button("Run Prediction Model", type="primary"):
        if not position_filter:
            st.error("Please select at least one position.")
            return

        valid_types = [pos_map[p] for p in position_filter]
        candidates = []
        
        with st.spinner("Analyzing matchups against next opponents..."):
            for p in data['elements']:
                # 1. Basic Filters
                if p['status'] != 'a' or p['minutes'] < 90: # Must have played some minutes
                    continue
                
                price = p['now_cost'] / 10.0
                if not (min_price <= price <= max_price):
                    continue
                
                if p['element_type'] not in valid_types:
                    continue

                # 2. Get Upcoming Matchup
                tid = p['team']
                if tid not in fixture_map:
                    continue # Team has a blank gameweek
                
                match_info = fixture_map[tid]
                opp_id = match_info['opponent']
                is_home = match_info['is_home']
                
                # 3. Calculate Metrics
                try:
                    xgi = float(p.get('expected_goal_involvements_per_90', 0))
                    form = float(p['form'])
                except:
                    continue
                
                # SKIP if stats are negligible (Optimization)
                if xgi < 0.1 and form < 2.0:
                    continue

                # --- SCORING ALGORITHM ---
                
                # A. PLAYER STATS SCORE (0-10)
                # Normalizing xGI (Top players represent ~1.0 xGI/90) -> Scale to 10
                stat_score = (xgi * 8) + (form / 2) 
                
                # B. FIXTURE SCORE (0-10)
                # How weak is the opponent?
                opp_leakiness = opp_weakness_map.get(opp_id, 5.0)
                
                # Home advantage bonus (+1.5)
                home_boost = 1.5 if is_home else 0
                
                fixture_score = opp_leakiness + home_boost

                # C. DEFENDER SPECIFIC LOGIC
                # If Defender/GK, add Clean Sheet Potential (Their own team strength vs Opponent)
                if p['element_type'] in [1, 2]:
                    my_team_def = team_strength_map.get(tid, 5.0)
                    # Defenders rely 50% on CS and 50% on Attacking Return in this model
                    # We average the 'stat_score' (attack) with 'my_team_def'
                    stat_score = (stat_score * 0.6) + (my_team_def * 0.4)

                # D. FINAL WEIGHTED SCORE
                total_score = (stat_score * w_form) + (fixture_score * w_fix)
                
                candidates.append({
                    "Player": p['web_name'],
                    "Team": team_names[tid],
                    "Pos": [k for k, v in pos_map.items() if v == p['element_type']][0][:3].upper(),
                    "Next Opp": team_names[opp_id] + (" (H)" if is_home else " (A)"),
                    "Price": price,
                    "Form": form,
                    "xGI/90": xgi,
                    "Prediction": total_score
                })

        # Display Results
        df = pd.DataFrame(candidates)
        
        if not df.empty:
            df = df.sort_values(by="Prediction", ascending=False).head(25)
            
            st.success(f"Analysis Complete! Found {len(candidates)} eligible players.")
            
            st.dataframe(
                df,
                column_config={
                    "Prediction": st.column_config.ProgressColumn(
                        "Predicted Points Potential",
                        help="Weighted score based on Form, xGI, and Opponent Difficulty",
                        format="%.2f",
                        min_value=0,
                        max_value=max(df["Prediction"]),
                    ),
                    "Price": st.column_config.NumberColumn("Price", format="£%.1f"),
                    "xGI/90": st.column_config.NumberColumn("xGI/90", format="%.2f"),
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("No players found. Try lowering the minimum price or checking filters.")

if __name__ == "__main__":
    main()
