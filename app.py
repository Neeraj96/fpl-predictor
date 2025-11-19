import streamlit as st
import requests
import pandas as pd
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="FPL Explosion Predictor", page_icon="⚽", layout="wide")

# --- TITLE & HEADER ---
st.title("⚽ FPL Explosion Predictor (2025/26)")
st.markdown("""
This app identifies players who are **statistically due** for a big haul.
It looks for high **Expected Goal Involvement (xGI)** and **Form**, combined with **Easy Upcoming Fixtures**.
""")

# --- SIDEBAR FILTERS ---
st.sidebar.header("⚙️ Prediction Filters")
min_minutes = st.sidebar.slider("Min Minutes Played", 0, 1000, 500)
min_xgi = st.sidebar.slider("Min xGI per 90", 0.0, 1.0, 0.40)
min_form = st.sidebar.slider("Min Form", 0.0, 10.0, 3.5)
max_difficulty = st.sidebar.slider("Max Next Match Difficulty (1-5)", 1, 5, 3)

# --- FUNCTION TO FETCH DATA ---
@st.cache_data(ttl=300)  # Cache data for 5 mins to speed up app
def fetch_fpl_data():
    base_url = "https://fantasy.premierleague.com/api/"
    static = requests.get(base_url + "bootstrap-static/").json()
    return static

# --- MAIN APP LOGIC ---
if st.button("🚀 Find Explosive Players"):
    with st.spinner("Scouting the market..."):
        try:
            data = fetch_fpl_data()
            teams = {t['id']: t['name'] for t in data['teams']}
            
            candidates = []
            base_url = "https://fantasy.premierleague.com/api/"
            
            # Filter active players first
            active_players = [
                p for p in data['elements'] 
                if p['minutes'] >= min_minutes and p['element_type'] in [3, 4] # Mids & fwds
            ]

            # Progress bar
            progress_bar = st.progress(0)
            total_checked = len(active_players)
            
            for i, p in enumerate(active_players):
                # Update progress bar every 10 players
                if i % 10 == 0:
                    progress_bar.progress((i + 1) / total_checked)

                try:
                    xgi = float(p.get('expected_goal_involvements_per_90', 0))
                    form = float(p['form'])
                    
                    # CHECK 1: Do they meet the stat criteria?
                    if xgi >= min_xgi and form >= min_form:
                        
                        # CHECK 2: Check Fixture Difficulty
                        # We fetch this only for players who pass the first check to save time
                        p_id = p['id']
                        p_summary = requests.get(base_url + f"element-summary/{p_id}/").json()
                        fixtures = p_summary.get('fixtures', [])
                        
                        if fixtures:
                            next_match = fixtures[0]
                            diff = next_match['difficulty']
                            
                            if diff <= max_difficulty:
                                # Get Opponent Name
                                is_home = next_match['is_home']
                                opp_id = next_match['team_a'] if is_home else next_match['team_h']
                                opp_name = teams.get(opp_id, "Unknown")
                                venue = "Home" if is_home else "Away"
                                
                                candidates.append({
                                    "Player": p['web_name'],
                                    "Team": teams[p['team']],
                                    "Price": f"£{p['now_cost']/10}m",
                                    "Pos": "MID" if p['element_type'] == 3 else "FWD",
                                    "xGI/90": xgi,
                                    "Form": form,
                                    "Next Opponent": f"{opp_name} ({venue})",
                                    "Difficulty": diff
                                })
                                
                        # Polite sleep
                        time.sleep(0.01)
                        
                except Exception as e:
                    continue
            
            progress_bar.empty() # Remove bar when done

            # --- DISPLAY RESULTS ---
            if candidates:
                df = pd.DataFrame(candidates)
                # Sort by xGI
                df = df.sort_values(by="xGI/90", ascending=False)
                
                st.success(f"Found {len(df)} potential differentials!")
                st.dataframe(
                    df,
                    column_config={
                        "xGI/90": st.column_config.NumberColumn("xGI (Expected)", format="%.2f"),
                        "Difficulty": st.column_config.NumberColumn("Diff", help="1=Easy, 5=Hard"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("No players found matching these strict filters. Try lowering the xGI or Form threshold in the sidebar.")
                
        except Exception as e:
            st.error(f"Error fetching data: {e}")