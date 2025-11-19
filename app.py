import streamlit as st
import requests
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="FPL Pro Predictor 25/26", page_icon="⚽", layout="wide")
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 5px; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #e6ffe6; border: 1px solid #00cc00; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
API_BASE = "https://fantasy.premierleague.com/api"

# --- DATA LOADING ---
@st.cache_data(ttl=1800)
def load_data():
    # 1. Fetch Bootstrap Static
    try:
        bootstrap = requests.get(f"{API_BASE}/bootstrap-static/").json()
    except:
        st.error("API Error: Could not fetch static data.")
        return None, None

    # 2. Fetch All Fixtures
    try:
        fixtures = requests.get(f"{API_BASE}/fixtures/").json()
    except:
        st.error("API Error: Could not fetch fixtures.")
        return bootstrap, None

    return bootstrap, fixtures

# --- PRO FLUID LOGIC ENGINE ---
def process_fixture_difficulty(fixtures, teams_count):
    """
    Analyzes every fixture to create a map of Past and Future difficulties for every team.
    Returns: dict { team_id: { 'past_fav': [], 'future_fav': [] } }
    Favourability = 6 - Difficulty (Higher is better)
    """
    # Initialize
    team_sched = {i: {'past': [], 'future': []} for i in range(1, teams_count + 1)}

    for f in fixtures:
        if not f['finished'] and not f['kickoff_time']: continue # Skip TBC games

        h = f['team_h']
        a = f['team_a']
        h_diff = f['team_h_difficulty']
        a_diff = f['team_a_difficulty']

        # Calculate Favourability (6 - Difficulty). 
        # Home Advantage: Add 0.5 favorability for Home games.
        h_fav = (6 - h_diff) + 0.5
        a_fav = (6 - a_diff)

        if f['finished']:
            team_sched[h]['past'].append(h_fav)
            team_sched[a]['past'].append(a_fav)
        else:
            team_sched[h]['future'].append(h_fav)
            team_sched[a]['future'].append(a_fav)

    return team_sched

def get_avg_favourability(schedule_list, limit=None):
    if not schedule_list:
        return 3.0 # Default average
    subset = schedule_list[:limit] if limit else schedule_list
    return sum(subset) / len(subset)

def normalize_scores(df, target_col):
    """Scales a column to 1-10 range"""
    if df.empty: return df
    min_v = df[target_col].min()
    max_v = df[target_col].max()
    
    if max_v == min_v:
        df['ROI Index'] = 5.0
    else:
        df['ROI Index'] = ((df[target_col] - min_v) / (max_v - min_v)) * 9 + 1
    return df

# --- MAIN APP ---
def main():
    st.title("🧠 FPL Pro Predictor: ROI Engine")
    st.write("Advanced algorithmic prediction based on Historical Resistance vs Future Opportunity.")

    data, fixtures = load_data()
    if not data or not fixtures:
        return

    # 1. Pre-Process Data
    teams = data['teams']
    team_names = {t['id']: t['name'] for t in teams}
    
    # Analyze Fixtures for all teams
    team_schedule = process_fixture_difficulty(fixtures, len(teams))
    
    # Calculate Team Defensive Strength (for Clean Sheet Potential)
    # Logic: Lower total conceded = Higher Strength
    team_conceded = {t['id']: t['strength_defence_home'] + t['strength_defence_away'] for t in teams}
    max_str = max(team_conceded.values())
    # Normalized 0-10 (10 is best defense)
    team_def_strength = {k: 10 - ((v/max_str)*10) + 5 for k,v in team_conceded.items()}

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("🔮 Prediction Horizon")
    horizon_option = st.sidebar.selectbox(
        "Predict for upcoming:",
        options=[1, 5, 10],
        format_func=lambda x: f"Next {x} Fixture{'s' if x > 1 else ''}"
    )

    st.sidebar.divider()
    st.sidebar.header("⚖️ Model Weights")
    
    # Weights for GK/DEF
    with st.sidebar.expander("GK & Defender Weights"):
        w_cs = st.slider("Clean Sheet Potential", 0.1, 1.0, 0.8)
        w_ppm_def = st.slider("Points Per Match (DEF)", 0.1, 1.0, 0.6)
        w_fix_def = st.slider("Fixture Favourability (DEF)", 0.1, 1.0, 0.7)

    # Weights for MID/FWD
    with st.sidebar.expander("Mid & Attacker Weights"):
        w_xgi = st.slider("Total xGI Threat", 0.1, 1.0, 0.9)
        w_ppm_att = st.slider("Points Per Match (ATT)", 0.1, 1.0, 0.5)
        w_fix_att = st.slider("Fixture Favourability (ATT)", 0.1, 1.0, 0.7)

    st.sidebar.divider()
    min_minutes = st.sidebar.slider("Min. Minutes Played", 0, 2000, 400)

    # --- ANALYSIS FUNCTION ---
    def run_analysis(player_type_ids, is_defense):
        candidates = []
        
        for p in data['elements']:
            # Filters
            if p['element_type'] not in player_type_ids: continue
            if p['status'] != 'a': continue
            if p['minutes'] < min_minutes: continue

            tid = p['team']
            
            # 1. Fixture Metrics
            past_favs = team_schedule[tid]['past']
            future_favs = team_schedule[tid]['future']
            
            # Average rating of past opponents
            past_score = get_avg_favourability(past_favs)
            
            # Average rating of next N opponents
            future_score = get_avg_favourability(future_favs, limit=horizon_option)

            # 2. Player Stats
            try:
                ppm = float(p['points_per_game'])
                cost = p['now_cost'] / 10.0
                
                if is_defense:
                    # DEF FORMULA
                    # CS Potential: Combination of player's clean sheets & team strength
                    cs_potential = (float(p['clean_sheets_per_90']) * 10) + (team_def_strength[tid] / 2)
                    
                    base_score = (cs_potential * w_cs) + (ppm * w_ppm_def) + (future_score * w_fix_def)
                else:
                    # ATT FORMULA
                    xgi = float(p.get('expected_goal_involvements_per_90', 0)) * 10
                    
                    base_score = (xgi * w_xgi) + (ppm * w_ppm_att) + (future_score * w_fix_att)

                # 3. THE "RESISTANCE" CALCULATION
                # Logic: Divide by Past Favourability.
                # If past games were easy (High Favourability), Score reduces.
                # If past games were hard (Low Favourability), Score increases.
                # We clamp past_score to avoid extreme skews (e.g. between 2.0 and 5.0)
                clamped_past = max(2.0, min(past_score, 5.0))
                
                raw_roi = base_score / clamped_past

                candidates.append({
                    "Name": p['web_name'],
                    "Team": team_names[tid],
                    "Price": cost,
                    "PPM": ppm,
                    "Fix. Score (Fut)": round(future_score, 2),
                    "Fix. Score (Past)": round(past_score, 2),
                    "Raw Score": raw_roi
                })

            except Exception as e:
                continue

        # Create DF and Normalize
        df = pd.DataFrame(candidates)
        if not df.empty:
            df = normalize_scores(df, "Raw Score")
            df = df.sort_values(by="ROI Index", ascending=False).head(25)
            
            # Drop raw score for display
            df = df.drop(columns=["Raw Score"])
            
            # Reorder columns
            cols = ["ROI Index", "Name", "Team", "Price", "PPM", "Fix. Score (Fut)", "Fix. Score (Past)"]
            df = df[cols]
            
        return df

    # --- TABS ---
    tab_gk, tab_def, tab_mid, tab_fwd = st.tabs([
        "🧤 Goalkeepers", "🛡️ Defenders", "⚔️ Midfielders", "⚽ Forwards"
    ])

    # Common column config
    roi_config = st.column_config.ProgressColumn(
        "ROI Index (1-10)",
        help="10 = Must Buy. Calculated by dividing potential by past fixture ease.",
        format="%.1f",
        min_value=1,
        max_value=10,
    )
    price_config = st.column_config.NumberColumn("Price (£m)", format="£%.1f")

    # 1. GOALKEEPERS
    with tab_gk:
        st.caption("Ranking based on: Clean Sheets + Save Points + Fixture Swing")
        df_gk = run_analysis([1], is_defense=True)
        if not df_gk.empty:
            st.dataframe(df_gk, hide_index=True, column_config={"ROI Index": roi_config, "Price": price_config})
        else:
            st.warning("No data found.")

    # 2. DEFENDERS
    with tab_def:
        st.caption("Ranking based on: Clean Sheets + Attacking Threat (if any) + Fixture Swing")
        df_def = run_analysis([2], is_defense=True)
        if not df_def.empty:
            st.dataframe(df_def, hide_index=True, column_config={"ROI Index": roi_config, "Price": price_config})

    # 3. MIDFIELDERS
    with tab_mid:
        st.caption("Ranking based on: xGI + PPM + Fixture Swing")
        df_mid = run_analysis([3], is_defense=False)
        if not df_mid.empty:
            st.dataframe(df_mid, hide_index=True, column_config={"ROI Index": roi_config, "Price": price_config})

    # 4. FORWARDS
    with tab_fwd:
        st.caption("Ranking based on: xGI + PPM + Fixture Swing")
        df_fwd = run_analysis([4], is_defense=False)
        if not df_fwd.empty:
            st.dataframe(df_fwd, hide_index=True, column_config={"ROI Index": roi_config, "Price": price_config})

if __name__ == "__main__":
    main()
