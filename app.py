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

# --- LOGIC ENGINE ---
def process_fixtures(fixtures, teams_data):
    """
    Analyzes fixtures to create a schedule for every team.
    Stores both numerical difficulty (for math) and display strings (for UI).
    """
    # Map ID to Short Name (e.g. 1 -> ARS)
    team_map = {t['id']: t['short_name'] for t in teams_data}
    
    # Initialize structure: { team_id: { 'past': [], 'future': [] } }
    # Items in lists will be dicts: {'diff_score': float, 'display': str}
    team_sched = {t['id']: {'past': [], 'future': []} for t in teams_data}

    for f in fixtures:
        if not f['kickoff_time']: continue # Skip undefined games

        h = f['team_h']
        a = f['team_a']
        h_diff = f['team_h_difficulty']
        a_diff = f['team_a_difficulty']

        # Calculate Favourability (Higher = Easier)
        # Home advantage: +0.5 favourability
        h_fav = (6 - h_diff) + 0.5
        a_fav = (6 - a_diff)

        # Create Display Strings (e.g. "ARS(H)")
        h_display = f"{team_map[a]}(H)"
        a_display = f"{team_map[h]}(A)"

        # Data Objects
        h_obj = {'score': h_fav, 'display': h_display, 'diff': h_diff}
        a_obj = {'score': a_fav, 'display': a_display, 'diff': a_diff}

        if f['finished']:
            team_sched[h]['past'].append(h_obj)
            team_sched[a]['past'].append(a_obj)
        else:
            team_sched[h]['future'].append(h_obj)
            team_sched[a]['future'].append(a_obj)

    return team_sched

def get_aggregated_data(schedule_list, limit=None):
    """
    Returns:
    1. Average Favourability Score
    2. String representation of opponents
    """
    if not schedule_list:
        return 3.0, "-"
        
    subset = schedule_list[:limit] if limit else schedule_list
    
    # Math
    avg_score = sum(item['score'] for item in subset) / len(subset)
    
    # Visuals (Join with comma)
    display_str = ", ".join([item['display'] for item in subset])
    
    return avg_score, display_str

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
    st.markdown("### Advanced algorithmic prediction")

    data, fixtures = load_data()
    if not data or not fixtures:
        return

    # 1. Pre-Process Data
    teams = data['teams']
    team_names = {t['id']: t['name'] for t in teams}
    
    # Analyze Fixtures
    team_schedule = process_fixtures(fixtures, teams)
    
    # Calculate Team Strength (for Def Calculations)
    team_conceded = {t['id']: t['strength_defence_home'] + t['strength_defence_away'] for t in teams}
    max_str = max(team_conceded.values()) if team_conceded else 1
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
    st.sidebar.info("Defaults set to Equal (0.5). Adjust to bias the model.")
    
    # Equalized Weights Default = 0.5
    with st.sidebar.expander("GK & Defender Weights", expanded=True):
        w_cs = st.slider("Clean Sheet Potential", 0.1, 1.0, 0.5)
        w_ppm_def = st.slider("Points Per Match (DEF)", 0.1, 1.0, 0.5)
        w_fix_def = st.slider("Fixture Favourability (DEF)", 0.1, 1.0, 0.5)

    with st.sidebar.expander("Mid & Attacker Weights", expanded=True):
        w_xgi = st.slider("Total xGI Threat", 0.1, 1.0, 0.5)
        w_ppm_att = st.slider("Points Per Match (ATT)", 0.1, 1.0, 0.5)
        w_fix_att = st.slider("Fixture Favourability (ATT)", 0.1, 1.0, 0.5)

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
            past_data = team_schedule[tid]['past']
            future_data = team_schedule[tid]['future']
            
            # Past (All played so far)
            past_score, _ = get_aggregated_data(past_data)
            
            # Future (Selected Horizon)
            future_score, future_display = get_aggregated_data(future_data, limit=horizon_option)

            # 2. Player Stats
            try:
                ppm = float(p['points_per_game'])
                cost = p['now_cost'] / 10.0
                
                if is_defense:
                    # DEF FORMULA
                    cs_potential = (float(p['clean_sheets_per_90']) * 10) + (team_def_strength[tid] / 2)
                    base_score = (cs_potential * w_cs) + (ppm * w_ppm_def) + (future_score * w_fix_def)
                else:
                    # ATT FORMULA
                    xgi = float(p.get('expected_goal_involvements_per_90', 0)) * 10
                    base_score = (xgi * w_xgi) + (ppm * w_ppm_att) + (future_score * w_fix_att)

                # 3. RESISTANCE CALCULATION (ROI)
                clamped_past = max(2.0, min(past_score, 5.0))
                raw_roi = base_score / clamped_past

                candidates.append({
                    "Name": p['web_name'],
                    "Team": team_names[tid],
                    "Price": cost,
                    "Upcoming Fixtures": future_display,
                    "PPM": ppm,
                    "Fix. Score": round(future_score, 2),
                    "Raw Score": raw_roi
                })

            except Exception as e:
                continue

        # Create DF
        df = pd.DataFrame(candidates)
        if not df.empty:
            df = normalize_scores(df, "Raw Score")
            df = df.sort_values(by="ROI Index", ascending=False).head(30)
            
            # Select Columns
            cols = ["ROI Index", "Name", "Team", "Price", "Upcoming Fixtures", "PPM", "Fix. Score"]
            df = df[cols]
            
        return df

    # --- DISPLAY TABS ---
    tab_gk, tab_def, tab_mid, tab_fwd = st.tabs([
        "🧤 Goalkeepers", "🛡️ Defenders", "⚔️ Midfielders", "⚽ Forwards"
    ])

    # Column Configuration
    col_config = {
        "ROI Index": st.column_config.ProgressColumn(
            "ROI Index", format="%.1f", min_value=1, max_value=10,
            help="10 = High Potential relative to past performance."
        ),
        "Price": st.column_config.NumberColumn("£", format="£%.1f"),
        "Upcoming Fixtures": st.column_config.TextColumn(
            "Upcoming Fixtures", 
            width="large",
            help=f"Next {horizon_option} Opponents (H=Home, A=Away)"
        ),
        "PPM": st.column_config.NumberColumn("Pts/Game", format="%.1f"),
    }

    # 1. GOALKEEPERS
    with tab_gk:
        df_gk = run_analysis([1], is_defense=True)
        if not df_gk.empty:
            st.dataframe(df_gk, hide_index=True, column_config=col_config, use_container_width=True)
        else:
            st.warning("No players found. Try lowering the minutes filter.")

    # 2. DEFENDERS
    with tab_def:
        df_def = run_analysis([2], is_defense=True)
        if not df_def.empty:
            st.dataframe(df_def, hide_index=True, column_config=col_config, use_container_width=True)

    # 3. MIDFIELDERS
    with tab_mid:
        df_mid = run_analysis([3], is_defense=False)
        if not df_mid.empty:
            st.dataframe(df_mid, hide_index=True, column_config=col_config, use_container_width=True)

    # 4. FORWARDS
    with tab_fwd:
        df_fwd = run_analysis([4], is_defense=False)
        if not df_fwd.empty:
            st.dataframe(df_fwd, hide_index=True, column_config=col_config, use_container_widt
