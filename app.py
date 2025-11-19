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

# --- LOGIC ENGINE ---
def process_fixtures(fixtures, teams_data):
    team_map = {t['id']: t['short_name'] for t in teams_data}
    team_sched = {t['id']: {'past': [], 'future': []} for t in teams_data}

    for f in fixtures:
        if not f['kickoff_time']: continue

        h = f['team_h']
        a = f['team_a']
        h_diff = f['team_h_difficulty']
        a_diff = f['team_a_difficulty']

        # Favourability: 6 - Difficulty + 0.5 for Home
        h_fav = (6 - h_diff) + 0.5
        a_fav = (6 - a_diff)

        h_display = f"{team_map[a]}(H)"
        a_display = f"{team_map[h]}(A)"

        h_obj = {'score': h_fav, 'display': h_display}
        a_obj = {'score': a_fav, 'display': a_display}

        if f['finished']:
            team_sched[h]['past'].append(h_obj)
            team_sched[a]['past'].append(a_obj)
        else:
            team_sched[h]['future'].append(h_obj)
            team_sched[a]['future'].append(a_obj)

    return team_sched

def get_aggregated_data(schedule_list, limit=None):
    if not schedule_list:
        return 3.0, "-"
    subset = schedule_list[:limit] if limit else schedule_list
    avg_score = sum(item['score'] for item in subset) / len(subset)
    display_str = ", ".join([item['display'] for item in subset])
    return avg_score, display_str

def min_max_scale(series):
    """Scales a pandas series to 0-10 range"""
    if series.empty: return series
    min_v = series.min()
    max_v = series.max()
    if max_v == min_v: return 5.0
    return ((series - min_v) / (max_v - min_v)) * 10

# --- MAIN APP ---
def main():
    st.title("🧠 FPL Pro Predictor: Weighted ROI Engine")
    st.markdown("### Identifies assets by balancing **Raw Points Potential** with **Price Value**.")

    data, fixtures = load_data()
    if not data or not fixtures:
        return

    teams = data['teams']
    team_names = {t['id']: t['name'] for t in teams}
    team_schedule = process_fixtures(fixtures, teams)
    
    # Team Defense Strength (for Clean Sheet Calc)
    team_conceded = {t['id']: t['strength_defence_home'] + t['strength_defence_away'] for t in teams}
    max_str = max(team_conceded.values()) if team_conceded else 1
    team_def_strength = {k: 10 - ((v/max_str)*10) + 5 for k,v in team_conceded.items()}

    # --- SIDEBAR ---
    st.sidebar.header("🔮 Prediction Horizon")
    horizon_option = st.sidebar.selectbox(
        "Predict for upcoming:",
        options=[1, 5, 10],
        format_func=lambda x: f"Next {x} Fixture{'s' if x > 1 else ''}"
    )

    st.sidebar.divider()
    st.sidebar.header("⚖️ Global Model Weights")
    
    st.sidebar.markdown("**Price Sensitivity**")
    w_budget = st.sidebar.slider(
        "Price Importance (Value vs Raw Points)", 
        0.0, 1.0, 0.5,
        help="0.0 = Ignore Price (Haaland/Salah Top). 1.0 = Pure Value (Cheap Gems Top). 0.5 = Balanced."
    )
    
    st.sidebar.markdown("**Attribute Weights (Default 0.5)**")
    with st.sidebar.expander("GK & Defender", expanded=False):
        w_cs = st.slider("Clean Sheet Potential", 0.1, 1.0, 0.5)
        w_ppm_def = st.slider("Points Per Match (DEF)", 0.1, 1.0, 0.5)
        w_fix_def = st.slider("Fixture Favourability (DEF)", 0.1, 1.0, 0.5)

    with st.sidebar.expander("Mid & Attacker", expanded=False):
        w_xgi = st.slider("Total xGI Threat", 0.1, 1.0, 0.5)
        w_ppm_att = st.slider("Points Per Match (ATT)", 0.1, 1.0, 0.5)
        w_fix_att = st.slider("Fixture Favourability (ATT)", 0.1, 1.0, 0.5)

    st.sidebar.divider()
    min_minutes = st.sidebar.slider("Min. Minutes Played", 0, 2000, 400)

    # --- ANALYSIS ---
    def run_analysis(player_type_ids, is_defense):
        candidates = []
        
        for p in data['elements']:
            if p['element_type'] not in player_type_ids: continue
            if p['minutes'] < min_minutes: continue

            tid = p['team']
            
            # 1. Fixture Metrics
            past_score, _ = get_aggregated_data(team_schedule[tid]['past'])
            future_score, future_display = get_aggregated_data(team_schedule[tid]['future'], limit=horizon_option)

            # 2. Base Metrics
            try:
                ppm = float(p['points_per_game'])
                price = p['now_cost'] / 10.0
                
                if is_defense:
                    # GK/DEF Logic
                    cs_potential = (float(p['clean_sheets_per_90']) * 10) + (team_def_strength[tid] / 2)
                    base_score = (cs_potential * w_cs) + (ppm * w_ppm_def) + (future_score * w_fix_def)
                else:
                    # MID/FWD Logic
                    xgi = float(p.get('expected_goal_involvements_per_90', 0)) * 10
                    base_score = (xgi * w_xgi) + (ppm * w_ppm_att) + (future_score * w_fix_att)

                # 3. Resistance Adjustment (Performance vs Past Difficulty)
                # A high base score achieved against Hard teams (Low Past Score) is worth MORE.
                resistance_factor = max(2.0, min(past_score, 5.0))
                
                # This is the "Raw Performance" metric (Ignoring Price)
                raw_perf_metric = base_score / resistance_factor
                
                # Status
                status_icon = "✅" if p['status'] == 'a' else ("⚠️" if p['status'] == 'd' else "❌")

                candidates.append({
                    "Name": f"{status_icon} {p['web_name']}",
                    "Team": team_names[tid],
                    "Price": price,
                    "Upcoming Fixtures": future_display,
                    "PPM": ppm,
                    "Future Fix": round(future_score, 2),
                    "Past Fix": round(past_score, 2),
                    "Raw_Metric": raw_perf_metric, # For internal calc
                })

            except Exception:
                continue

        # --- DATAFRAME & WEIGHTING ---
        df = pd.DataFrame(candidates)
        if df.empty: return df

        # 1. Normalize Raw Performance (0-10)
        # This rates Haaland 10/10 even if he costs £15m
        df['Norm_Perf'] = min_max_scale(df['Raw_Metric'])
        
        # 2. Calculate & Normalize Value (0-10)
        # This rates cheap players high
        df['Value_Metric'] = df['Raw_Metric'] / df['Price']
        df['Norm_Value'] = min_max_scale(df['Value_Metric'])
        
        # 3. Apply Price Sensitivity Weight
        # If w_budget is 0.5:  50% Performance Score + 50% Value Score
        df['ROI Index'] = (df['Norm_Perf'] * (1 - w_budget)) + (df['Norm_Value'] * w_budget)
        
        # Sort and Cleanup
        df = df.sort_values(by="ROI Index", ascending=False).head(30)
        
        cols = ["ROI Index", "Name", "Team", "Price", "Upcoming Fixtures", "PPM", "Future Fix", "Past Fix"]
        return df[cols]

    # --- DISPLAY ---
    tab_gk, tab_def, tab_mid, tab_fwd = st.tabs(["🧤 GK", "🛡️ DEF", "⚔️ MID", "⚽ FWD"])

    # Config
    col_config = {
        "ROI Index": st.column_config.ProgressColumn(
            "ROI Index", format="%.1f", min_value=0, max_value=10,
            help="Weighted combination of Raw Points Potential and Price Value."
        ),
        "Price": st.column_config.NumberColumn("£", format="£%.1f"),
        "Upcoming Fixtures": st.column_config.TextColumn("Opponents", width="medium"),
        "PPM": st.column_config.NumberColumn("Pts/G", format="%.1f"),
        "Future Fix": st.column_config.NumberColumn("Fut Diff", help="Higher = Easier upcoming games"),
        "Past Fix": st.column_config.NumberColumn("Past Diff", help="Higher = Easier past games (Low score here boosts rating)"),
    }

    # Render Tabs
    for tab, p_ids, is_def in [
        (tab_gk, [1], True),
        (tab_def, [2], True),
        (tab_mid, [3], False),
        (tab_fwd, [4], False)
    ]:
        with tab:
            df = run_analysis(p_ids, is_def)
            if not df.empty:
                st.dataframe(df, hide_index=True, column_config=col_config, use_container_width=True)
            else:
                st.warning("No players found.")

if __name__ == "__main__":
    main()
