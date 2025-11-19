import streamlit as st
import requests
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="FPL Pro Predictor 25/26", page_icon="⚽", layout="wide")

# --- CUSTOM CSS FOR BIG TABS ---
st.markdown("""
<style>
    /* Increase Tab Size and Font */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-size: 20px; /* Bigger Text */
        font-weight: 700; /* Bold */
        color: #4a4a4a;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e2e6;
        color: #1f77b4;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff;
        border-top: 3px solid #00cc00;
        color: #00cc00;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
    }
    /* Button Styling */
    .stButton button { width: 100%; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'page' not in st.session_state:
    st.session_state.page = 0

def reset_page():
    st.session_state.page = 0

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
    if max_v == min_v: return pd.Series([5.0] * len(series), index=series.index)
    return ((series - min_v) / (max_v - min_v)) * 10

# --- MAIN APP ---
def main():
    st.title("🧠 FPL Pro Predictor: Advanced Metrics")
    st.markdown("### Deep-Dive Analysis using Position-Specific Algorithms")

    data, fixtures = load_data()
    if not data or not fixtures:
        return

    teams = data['teams']
    team_names = {t['id']: t['name'] for t in teams}
    team_schedule = process_fixtures(fixtures, teams)
    
    # Team Defense Strength
    team_conceded = {t['id']: t['strength_defence_home'] + t['strength_defence_away'] for t in teams}
    max_str = max(team_conceded.values()) if team_conceded else 1
    team_def_strength = {k: 10 - ((v/max_str)*10) + 5 for k,v in team_conceded.items()}

    # --- SIDEBAR ---
    st.sidebar.header("🔮 Prediction Horizon")
    horizon_option = st.sidebar.selectbox(
        "Predict for upcoming:",
        options=[1, 5, 10],
        format_func=lambda x: f"Next {x} Fixture{'s' if x > 1 else ''}",
        on_change=reset_page
    )

    st.sidebar.divider()
    st.sidebar.header("⚖️ Model Weights")
    
    # PRICE SENSITIVITY
    w_budget = st.sidebar.slider(
        "Price Importance", 
        0.0, 1.0, 0.5,
        help="0.0 = Performance Only. 1.0 = Value Only.",
        on_change=reset_page
    )
    
    # ATTRIBUTE WEIGHTS
    st.sidebar.subheader("Algorithm Tweaks")
    with st.sidebar.expander("GK & Defender Settings", expanded=False):
        w_cs = st.slider("Clean Sheet / Solidity", 0.1, 1.0, 0.5, on_change=reset_page)
        w_ppm_def = st.slider("Form (PPM)", 0.1, 1.0, 0.5, on_change=reset_page)
        w_fix_def = st.slider("Fixture Ease", 0.1, 1.0, 0.5, on_change=reset_page)

    with st.sidebar.expander("Mid & Attacker Settings", expanded=False):
        w_xgi = st.slider("Attacking Threat (xGI)", 0.1, 1.0, 0.5, on_change=reset_page)
        w_ppm_att = st.slider("Form (PPM)", 0.1, 1.0, 0.5, on_change=reset_page)
        w_fix_att = st.slider("Fixture Ease", 0.1, 1.0, 0.5, on_change=reset_page)

    st.sidebar.divider()
    min_minutes = st.sidebar.slider(
        "Min. Minutes Played", 0, 2000, 0, 
        help="Set to 0 to analyze ALL players.",
        on_change=reset_page
    )

    # --- ANALYSIS ---
    def run_analysis(player_type_ids, pos_category):
        candidates = []
        
        for p in data['elements']:
            if p['element_type'] not in player_type_ids: continue
            if p['minutes'] < min_minutes: continue

            tid = p['team']
            
            # 1. Fixture Metrics
            past_score, _ = get_aggregated_data(team_schedule[tid]['past'])
            future_score, future_display = get_aggregated_data(team_schedule[tid]['future'], limit=horizon_option)

            # 2. Extract Advanced Stats
            try:
                ppm = float(p['points_per_game'])
                price = p['now_cost'] / 10.0
                if price <= 0: price = 4.0

                # New Metrics
                xgi_90 = float(p.get('expected_goal_involvements_per_90', 0))
                xgc_90 = float(p.get('expected_goals_conceded_per_90', 0))
                saves_90 = float(p.get('saves_per_90', 0))
                
                # --- ALGORITHMS BY POSITION ---
                
                if pos_category == "GK":
                    # GK Logic: PPM + Saves + Clean Sheet Potential (Inverse xGC)
                    # Note: xGC is bad for CS, but High xGC often leads to Save Points.
                    # Hybrid Score: 
                    # 1. Save Potential (Saves/90)
                    # 2. Team Strength (For CS)
                    
                    cs_potential = (10 - (xgc_90 * 3)) # Approx inversion (Lower xGC is better)
                    cs_potential = max(0, cs_potential) + (team_def_strength[tid] / 2)
                    
                    # Base Score
                    base_score = (cs_potential * w_cs) + (saves_90 * 20 * 0.2) + (ppm * w_ppm_def) + (future_score * w_fix_def)
                    
                    # Projection
                    base_strength = (ppm * 0.6) + (saves_90 * 3) + (cs_potential * 0.1)
                    
                elif pos_category == "DEF":
                    # DEF Logic: PPM + xGI (Attack) + xGC (Defense/Solidty)
                    # "Defensive Contribution" approximated by inverse xGC
                    def_solidity = max(0, (2.5 - xgc_90) * 4) # Scale 0-10
                    
                    att_threat = xgi_90 * 10
                    
                    base_score = (def_solidity * w_cs) + (att_threat * 0.3) + (ppm * w_ppm_def) + (future_score * w_fix_def)
                    
                    # Projection
                    base_strength = (ppm * 0.6) + (xgi_90 * 5) + (def_solidity * 0.2)

                elif pos_category == "MID":
                    # MID Logic: PPM + xGI + Slight Defensive Penalty
                    # Mids get CS points, so xGC matters slightly
                    def_solidity = max(0, (3.0 - xgc_90) * 2)
                    
                    base_score = ((xgi_90 * 10) * w_xgi) + (ppm * w_ppm_att) + (future_score * w_fix_att) + (def_solidity * 0.1)
                    
                    # Projection
                    base_strength = (ppm * 0.7) + (xgi_90 * 8)
                    
                else: # FWD
                    # FWD Logic: Pure Attack
                    base_score = ((xgi_90 * 10) * w_xgi) + (ppm * w_ppm_att) + (future_score * w_fix_att)
                    
                    # Projection
                    base_strength = (ppm * 0.7) + (xgi_90 * 9)

                # 3. PREDICTED POINTS CALCULATION
                fix_multiplier = future_score / 3.5
                proj_points = base_strength * fix_multiplier

                # 4. Resistance Adjustment (For ROI Index)
                resistance_factor = max(2.0, min(past_score, 5.0))
                raw_perf_metric = base_score / resistance_factor
                
                status_icon = "✅" if p['status'] == 'a' else ("⚠️" if p['status'] == 'd' else "❌")

                candidates.append({
                    "Name": f"{status_icon} {p['web_name']}",
                    "Team": team_names[tid],
                    "Price": price,
                    "xGI": xgi_90,
                    "xGC": xgc_90,
                    "Saves": saves_90,
                    "Upcoming Fixtures": future_display,
                    "PPM": ppm,
                    "Exp. Pts": proj_points,
                    "Future Fix": round(future_score, 2),
                    "Past Fix": round(past_score, 2),
                    "Raw_Metric": raw_perf_metric,
                })

            except Exception:
                continue

        # --- DATAFRAME ---
        df = pd.DataFrame(candidates)
        if df.empty: return df

        # Normalize Performance (0-10)
        df['Norm_Perf'] = min_max_scale(df['Raw_Metric'])
        
        # Normalize Value (0-10)
        df['Value_Metric'] = df['Raw_Metric'] / df['Price']
        df['Norm_Value'] = min_max_scale(df['Value_Metric'])
        
        # ROI Calculation
        df['ROI Index'] = (df['Norm_Perf'] * (1 - w_budget)) + (df['Norm_Value'] * w_budget)
        
        df = df.sort_values(by="ROI Index", ascending=False)
        
        return df

    # --- DISPLAY ---
    def render_tab(p_ids, pos_category):
        df = run_analysis(p_ids, pos_category)
        
        if df.empty:
            st.warning("No players found.")
            return

        # Pagination
        items_per_page = 50
        total_items = len(df)
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
        
        if st.session_state.page >= total_pages: st.session_state.page = total_pages - 1
        if st.session_state.page < 0: st.session_state.page = 0
            
        start_idx = st.session_state.page * items_per_page
        end_idx = start_idx + items_per_page
        
        df_display = df.iloc[start_idx:end_idx]
        
        st.caption(f"Showing **{start_idx + 1}-{min(end_idx, total_items)}** of **{total_items}** players")
        
        # DYNAMIC COLUMNS BASED ON POSITION
        base_cols = {
            "ROI Index": st.column_config.ProgressColumn("ROI Index", format="%.1f", min_value=0, max_value=10),
            "Exp. Pts": st.column_config.NumberColumn("Exp. Pts", format="%.1f", help="Projected Points based on Form + Metrics + Fixture"),
            "Price": st.column_config.NumberColumn("£", format="£%.1f"),
            "Upcoming Fixtures": st.column_config.TextColumn("Opponents", width="medium"),
            "PPM": st.column_config.NumberColumn("Pts/G", format="%.1f"),
        }
        
        # Add specific stat columns
        if pos_category == "GK":
            df_final = df_display[["ROI Index", "Name", "Team", "Exp. Pts", "Price", "Saves", "xGC", "Upcoming Fixtures", "PPM"]]
            base_cols["Saves"] = st.column_config.NumberColumn("Saves/90", format="%.2f")
            base_cols["xGC"] = st.column_config.NumberColumn("xGC/90", format="%.2f", help="Exp. Goals Conceded (Lower is better for CS)")
            
        elif pos_category in ["DEF", "MID"]:
            df_final = df_display[["ROI Index", "Name", "Team", "Exp. Pts", "Price", "xGI", "xGC", "Upcoming Fixtures", "PPM"]]
            base_cols["xGI"] = st.column_config.NumberColumn("xGI/90", format="%.2f", help="Exp. Goal Involvement (Attack)")
            base_cols["xGC"] = st.column_config.NumberColumn("xGC/90", format="%.2f", help="Defensive Liability/Solidity")
            
        else: # FWD
            df_final = df_display[["ROI Index", "Name", "Team", "Exp. Pts", "Price", "xGI", "Upcoming Fixtures", "PPM"]]
            base_cols["xGI"] = st.column_config.NumberColumn("xGI/90", format="%.2f")

        st.dataframe(
            df_final, 
            hide_index=True, 
            column_config=base_cols,
            use_container_width=True
        )
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("⬅️ Previous 50", disabled=(st.session_state.page == 0), key=f"prev_{pos_category}"):
                st.session_state.page -= 1
                st.rerun()
        with c3:
            if st.button("Next 50 ➡️", disabled=(st.session_state.page == total_pages - 1), key=f"next_{pos_category}"):
                st.session_state.page += 1
                st.rerun()

    # --- RENDER BIG TABS ---
    tab_gk, tab_def, tab_mid, tab_fwd = st.tabs([
        "🧤 GOALKEEPERS", 
        "🛡️ DEFENDERS", 
        "⚔️ MIDFIELDERS", 
        "⚽ FORWARDS"
    ])

    with tab_gk: render_tab([1], "GK")
    with tab_def: render_tab([2], "DEF")
    with tab_mid: render_tab([3], "MID")
    with tab_fwd: render_tab([4], "FWD")

if __name__ == "__main__":
    main()
