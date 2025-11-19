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
    .stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

if 'page' not in st.session_state: st.session_state.page = 0

def reset_page(): st.session_state.page = 0

API_BASE = "https://fantasy.premierleague.com/api"

@st.cache_data(ttl=1800)
def load_data():
    try:
        bootstrap = requests.get(f"{API_BASE}/bootstrap-static/").json()
        fixtures = requests.get(f"{API_BASE}/fixtures/").json()
        return bootstrap, fixtures
    except:
        st.error("API Error")
        return None, None

def process_fixtures(fixtures, teams_data):
    team_map = {t['id']: t['short_name'] for t in teams_data}
    team_sched = {t['id']: {'past': [], 'future': []} for t in teams_data}

    for f in fixtures:
        if not f['kickoff_time']: continue
        h, a = f['team_h'], f['team_a']
        h_diff, a_diff = f['team_h_difficulty'], f['team_a_difficulty']
        
        h_fav = (6 - h_diff) + 0.5
        a_fav = (6 - a_diff)
        
        h_obj = {'score': h_fav, 'display': f"{team_map[a]}(H)"}
        a_obj = {'score': a_fav, 'display': f"{team_map[h]}(A)"}

        if f['finished']:
            team_sched[h]['past'].append(h_obj)
            team_sched[a]['past'].append(a_obj)
        else:
            team_sched[h]['future'].append(h_obj)
            team_sched[a]['future'].append(a_obj)
    return team_sched

def get_aggregated_data(schedule_list, limit=None):
    if not schedule_list: return 3.0, "-"
    subset = schedule_list[:limit] if limit else schedule_list
    avg_score = sum(item['score'] for item in subset) / len(subset)
    return avg_score, ", ".join([item['display'] for item in subset])

def min_max_scale(series):
    if series.empty: return series
    min_v, max_v = series.min(), series.max()
    if max_v == min_v: return pd.Series([5.0]*len(series), index=series.index)
    return ((series - min_v) / (max_v - min_v)) * 10

def main():
    st.title("🧠 FPL Pro Predictor: Diagnostic Mode")
    st.markdown("### Comparison of Raw Metrics vs Scaled ROI")

    data, fixtures = load_data()
    if not data or not fixtures: return

    teams = data['teams']
    team_names = {t['id']: t['name'] for t in teams}
    team_schedule = process_fixtures(fixtures, teams)
    
    team_conceded = {t['id']: t['strength_defence_home'] + t['strength_defence_away'] for t in teams}
    max_str = max(team_conceded.values()) if team_conceded else 1
    team_def_strength = {k: 10 - ((v/max_str)*10) + 5 for k,v in team_conceded.items()}

    # --- SIDEBAR ---
    st.sidebar.header("🔮 Settings")
    horizon_option = st.sidebar.selectbox("Fixtures Lookahead", [1, 5, 10], on_change=reset_page)
    w_budget = st.sidebar.slider("Price Weight (Value)", 0.0, 1.0, 0.5, on_change=reset_page)
    
    st.sidebar.divider()
    st.sidebar.info("Forcing Equal Weights (0.5) for attributes to isolate variables.")
    
    min_minutes = st.sidebar.slider("Min Minutes", 0, 2000, 0, on_change=reset_page)

    def run_analysis(p_ids, is_defense):
        candidates = []
        w_stats = 0.5
        w_ppm = 0.5
        w_fix = 0.5

        for p in data['elements']:
            if p['element_type'] not in p_ids: continue
            if p['minutes'] < min_minutes: continue

            tid = p['team']
            past_score, _ = get_aggregated_data(team_schedule[tid]['past'])
            future_score, future_display = get_aggregated_data(team_schedule[tid]['future'], limit=horizon_option)

            try:
                ppm = float(p['points_per_game'])
                price = p['now_cost'] / 10.0
                if price <= 0: price = 4.0
                
                if is_defense:
                    cs_per_90 = float(p['clean_sheets_per_90'])
                    cs_potential = (cs_per_90 * 10) + (team_def_strength[tid] / 2)
                    base_score = (cs_potential * w_stats) + (ppm * w_ppm) + (future_score * w_fix)
                    stat_display = cs_per_90
                else:
                    xgi = float(p.get('expected_goal_involvements_per_90', 0))
                    # Note: xGI is usually 0.0 to 1.2 range. We multiply by 10 to match PPM scale (0-10)
                    base_score = ((xgi * 10) * w_stats) + (ppm * w_ppm) + (future_score * w_fix)
                    stat_display = xgi

                resistance_factor = max(2.0, min(past_score, 5.0))
                raw_perf_metric = base_score / resistance_factor
                
                candidates.append({
                    "Name": p['web_name'],
                    "Team": team_names[tid],
                    "Price": price,
                    "xGI/CS": stat_display, # New Column
                    "Resistance": resistance_factor,
                    "Raw Score": raw_perf_metric, # New Column
                    "Upcoming": future_display
                })
            except: continue

        df = pd.DataFrame(candidates)
        if df.empty: return df

        df['Norm_Perf'] = min_max_scale(df['Raw Score'])
        df['Value_Metric'] = df['Raw Score'] / df['Price']
        df['Norm_Value'] = min_max_scale(df['Value_Metric'])
        df['ROI Index'] = (df['Norm_Perf'] * (1 - w_budget)) + (df['Norm_Value'] * w_budget)
        
        return df.sort_values(by="ROI Index", ascending=False)

    def render_tab(p_ids, is_def):
        df = run_analysis(p_ids, is_def)
        if df.empty:
            st.warning("No players found.")
            return

        # Pagination
        items_per_page = 50
        total_items = len(df)
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
        if st.session_state.page >= total_pages: st.session_state.page = total_pages - 1
        
        start, end = st.session_state.page * items_per_page, (st.session_state.page + 1) * items_per_page
        
        col_label = "CS/90" if is_def else "xGI/90"
        
        st.dataframe(
            df.iloc[start:end],
            hide_index=True,
            column_config={
                "ROI Index": st.column_config.ProgressColumn("ROI Index", format="%.1f", min_value=0, max_value=10),
                "Price": st.column_config.NumberColumn("£", format="£%.1f"),
                "xGI/CS": st.column_config.NumberColumn(col_label, format="%.2f", help="Underlying stat (Expected Goals or Clean Sheets)"),
                "Raw Score": st.column_config.NumberColumn("Raw Score", format="%.2f", help="The unscaled score before comparing to other players"),
                "Resistance": st.column_config.NumberColumn("Resist.", format="%.2f"),
            },
            use_container_width=True
        )
        
        c1, _, c3 = st.columns([1, 2, 1])
        if c1.button("Prev", key=f"p{p_ids}") and st.session_state.page > 0: 
            st.session_state.page -= 1
            st.rerun()
        if c3.button("Next", key=f"n{p_ids}") and st.session_state.page < total_pages - 1: 
            st.session_state.page += 1
            st.rerun()

    t1, t2, t3, t4 = st.tabs(["GK", "DEF", "MID", "FWD"])
    with t1: render_tab([1], True)
    with t2: render_tab([2], True)
    with t3: render_tab([3], False)
    with t4: render_tab([4], False)

if __name__ == "__main__":
    main()
