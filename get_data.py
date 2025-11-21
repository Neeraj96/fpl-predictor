import pandas as pd
import requests
import io

def download_historical_data():
    print("⏳ Initializing Machine Learning Data Warehouse...")
    
    # We are in the 2025-26 Season.
    # We need history for training (2020-2025).
    # We try to get 2025-26 so far, but if it fails, we rely on Live API.
    seasons = [
        "2020-21",
        "2021-22",
        "2022-23",
        "2023-24",
        "2024-25", # Completed Season (Corrupted file needs fixing)
        "2025-26"  # Current Season (Might be partial)
    ]
    
    base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
    all_data = []

    for season in seasons:
        print(f"📥 Fetching season: {season}...")
        try:
            url = f"{base_url}/{season}/gws/merged_gw.csv"
            
            # Fetch raw content first to handle errors gracefully
            response = requests.get(url)
            
            if response.status_code == 404:
                print(f"   ⚠️ Season {season} data not found in repo (Using Live API instead).")
                continue
                
            # FIX: Use 'python' engine with on_bad_lines to skip the corrupted rows in 24-25
            df = pd.read_csv(
                io.BytesIO(response.content), 
                encoding='utf-8', 
                on_bad_lines='skip', 
                low_memory=False
            )
            
            # Add context
            df['season_id'] = season
            
            # Normalize columns (FPL repo changes headers slightly over years)
            # We map them to a standard set
            if 'total_points' not in df.columns: continue # Skip if critical data missing
            
            # Standardize column selection
            # We only keep columns that consistently exist across 5 years
            cols_to_keep = [
                'name', 'position', 'team', 'minutes', 'total_points',
                'was_home', 'opponent_team', 'goals_scored', 'assists',
                'clean_sheets', 'goals_conceded', 'expected_goals', 
                'expected_assists', 'influence', 'creativity', 'threat',
                'ict_index', 'value', 'season_id', 'GW'
            ]
            
            # Safety check for columns
            existing = [c for c in cols_to_keep if c in df.columns]
            df = df[existing]
            
            # Conversions
            if 'value' in df.columns:
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                
            all_data.append(df)
            print(f"   ✅ Loaded {len(df)} rows.")
            
        except Exception as e:
            print(f"   ❌ Error processing {season}: {str(e)[:100]}")

    if not all_data:
        print("❌ CRITICAL: No data downloaded. Check internet.")
        return

    print("🔄 Combining 5 Years of Data...")
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Final Polish
    master_df.fillna(0, inplace=True)
    if 'was_home' in master_df.columns:
        master_df['was_home'] = master_df['was_home'].astype(bool).astype(int)
    
    output_file = "fpl_5_year_history.csv"
    master_df.to_csv(output_file, index=False)
    print(f"🚀 SUCCESS! AI Database built: '{output_file}' ({len(master_df)} matches).")

if __name__ == "__main__":
    download_historical_data()