import pandas as pd
import os

def main():
    file_path = 'predictions_r24.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # Columns to show
    cols = ['player_name', 'prob_play', 'predicted_points', 'price', 'value']
    
    positions = ['G', 'D', 'M', 'F']
    pos_map = {'G': 'GOALKEEPER', 'D': 'DEFENDER', 'M': 'MIDFIELDER', 'F': 'FORWARD'}

    for pos in positions:
        print("\n" + "="*80)
        print(f" POSITION: {pos_map[pos]} ".center(80, "="))
        print("="*80)
        
        pos_df = df[df['position'] == pos]
        
        print(f"\n--- TOP 8 BY PREDICTED POINTS (ROUND 24) ---")
        top_pts = pos_df.sort_values('predicted_points', ascending=False).head(8)
        print(top_pts[cols].to_string(index=False))
        
        print(f"\n--- TOP 8 BY VALUE (PTS / PRICE) ---")
        top_val = pos_df.sort_values('value', ascending=False).head(8)
        print(top_val[cols].to_string(index=False))

if __name__ == "__main__":
    main()
