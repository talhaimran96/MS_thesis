import os
import pandas as pd

raw_data_dir = "Data/raw/"
print(f"Exploring {raw_data_dir}...\n")

def explore_dir(path, depth=0, max_depth=2):
    if depth > max_depth:
        return
    try:
        items = os.listdir(path)
    except Exception as e:
        print(f"{'  ' * depth}Error reading {path}: {e}")
        return
    
    for item in items[:15]:
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            print(f"{'  ' * depth}- [DIR] {item}/")
            explore_dir(full_path, depth + 1, max_depth)
        else:
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"{'  ' * depth}- [FILE] {item} ({size_mb:.2f} MB)")
            if item.endswith('.csv') and depth == 0:
                try:
                    df = pd.read_csv(full_path, nrows=5)
                    print(f"{'  ' * (depth+1)}-> CSV Columns: {list(df.columns)}")
                    # getting shape by reading chunk to avoid massive memory usage if it was huge
                    # but 80MB is small enough
                    full_df = pd.read_csv(full_path)
                    print(f"{'  ' * (depth+1)}-> CSV Shape: {full_df.shape}")
                except Exception as e:
                    print(f"{'  ' * (depth+1)}-> Error reading CSV: {e}")
    if len(items) > 15:
        print(f"{'  ' * depth}- ... and {len(items) - 15} more items.")

if __name__ == "__main__":
    explore_dir(raw_data_dir)
