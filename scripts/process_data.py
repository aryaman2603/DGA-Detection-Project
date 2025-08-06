import pandas as pd
from pathlib import Path

def main():
    """
    Loads raw data from three different sources with specific formats,
    labels them, combines them, shuffles the result, and saves a final,
    processed CSV file.
    """
    # Define project root and data paths
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / 'data'
    
    print("Starting data processing...")

    # 1. Process Benign Domains (Alexa)
    # ==================================
    print("Loading Alexa top 1 million...")
    alexa_path = DATA_DIR / 'top-1m.csv'
    # The file has no header, so we provide column names.
    alexa_df = pd.read_csv(alexa_path, names=['rank', 'domain'])
    # Keep only the 'domain' column.
    benign_df = alexa_df[['domain']].copy()
    # Add the label for benign domains.
    benign_df['label'] = 0

    # 2. Process 360 DGA Domains
    # ==================================
    print("Loading 360 DGA domains...")
    netlab_path = DATA_DIR / '360_dga.csv'
    # This file has a header, so pandas reads it automatically.
    netlab_df = pd.read_csv(netlab_path)
    # The domain column is capitalized ('Domain'), so we select it and rename for consistency.
    dga_360_df = netlab_df[['Domain']].rename(columns={'Domain': 'domain'}).copy()
    # Add the label for malicious domains.
    dga_360_df['label'] = 1

    # 3. Process Bambenek DGA Domains
    # ==================================
    print("Loading Bambenek DGA domains...")
    bambenek_path = DATA_DIR / 'bambenek_dga.csv'
    bambenek_df = pd.read_csv(bambenek_path)
    # Also select and rename the capitalized 'Domain' column.
    dga_bambenek_df = bambenek_df[['Domain']].rename(columns={'Domain': 'domain'}).copy()
    # Add the label for malicious domains.
    dga_bambenek_df['label'] = 1
    
    # 4. Combine and Shuffle All Datasets
    # ==================================
    print("Combining all datasets...")
    # Put all the processed dataframes into a list to be concatenated.
    all_dfs = [benign_df, dga_360_df, dga_bambenek_df]
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print("Shuffling data for randomness...")
    # Shuffle the combined dataframe. frac=1 means shuffle 100% of the rows.
    shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 5. Save the final dataset
    # ==================================
    output_path = DATA_DIR / 'processed_binary_dga_dataset.csv'
    shuffled_df.to_csv(output_path, index=False)
    
    print(f"✅ Processing complete! Unified dataset saved to: {output_path}")
    print(f"Total domains: {len(shuffled_df)}")
    print("Final label distribution:")
    print(shuffled_df['label'].value_counts())

if __name__ == "__main__":
    main()