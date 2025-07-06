import pandas as pd
import re
import os

def load_data(file_path):
    """
    Loads the CFPB complaint dataset from a specified file path.
    Ensures efficient memory usage.
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please ensure the file exists at the specified path.")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None

def filter_data(df, products, narrative_column='Consumer complaint narrative'):
    """
    Filters the DataFrame to include only specified products and
    removes records with empty complaint narratives.
    """
    initial_rows = df.shape[0]
    print(f"Initial number of rows: {initial_rows}")

    # Product filtering
    df_filtered = df[df['Product'].isin(products)].copy()
    print(f"Rows after filtering by products ({', '.join(products)}): {df_filtered.shape[0]}")

    # Remove records with empty narratives
    # Use .astype(str) to handle potential mixed types or non-string entries robustly
    df_filtered.dropna(subset=[narrative_column], inplace=True)
    df_filtered = df_filtered[df_filtered[narrative_column].astype(str).str.strip() != ''].copy() # Also check for empty strings
    print(f"Rows after removing empty narratives: {df_filtered.shape[0]}")

    print(f"Total rows removed during filtering: {initial_rows - df_filtered.shape[0]}")
    return df_filtered

def load_boilerplate_patterns(file_path='data/boilerplate_patterns.txt'):
    """
    Loads boilerplate text patterns from a specified file.
    Each line in the file is treated as a separate pattern.
    """
    patterns = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'): # Ignore empty lines and comments
                    patterns.append(line)
        print(f"Loaded {len(patterns)} boilerplate patterns from {file_path}")
    else:
        print(f"Warning: Boilerplate patterns file not found at {file_path}. No patterns will be removed.")
    return patterns

def clean_text(text, boilerplate_patterns=None):
    """
    Cleans a single text narrative:
    - Lowercasing text
    - Removing boilerplate text
    - Removing special characters (non-alphanumeric, keep spaces)
    """
    if not isinstance(text, str):
        return "" # Handle non-string inputs

    # Lowercasing text
    text = text.lower()

    # Removing boilerplate text
    if boilerplate_patterns:
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Remove special characters, keeping only letters, numbers, and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_data(file_path, output_path, products_to_include, narrative_column='Consumer complaint narrative', boilerplate_patterns_file='data/boilerplate_patterns.txt'):
    """
    Orchestrates the full data preprocessing pipeline.
    """
    print("Starting data preprocessing...")
    df = load_data(file_path)
    if df is None:
        print("Data loading failed. Preprocessing aborted.")
        return

    # Load boilerplate patterns
    boilerplate_patterns = load_boilerplate_patterns(boilerplate_patterns_file)

    df_filtered = filter_data(df, products_to_include, narrative_column)

    if df_filtered.empty:
        print("No data remaining after initial filtering. Skipping text cleaning and saving.")
        return

    print("Cleaning consumer complaint narratives...")
    # Apply cleaning to the filtered DataFrame
    # Using .astype(str) for apply to ensure consistent string operations
    df_filtered[f'cleaned_{narrative_column}'] = df_filtered[narrative_column].astype(str).apply(
        lambda x: clean_text(x, boilerplate_patterns)
    )

    # Remove rows where cleaned narrative might become empty after cleaning
    initial_cleaned_rows = df_filtered.shape[0]
    df_filtered = df_filtered[df_filtered[f'cleaned_{narrative_column}'].astype(str).str.strip() != ''].copy()
    if df_filtered.shape[0] < initial_cleaned_rows:
        print(f"Removed {initial_cleaned_rows - df_filtered.shape[0]} rows where narratives became empty after cleaning.")

    print(f"Final preprocessed data shape: {df_filtered.shape}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure output directory exists
    df_filtered.to_csv(output_path, index=False)
    print(f"Cleaned and filtered dataset saved to {output_path}")

    return df_filtered

if __name__ == "__main__":
    # Data paths
    RAW_DATA_PATH = 'data/raw/complaints.csv'
    FILTERED_DATA_PATH = 'data/processed/filtered_complaints.csv'
    BOILERPLATE_PATTERNS_FILE = 'data/raw/boilerplate_patterns.txt'

    # These are the five specified products for the project
    # Ensure these product names exactly match those in your 'complaints.csv' file.
    TARGET_PRODUCTS = [
        'Credit card',
        'Personal loan',
        'Buy Now, Pay Later (BNPL)',
        'Savings account',
        'Money transfer'
    ]

    # Run the full preprocessing pipeline
    preprocessed_df = preprocess_data(RAW_DATA_PATH, FILTERED_DATA_PATH, TARGET_PRODUCTS,
                                      boilerplate_patterns_file=BOILERPLATE_PATTERNS_FILE)

    if preprocessed_df is not None:
        print("\nSample of preprocessed data:")
        # Display only relevant columns for verification
        display_columns = ['Product', 'Consumer complaint narrative', 'cleaned_Consumer complaint narrative']
        # Check if original narrative column exists before trying to display it
        display_columns = [col for col in display_columns if col in preprocessed_df.columns]
        print(preprocessed_df[display_columns].head())