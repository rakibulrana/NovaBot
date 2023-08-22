import pandas as pd


def preprocess_csv(file_path):
    # Data preprocessing logic - You can add your preprocessing steps here
    df = pd.read_csv(file_path)
    # Example preprocessing: Convert all column names to lowercase
    df.columns = df.columns.str.lower()

    # Return the preprocessed data (you can return any other relevant result)
    return df.to_html()
