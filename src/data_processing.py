import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib


def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_data(df):
    # Example preprocessing - adjust based on your actual data
    # Drop unnecessary columns
    df = df.drop(['Field_ID', 'Recorded_Date'], axis=1, errors='ignore')

    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))

    return df


def prepare_features_target(df, target_col='Yield'):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y


def get_preprocessor(X):
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Create transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


def save_preprocessor(preprocessor, filepath):
    joblib.dump(preprocessor, filepath)


def load_preprocessor(filepath):
    return joblib.load(filepath)