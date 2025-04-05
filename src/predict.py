import pandas as pd
from .model_training import load_model


def predict_yield(input_data, model_path='data/models/model.joblib'):
    # Load trained model
    model = load_model(model_path)

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)

    return prediction[0]