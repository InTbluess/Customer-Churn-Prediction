from model_loader import load_model, load_feature_names
from src.preprocessing import preprocess


model = load_model()
feature_names = load_feature_names()


def predict(customer_data):

    processed = preprocess(customer_data, feature_names)

    prediction = model.predict(processed)[0]

    probability = model.predict_proba(processed)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4)
    }