import pandas as pd
from src.features import BINARY_COLUMNS

def preprocess(customer_data, feature_names):

    df = pd.DataFrame([customer_data])

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    df.replace({
        "No internet service": "No",
        "No phone service": "No"
    }, inplace=True)

    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({
                "No": 0,
                "Yes": 1,
                "Female": 0,
                "Male": 1
            })

    df = pd.get_dummies(
        df,
        columns=[
            "InternetService",
            "Contract",
            "PaymentMethod"
        ],
        drop_first=True
    )

    df = df.reindex(columns=feature_names, fill_value=0)

    return df