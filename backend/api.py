"""This module contains the API endpoints for the models in the app."""

import pickle
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from text_data import TextData


app = FastAPI()


@app.get("/")
def root():
    """This endpoint returns a welcome message."""
    return {"message": "Welcome to Medical Docs Classifier APP!"}


@app.post("/predict")
def predict(data: TextData):
    """This endpoint takes in a TextData object and returns a prediction."""
    # Load the model
    with open("../models/model.pkl", "rb") as f:
        model = pickle.load(f)

    # transform the data into a dataframe
    payload = jsonable_encoder(data)

    prediction = model.predict(payload)
    # probability = model.predict_proba(payload)

    # Return the predictions as a dictionary
    prediction = prediction[0].item()
    # probability = probability[0][1].item()

    result = {
        "prediction": prediction,
        # "probability": probability
    }

    return result


if __name__ == "__main__":
    uvicorn.run(
        app, host="127.0.0.1", port=8000, log_level="info", reload=True
    )
