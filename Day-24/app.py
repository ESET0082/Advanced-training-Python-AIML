from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

# --- Load model and preprocessor ---
model = keras.models.load_model("final_model.keras")
preprocessor = joblib.load("preprocessor.pkl")  # You must save this after training

app = FastAPI()
templates = Jinja2Templates(directory="template")

# --- Define feature names (same as training) ---
features_num = [
    "lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "required_car_parking_spaces",
    "total_of_special_requests", "adr",
]

features_cat = [
    "hotel", "arrival_date_month", "meal", "market_segment",
    "distribution_channel", "reserved_room_type", "deposit_type",
    "customer_type",
]

# --- Route for HTML form ---
@app.get("/", response_class=HTMLResponse)
async def form_view(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --- Prediction route ---
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    lead_time: float = Form(0),
    arrival_date_week_number: int = Form(0),
    arrival_date_day_of_month: int = Form(0),
    stays_in_weekend_nights: int = Form(0),
    stays_in_week_nights: int = Form(0),
    adults: int = Form(1),
    children: float = Form(0),
    babies: int = Form(0),
    is_repeated_guest: int = Form(0),
    previous_cancellations: int = Form(0),
    previous_bookings_not_canceled: int = Form(0),
    required_car_parking_spaces: int = Form(0),
    total_of_special_requests: int = Form(0),
    adr: float = Form(0),
    hotel: str = Form("Resort Hotel"),
    arrival_date_month: str = Form("January"),
    meal: str = Form("BB"),
    market_segment: str = Form("Online TA"),
    distribution_channel: str = Form("TA/TO"),
    reserved_room_type: str = Form("A"),
    deposit_type: str = Form("No Deposit"),
    customer_type: str = Form("Transient"),
):
    try:
        # --- Create dataframe with user input ---
        input_data = pd.DataFrame([{
            "lead_time": lead_time,
            "arrival_date_week_number": arrival_date_week_number,
            "arrival_date_day_of_month": arrival_date_day_of_month,
            "stays_in_weekend_nights": stays_in_weekend_nights,
            "stays_in_week_nights": stays_in_week_nights,
            "adults": adults,
            "children": children,
            "babies": babies,
            "is_repeated_guest": is_repeated_guest,
            "previous_cancellations": previous_cancellations,
            "previous_bookings_not_canceled": previous_bookings_not_canceled,
            "required_car_parking_spaces": required_car_parking_spaces,
            "total_of_special_requests": total_of_special_requests,
            "adr": adr,
            "hotel": hotel,
            "arrival_date_month": arrival_date_month,
            "meal": meal,
            "market_segment": market_segment,
            "distribution_channel": distribution_channel,
            "reserved_room_type": reserved_room_type,
            "deposit_type": deposit_type,
            "customer_type": customer_type
        }])

        # --- Apply same preprocessing used in training ---
        X_processed = preprocessor.transform(input_data)

        # --- Make prediction ---
        prediction = model.predict(X_processed)
        prediction_class = (prediction > 0.5).astype("int")[0][0]
        result_text = "Booking Cancelled" if prediction_class == 1 else "Not Cancelled"

        # --- Return result page ---
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "prediction": float(prediction[0][0]),
                "prediction_class": int(prediction_class),
                "result_text": result_text,
                **input_data.iloc[0].to_dict()
            }
        )
    except Exception as e:
        return HTMLResponse(f"<h3>Error during prediction: {str(e)}</h3>")

# # --- Run locally ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
