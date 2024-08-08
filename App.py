# App.py
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load the trained model
model = joblib.load('best_xgboost_model.pkl')

# Initialize FastAPI
app = FastAPI()

# Define the request body for the prediction endpoint
class ClaimPredictionRequest(BaseModel):
    Provider_x: str
    PotentialFraud: str
    InscClaimAmtReimbursed_x: float
    ClmProcedureCode_1_x: float
    ClmProcedureCode_2_x: float
    ClmProcedureCode_3_x: float
    DeductibleAmtPaid_x: float
    Gender: float
    Race: float
    State: float
    County: float
    NoOfMonths_PartACov: float
    NoOfMonths_PartBCov: float
    ChronicCond_Alzheimer: float
    ChronicCond_Heartfailure: float
    ChronicCond_KidneyDisease: float
    ChronicCond_Cancer: float
    ChronicCond_ObstrPulmonary: float
    ChronicCond_Depression: float
    ChronicCond_Diabetes: float
    ChronicCond_IschemicHeart: float
    ChronicCond_Osteoporasis: float
    ChronicCond_rheumatoidarthritis: float
    ChronicCond_stroke: float
    IPAnnualReimbursementAmt: float
    IPAnnualDeductibleAmt: float
    OPAnnualReimbursementAmt: float
    OPAnnualDeductibleAmt: float
    ClmProcedureCode_1: float
    ClmProcedureCode_2: float
    ClmProcedureCode_3: float
    DeductibleAmtPaid: float

# Define the response body for the prediction endpoint
class ClaimPredictionResponse(BaseModel):
    predicted_amount: float

@app.post("/predict", response_model=ClaimPredictionResponse)
def predict_claim_amount(request: ClaimPredictionRequest):
    try:
        # Convert the input data to a DataFrame
        input_data = pd.DataFrame([request.dict()])

        # Ensure the input data has the correct columns and order
        expected_features = [
            "Provider_x", "PotentialFraud", "InscClaimAmtReimbursed_x", "ClmProcedureCode_1_x",
            "ClmProcedureCode_2_x", "ClmProcedureCode_3_x", "DeductibleAmtPaid_x", "Gender",
            "Race", "State", "County", "NoOfMonths_PartACov", "NoOfMonths_PartBCov",
            "ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease",
            "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
            "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis",
            "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke", "IPAnnualReimbursementAmt",
            "IPAnnualDeductibleAmt", "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt",
            "ClmProcedureCode_1", "ClmProcedureCode_2", "ClmProcedureCode_3", "DeductibleAmtPaid"
        ]

        # Drop any extra columns and reorder columns to match expected features
        input_data = input_data[expected_features]

        # Make a prediction
        prediction = model.predict(input_data)

        # Return the prediction as the response
        return ClaimPredictionResponse(predicted_amount=prediction[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Health Claims Prediction API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
