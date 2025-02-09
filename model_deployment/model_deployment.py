# model_deployment.py
from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager


app = FastAPI()

# Global model variable
model = None

# Lifespan handler to load the model at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load('model_taining/model/best_model.pkl')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
    yield  # This allows the app to run
    # (Optional) Cleanup code can be placed here if needed when the app shuts down

# Attach the lifespan handler to the app
app = FastAPI(lifespan=lifespan)
# Define the expected JSON payload using Pydantic
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(iris: IrisInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    features = [[
        iris.sepal_length,
        iris.sepal_width,
        iris.petal_length,
        iris.petal_width
    ]]
    
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
