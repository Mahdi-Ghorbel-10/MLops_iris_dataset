# webhook_trigger.py
from fastapi import FastAPI, Request, HTTPException
import uvicorn
import kfp
from kfp import Client

app = FastAPI()

# Replace <KFP_HOST> with your actual Kubeflow Pipelines endpoint
KFP_HOST = 'http://<KFP_HOST>'
client = Client(host=KFP_HOST)

# Define constants for pipeline triggering
PIPELINE_PACKAGE_PATH = 'iris_pipeline.yaml'
EXPERIMENT_NAME = 'Retraining_Experiment'

@app.post("/webhook")
async def webhook_trigger(request: Request):
    alert_payload = await request.json()
    print("Received alert payload:", alert_payload)
    
    # Check for the specific alert label (e.g., DataDriftDetected)
    alerts = alert_payload.get("alerts", [])
    for alert in alerts:
        if alert.get("labels", {}).get("alertname") == "DataDriftDetected":
            try:
                run = client.run_pipeline(
                    experiment_name=EXPERIMENT_NAME,
                    job_name='RetrainingJob',
                    pipeline_package_path=PIPELINE_PACKAGE_PATH,
                    params={}
                )
                print("Triggered retraining pipeline with run id:", run.id)
                return {"status": "success", "run_id": run.id}
            except Exception as e:
                print("Error triggering pipeline:", str(e))
                raise HTTPException(status_code=500, detail=str(e))
    
    return {"status": "ignored", "message": "No relevant alert found"}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)
