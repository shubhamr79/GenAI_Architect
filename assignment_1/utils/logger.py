import mlflow
import os

def init_mlflow():
    # if not is_mlflow_running(os.getenv("MLFLOW_TRACKING_URI")):
    #     start_mlflow_server()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("Real-Time Sentiment Analyzer")

def start_run(company_name):
    return mlflow.start_run(run_name=f"Sentiment_Analysis_{company_name}")

def register_prompt(prompt_name: str, prompt_text: str, commit_message: str):
    prompt = mlflow.genai.register_prompt(
        name=prompt_name,
        template=prompt_text,
        commit_message=commit_message,
        tags={"type": "sentiment_analysis", "language": "en"},
    )
    print(f"Created prompt '{prompt.name}' (version {prompt.version})")
