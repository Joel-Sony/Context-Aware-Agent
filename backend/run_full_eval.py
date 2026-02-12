import requests
import json
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Configuration
API_URL = "http://127.0.0.1:5000/submit_message"
TEST_DATA_PATH = "evaluation_set.json"
LABELS = ["CHAT", "ASK_FOLLOWUP", "RETRIEVE_GUIDELINE", "ESCALATE"]
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = f"evaluation_results_{TIMESTAMP}.csv"

def run_industrial_evaluation():
    try:
        with open(TEST_DATA_PATH) as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"File Error: {e}")
        return

    # Prepare the CSV with headers immediately
    header = ["timestamp", "input", "expected", "predicted", "latency_sec", "risk_level", "correct", "reply"]
    pd.DataFrame(columns=header).to_csv(CSV_FILENAME, index=False)

    results = []
    session = requests.Session()

    print(f"Starting 1,000-case run. CSV saving to: {CSV_FILENAME}")
    
    for item in tqdm(test_data, desc="Evaluating"):
        start_time = time.time()
        payload = {"user_id": "eval_user_01", "message": item["input"]}
        
        expected = item["expected_action"].upper()
        predicted = "UNKNOWN"
        reply = ""
        latency = 0

        try:
            response = session.post(API_URL, json=payload, timeout=60)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                predicted = str(data.get("action", "UNKNOWN")).upper()
                reply = data.get("reply", "")
            else:
                predicted = f"HTTP_{response.status_code}"
        except Exception as e:
            predicted = "SYSTEM_ERROR"
            reply = str(e)

        # Create row for this specific result
        row = {
            "timestamp": datetime.now().isoformat(),
            "input": item["input"],
            "expected": expected,
            "predicted": predicted,
            "latency_sec": round(latency, 2),
            "risk_level": item.get("risk_level", item.get("expected_risk", "N/A")),
            "correct": expected == predicted,
            "reply": reply.replace("\n", " ")
        }
        
        results.append(row)
        
        # FAIL-SAFE: Append this single row to the CSV immediately
        pd.DataFrame([row]).to_csv(CSV_FILENAME, mode='a', header=False, index=False)

    # ===== POST-RUN ANALYSIS =====
    df = pd.DataFrame(results)
    df_valid = df[df['predicted'].isin(LABELS)].copy()
    
    if df_valid.empty:
        print("No valid predictions were made. Check the CSV for error codes.")
        return

    # 1. Summary Metrics
    acc = accuracy_score(df_valid['expected'], df_valid['predicted'])
    
    # 2. Risk-Based Performance (Crucial for Triage)
    risk_acc = df_valid.groupby('risk_level')['correct'].mean().to_dict()

    # 3. Full Report
    full_report = classification_report(df_valid['expected'], df_valid['predicted'], 
                                       labels=LABELS, output_dict=True, zero_division=0)

    # Save Permanent Summary JSON
    summary = {
        "overall_accuracy": round(acc, 4),
        "avg_latency": round(df['latency_sec'].mean(), 2),
        "accuracy_by_risk": risk_acc,
        "classification_report": full_report
    }
    
    with open(f"summary_metrics_{TIMESTAMP}.json", "w") as f:
        json.dump(summary, f, indent=4)

    # ===== VISUALIZATIONS =====
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(df_valid['expected'], df_valid['predicted'], labels=LABELS)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=LABELS, yticklabels=LABELS, cmap="Blues")
    plt.title(f"Confusion Matrix (Overall Acc: {acc:.2%})")
    plt.savefig(f"confusion_matrix_{TIMESTAMP}.png")

    # Plot 2: Latency Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['latency_sec'], bins=30, kde=True, color='purple')
    plt.title("API Latency Distribution (Seconds per Request)")
    plt.savefig(f"latency_distribution_{TIMESTAMP}.png")

    print(f"\nEvaluation Complete. Final Report generated in summary_metrics_{TIMESTAMP}.json")

if __name__ == "__main__":
    run_industrial_evaluation()