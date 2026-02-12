import requests
import json
import uuid
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

API_URL = "http://127.0.0.1:5000/submit_message"

with open("evaluation_set.json") as f:
    test_data = json.load(f)

y_true = []
y_pred = []

labels = ["ASK_FOLLOWUP", "RETRIEVE_GUIDELINE", "ESCALATE"]

# Create CSV file and write header immediately
csv_file = open("evaluation_results.csv", "w", encoding="utf-8")
csv_file.write("input,expected_action,predicted_action,correct,reply\n")

print("Running evaluation...\n")

for item in test_data:
    user_id = "evaluation_user"

    response = requests.post(
        API_URL,
        json={
            "user_id": user_id,
            "message": item["input"]
        }
    )

    if response.status_code != 200:
        print("ERROR STATUS:", response.status_code)
        print("ERROR BODY:", response.text)
        continue

    try:
        result = response.json()
    except Exception:
        print("INVALID JSON RETURNED:")
        print(response.text)
        continue

    predicted = result.get("action", "UNKNOWN")
    expected = item["expected_action"]
    reply = result.get("reply", "").replace("\n", " ").replace(",", " ")

    y_true.append(expected)
    y_pred.append(predicted)

    correct = expected == predicted

    # Write immediately to CSV
    csv_file.write(
        f"\"{item['input']}\",{expected},{predicted},{correct},\"{reply}\"\n"
    )

    print(f"Input: {item['input']}")
    print(f"Expected: {expected} | Predicted: {predicted} | Correct: {correct}")
    print("-" * 50)

csv_file.close()

# ===== CONFUSION MATRIX =====

print("\n===== CONFUSION MATRIX =====")
cm = confusion_matrix(y_true, y_pred, labels=labels)

cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)

# Save confusion matrix to CSV
cm_df.to_csv("confusion_matrix.csv")

# ===== CLASSIFICATION REPORT =====

print("\n===== CLASSIFICATION REPORT =====")
report = classification_report(y_true, y_pred, labels=labels)
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)

# ===== SAFETY METRIC =====

escalation_should = sum(1 for x in y_true if x == "ESCALATE")
escalation_caught = sum(
    1 for i in range(len(y_true))
    if y_true[i] == "ESCALATE" and y_pred[i] == "ESCALATE"
)

escalation_recall = (
    escalation_caught / escalation_should
    if escalation_should > 0 else 0
)

print("\n===== SAFETY METRIC =====")
print("Escalation Recall:", escalation_recall)

# ===== PLOT CONFUSION MATRIX =====

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels,
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\nSaved:")
print("- evaluation_results.csv")
print("- confusion_matrix.csv")
print("- confusion_matrix.png")
print("- classification_report.txt")
