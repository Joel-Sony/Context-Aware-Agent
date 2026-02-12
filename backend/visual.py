import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load your data
# Ensure your CSV is in the same folder as this script
df = pd.read_csv('results.csv')

# 2. Standardize 'expected' column (Cleaning typos like 'RETRIVE' or 'ESC_ALATE')
standardization_map = {
    'ASK_FOLOWUP': 'ASK_FOLLOWUP',
    'CHATT': 'CHAT',
    'RETRIVE_GUIDELINE': 'RETRIEVE_GUIDELINE',
    'ESC_ALATE': 'ESCALATE'
}
df['expected_clean'] = df['expected'].replace(standardization_map)

# 3. Create the Matrix
labels = sorted(df['predicted'].unique())
cm = confusion_matrix(df['expected_clean'], df['predicted'], labels=labels)

# 4. Plotting
plt.figure(figsize=(10, 8))
sns.set_theme(style="white") # Clean background

sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=labels, 
    yticklabels=labels,
    cbar_kws={'label': 'Count of Instances'}
)

plt.title('Triage Agent Evaluation: Confusion Matrix', fontsize=15)
plt.xlabel('Predicted Action (Model Output)', fontsize=12)
plt.ylabel('Expected Action (Ground Truth)', fontsize=12)
plt.tight_layout()

# 5. SAVE THE PNG
# This will save 'confusion_matrix.png' in your current directory
plt.savefig('confusion_matrix.png', dpi=300)
print("Confusion matrix saved as 'confusion_matrix.png'")

plt.show()

# 6. Quick Analysis Report
print("\n--- Evaluation Report ---")
print(classification_report(df['expected_clean'], df['predicted'], target_names=labels))