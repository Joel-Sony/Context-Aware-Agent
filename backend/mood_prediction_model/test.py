from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer from your directory
model_path = "./"  # or use absolute path

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Function to predict mood
def predict_mood(text):
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    return predicted_class, confidence

label_map = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sadness",
    6: "shame",
    7: "surprise"
}

# Test it
user_input = "Its my exam tomorrow and i dont wanna write it"
mood_class, confidence = predict_mood(user_input)
print(f"Predicted mood class: {label_map[mood_class]}")
print(f"Confidence: {confidence:.2%}")