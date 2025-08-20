import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict(texts, model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Try to load tokenizer from model directory, fallback to base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print(f"Loaded tokenizer from: {model_dir}")
    except Exception as e:
        print(f"Failed to load tokenizer from {model_dir}")
        print(f"Error: {e}")
        print("Falling back to base model tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Load the fine-tuned model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        print(f"Loaded model from: {model_dir}")
    except Exception as e:
        print(f"Failed to load model from {model_dir}")
        raise e
    
    model.eval()

    # Load label names
    label_path = os.path.join(model_dir, "label_names.txt")
    if os.path.exists(label_path):
        with open(label_path) as f:
            label_names = [line.strip() for line in f]
        print(f"Loaded {len(label_names)} label names")
    else:
        print(f"No label_names.txt found in {model_dir}")
        label_names = None

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        confs = torch.max(probs, dim=-1)[0]

    results = []
    for i, text in enumerate(texts):
        pred_id = preds[i].item()
        conf = confs[i].item()
        pred_label = label_names[pred_id] if label_names else f"label_{pred_id}"
        
        results.append({
            "text": text,
            "predicted_label": pred_label,
            "confidence": round(conf, 4),
            "label_id": pred_id
        })
    
    return results

def interactive_predict(model_dir):
    """Interactive prediction function that loads model once and keeps it in memory"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #print(f"Interactive prediction...")
    #print(f"Model directory: {model_dir}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        #print(f"Loaded tokenizer from: {model_dir}")
    except Exception as e:
        print(f"Failed to load tokenizer from {model_dir}")
        print(f"Error: {e}")
        print("Falling back to base model tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Load model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        #print(f"Loaded model from: {model_dir}")
    except Exception as e:
        print(f"Failed to load model from {model_dir}")
        raise e
    
    model.eval()

    # Load label names
    label_path = os.path.join(model_dir, "label_names.txt")
    if os.path.exists(label_path):
        with open(label_path) as f:
            label_names = [line.strip() for line in f]
        #print(f"Loaded {len(label_names)} label names")
    else:
        #print(f"No label_names.txt found in {model_dir}")
        label_names = None
    
    #print("\n" + "="*60)
    #print("Interactive Banking Intent Classification")
    #print("="*60)
    print("Type your banking query and press Enter")
    print("Type 'q' to quit")
    print(" " * 60)
    
    while True:
        try:
            # Get user input
            user_input = input("\nPlease enter your sentence: ").strip()
            
            # Check if user wants to quit
            if user_input.lower() == 'q':
                print("Goodbye!")
                break
            
            # Skip empty input
            if not user_input:
                print("Please enter a valid sentence")
                continue
            
            # Make prediction
            inputs = tokenizer([user_input], padding=True, truncation=True, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred_id = torch.argmax(logits, dim=-1)[0].item()
                confidence = torch.max(probs, dim=-1)[0][0].item()
            
            # Format output
            pred_label = label_names[pred_id] if label_names else f"label_{pred_id}"
            
            # Display result
            print(f"\nYou: {user_input}")
            print(f"[{confidence:.3f}] Predicted intent: {pred_label}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error processing input: {e}")
            continue

def main():
    # Check if model directory exists
    model_dir = "models/distilbert-banking77"
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} does not exist!")
        print("Please train the model first by running: python src/train.py")
        return
    
    try:
        interactive_predict(model_dir)
    except Exception as e:
        print(f"Error during setup: {e}")

if __name__ == "__main__":
    main()