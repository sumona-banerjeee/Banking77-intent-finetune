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

def main():
    # Check if model directory exists
    model_dir = "models/distilbert-banking77"
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} does not exist!")
        print("Please train the model first by running: python src/train.py")
        return
    
    # Test texts
    #https://www.unitxt.ai/en/1.10.0/catalog/catalog.cards.banking77.html
    #Above link is for the generating the prediction and there are 77 such labeled data
    test_texts = [
        "How do I reset my card PIN?",
        "My card was stolen, block it now!",
        "What are the fees for international transfers?",
        "I want to increase my withdrawal limit",
        "My money is debited without any reason",
         "Can I open a savings account online?",
        "How do I change my registered mobile number?"
    ]
    
    print(f"Running inference on {len(test_texts)} texts...")
    print(f"Model directory: {model_dir}")
    print("-" * 60)
    
    try:
        results = predict(test_texts, model_dir)
        
        print(f"\nResults:")
        print("-" * 60)
        for result in results:
            print(f"Text: {result['text']}")
            print(f"Predicted: {result['predicted_label']} (confidence: {result['confidence']:.3f})")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()