import json
import random

def analyze_vqa_mismatch():
    print("--- Analyzing VQA-RAD Question-Answer Mismatches ---")
    json_path = "data/VQA-RAD/dataset.json"
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("VQA-RAD dataset.json not found locally. Using mock samples for demonstration...")
        data = [
            {"num": 1, "question": "Is there cardiomegaly?", "answer": "The heart is enlarged.", "type": "CLOSED"},
            {"num": 2, "question": "What is the primary finding?", "answer": "yes", "type": "OPEN"},
            {"num": 3, "question": "Are the lungs clear?", "answer": "no", "type": "CLOSED"},
            {"num": 4, "question": "Is there a fracture?", "answer": "no evidence of fracture", "type": "CLOSED"}
        ] * 15
        
    random.shuffle(data)
    samples = data[:50]
    
    cleaned_data = []
    flags = 0
    
    print("\n[Flagged Mismatches]")
    for s in samples:
        q = s['question'].lower()
        a = str(s['answer']).lower()
        t = s.get('type', 'UNKNOWN').upper()
        
        is_yes_no_q = q.startswith(('is', 'are', 'does', 'do', 'has', 'have'))
        is_yes_no_a = a in ['yes', 'no']
        
        flagged = False
        reason = ""
        
        # Rule 1: Starts with Is/Are/Does but answer isn't exactly yes/no
        if is_yes_no_q and not is_yes_no_a:
            flagged = True
            reason = "Question expects Yes/No, answer is a phrase."
            # Quick fix for cleaned dataset
            if "no " in a or "not " in a or "negative" in a:
                s['answer'] = "no"
            else:
                s['answer'] = "yes"
                
        # Rule 2: Starts with What/Where/How but answer is yes/no
        elif not is_yes_no_q and is_yes_no_a:
            flagged = True
            reason = "Question asks What/Where, answer is Yes/No."
            # Unfixable automatically
            
        if flagged:
            flags += 1
            print(f"Q: {s['question']}")
            print(f"A: {a}")
            print(f"Reason: {reason}\n")
            
        cleaned_data.append(s)
        
    print(f"\nTotal Flagged in Sample: {flags}/50 ({flags/50*100:.1f}%)")
    
    out_path = "vqa_rad_cleaned.json"
    with open(out_path, 'w') as f:
        json.dump(cleaned_data, f, indent=4)
    print(f"Saved cleaned version (simulated) to {out_path}")

if __name__ == "__main__":
    analyze_vqa_mismatch()
