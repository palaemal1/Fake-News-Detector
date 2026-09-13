import json
import ollama

def generate_explanation(prediction, confidence, shap_features):
    
    feature_text = ", ".join([f"'{item['feature']}'" for item in shap_features])
    suggested_verdict = "Reliable" if prediction == "REAL" else "Misinformation"
    
    
    raw_conf = float(confidence)
    normalized_confidence = round(raw_conf / 100.0 if raw_conf > 1.0 else raw_conf, 2)

    system_prompt = """
    You are an expert fact-checking AI specialized in media literacy.
    Analyze the provided user text metrics for strong linguistic or structural markers of misinformation.

    You MUST respond strictly in a valid, clean JSON object with these exact keys:
    {
        "verdict": "Misinformation" or "Reliable" or "Unverified",
        "confidence_score": float,
        "explanation": "A concise 2-sentence breakdown explaining your structural evaluation."
    }
    Do not include any conversational filler words, Markdown code blocks, or trailing commentary.
    """

    user_prompt = f"""
    Analyze these machine learning classification findings:
    - Model Prediction Verdict: {prediction} (Suggested Mapping: {suggested_verdict})
    - Ensemble Confidence Score: {normalized_confidence}
    - High-Impact SHAP Features (Keywords): [{feature_text}]
    """

    response = ollama.chat(
        model="llama3.2",
        format="json",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ]
    )
    
    
    try:
        return json.loads(response["message"]["content"])
    except json.JSONDecodeError:
        return {
            "verdict": suggested_verdict,
            "confidence_score": normalized_confidence,
            "explanation": response["message"]["content"]
        }