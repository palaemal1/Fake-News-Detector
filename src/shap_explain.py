import joblib
import numpy as np
import shap

tfidf = joblib.load("./models/vectorizer.jb")
lr_model = joblib.load("./models/logistic_regression.jb")

explainer = shap.LinearExplainer(lr_model, masker=shap.maskers.Independent(data=np.zeros((1, len(tfidf.get_feature_names_out())))))

def explain_news(news_text, top_n=10):
    vector = tfidf.transform([news_text])
    shap_values = explainer.shap_values(vector)
    feature_names = tfidf.get_feature_names_out()
    
    
    values = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
    
   
    top_idx = np.argsort(np.abs(values))[-top_n:][::-1]
    
    return [
        {
            "feature": feature_names[idx],
            "impact": float(values[idx])
        }
        for idx in top_idx
    ]