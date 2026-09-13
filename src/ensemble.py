import joblib
import re
import string
def clean(text):
    text=text.lower()
    text=re.sub(r"\[.*?\]","",text)
    text=re.sub(r"\\W"," ",text)
    text=re.sub(r"https?://\S+|www\.\S+","",text)
    text=re.sub(r"<.*?>+","",text)
    text=re.sub(r"[%s]" % re.escape(string.punctuation),"",text)
    text=re.sub(r"\n","",text)
    text=re.sub(r"\w*\d\w*","",text)
    return text

print("Loading Classical Models...")

tfidf = joblib.load("./models/vectorizer.jb")
lr_model = joblib.load("./models/logistic_regression.jb")
rf_model = joblib.load("./models/random_forest.jb")
nb_model = joblib.load("./models/naive_bayes.jb")


print("Classical Models Loaded Successfully")


# ==========================================
# Classical Ensemble Weights (Sum = 1.0)
# ==========================================

# Logistic Regression & Random Forest often carry higher weights 
# due to superior calibration on TF-IDF vectors
LR_WEIGHT = 0.40
RF_WEIGHT = 0.40
NB_WEIGHT = 0.20


# ==========================================
# Prediction Function
# ==========================================

def get_classical_probs(text):
    vector = tfidf.transform([text])

    lr_prob = lr_model.predict_proba(vector)[0][1]
    rf_prob = rf_model.predict_proba(vector)[0][1]
    nb_prob = nb_model.predict_proba(vector)[0][1]

    return {
        "lr": lr_prob,
        "rf": rf_prob,
        "nb": nb_prob
    }


def ensemble_predict(news_text):
    cleaned_text = clean(news_text)
    probs = get_classical_probs(cleaned_text)

    # Weighted sum of probabilities
    final_score = (
        (LR_WEIGHT * probs["lr"]) +
        (RF_WEIGHT * probs["rf"]) +
        (NB_WEIGHT * probs["nb"])
    )

    prediction = "REAL" if final_score >= 0.5 else "FAKE"
    confidence = final_score * 100 if prediction == "REAL" else (1 - final_score) * 100

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "ensemble_score": round(final_score, 4),
        "individual_models": {
            "logistic_regression": round(probs["lr"], 4),
            "random_forest": round(probs["rf"], 4),
            "naive_bayes": round(probs["nb"], 4)
        }
    }


# ==========================================
# Testing Execution
# ==========================================

if __name__ == "__main__":
    sample_news = """
    Scientists discover breakthrough medical treatment 
    after multi-year successful clinical trials.
    """

    result = ensemble_predict(sample_news)

    print("\nClassical Ensemble Result")
    print("=" * 40)
    print(result)