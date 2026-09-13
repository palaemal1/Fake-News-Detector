import os
import json
import torch
import shap
import tempfile
import torch.nn.functional as F
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
from openai import OpenAI
from moviepy import AudioFileClip
import base64
from pypdf import PdfReader
from docx import Document

try:
    from src.ensemble import ensemble_predict
    from src.shap_explain import explain_news
    from src.llama_explain import generate_explanation
except ImportError:
    ensemble_predict = None
    explain_news = None
    generate_explanation = None

app = FastAPI(
    title="Multimodal Misinformation Detection API",
    description="Multimodal detection supporting Classical Ensemble, XLM-R Transformer, and Groq Qwen LLM."
)
translator=GoogleTranslator(source='auto',target='en')

# 1. Load Local RoBERTa Model
#ROBERTA_PATH = "./models/ROBERTa_with_Epoch2" 
path="./models/XLM-R"

try:
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("✅ Local XLM-RoBERTa model loaded successfully.")
except Exception as e:
    print(f"⚠️ Could not load XLM-RoBERTa model: {e}")
    tokenizer = None
    model = None

def predict_proba(texts):
    if isinstance(texts,str):
        texts=[texts]

    inputs=tokenizer(
                    list(texts),
                    padding=True,
                    truncation=True,
                    max_length=256, 
                    return_tensors="pt"
                    ).to(device)
    
    with torch.no_grad():
        output=model(**inputs)
        probs=F.softmax(output.logits,dim=-1)
        # fake_prob = float(probs[0])
        # real_prob = float(probs[1])
    return probs.cpu().numpy()



def translate_to_english(text:str)->dict:
    if not text or not text.strip():
        return {
            "text":"",
            "language":"unknown",
            "translated":False
        }

    try:
        detected_language=detect(text)

    except LangDetectException:
        detected_language="unknown"


    if detected_language =="en":
        return{
            "text":text,
            "language":"en",
            "translated":False
        }

    try:
        translate_text_to_english=translator.translate(text)
        return {
            "text":translate_text_to_english,
            "language":detected_language,
            "translated":True
        }

    except Exception as e:
        print (f"Translation error : {e}")

        return {
            "text":text,
            "language":detected_language,
            "translated":False
        }

async def encode_image(image: Optional[UploadFile]):
    if not image:
        return None

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Supported images: JPG, PNG, WEBP"
        )

    image_bytes = await image.read()

    # 20 MB limit
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image must be smaller than 20 MB."
        )

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    return f"data:{image.content_type};base64,{encoded_image}"


async def extract_text_from_file(file: Optional[UploadFile]) -> str:
    if not file:
        return ""

    filename = (file.filename or "").lower()

    try:
        file_bytes = await file.read()

        # TXT
        if filename.endswith(".txt"):
            return file_bytes.decode("utf-8", errors="ignore")

        # PDF
        elif filename.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp:
                temp.write(file_bytes)
                temp_path = temp.name

            try:
                reader = PdfReader(temp_path)

                text = "\n".join(
                    page.extract_text() or ""
                    for page in reader.pages
                )

                return text

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        # DOCX
        elif filename.endswith(".docx"):
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".docx"
            ) as temp:
                temp.write(file_bytes)
                temp_path = temp.name

            try:
                document = Document(temp_path)

                text = "\n".join(
                    paragraph.text
                    for paragraph in document.paragraphs
                )

                return text

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        else:
            raise HTTPException(
                status_code=400,
                detail="Supported text files: .txt, .pdf, .docx"
            )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed reading file: {str(e)}"
        )


# 2. Setup Groq Client via Environment Variable
api_key = "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"

if api_key:
    try:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
    except Exception as e:
        print(f"Initialization Error: Could not configure Groq client wrapper. {e}")
        client = None
else:
    client = None
    print("⚠️ GROQ_API_KEY is missing from environment variables.")

# Helper Function: Media Transcription
async def extract_transcription_from_media(
    audio: Optional[UploadFile] = None, 
    video: Optional[UploadFile] = None
) -> str:
    if not client:
        raise HTTPException(status_code=500, detail="Groq client is not initialized on the server.")

    extracted_text = ""

    # Process Video
    if video:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(await video.read())
                temp_video_path = temp_video.name

            temp_audio_path = temp_video_path.replace(".mp4", ".mp3")

            audio_clip = AudioFileClip(temp_video_path)
            audio_clip.write_audiofile(temp_audio_path, logger=None)
            audio_clip.close()

            with open(temp_audio_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(temp_audio_path), audio_file.read()),
                    model="whisper-large-v3",
                    response_format="text"
                )
            extracted_text += f" [Transcribed Video Speech: {transcription}]"

            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed processing video: {str(e)}")

    # Process Audio
    if audio:
        try:
            audio_bytes = await audio.read()
            transcription = client.audio.transcriptions.create(
                file=(audio.filename or "audio.mp3", audio_bytes),
                model="whisper-large-v3",
                response_format="text"
            )
            extracted_text += f" [Transcribed Audio Context: {transcription}]"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed processing audio: {str(e)}")

    return extracted_text

# Helper Function: Qwen Direct Prediction with Dynamic Metrics
async def run_qwen_prediction(
    full_text: str,
    image_data: Optional[str] = None
) -> dict:

    if not client:
        raise HTTPException(
            status_code=500,
            detail="Groq client is not initialized."
        )

    system_prompt = """
You are an expert fact-checking AI and multimodal media literacy engine.

Analyze the provided news content using:
1. Textual information
2. Image information, if an image is provided
3. Consistency between the text and image
4. Linguistic patterns
5. Factual consistency
6. Sensationalism and clickbait
7. Possible fabricated or misleading claims
8. Visual manipulation or misleading visual context when detectable

IMPORTANT:
- An image being suspicious does NOT automatically mean the news is fake.
- Do not claim that an image is manipulated unless there is evidence visible in the image.
- If the image cannot establish factual truth, say so.
- Compare the image with the supplied text and identify contradictions or supporting evidence.

You MUST respond strictly in a valid JSON object.

Use exactly this structure:

{
  "prediction": "REAL",
  "confidence": 85.0,
  "model_score": 0.85,
  "explanation": "A concise explanation of the evaluation.",
  "image_analysis": {
    "present": true,
    "description": "Description of the image.",
    "supports_text": true,
    "contradicts_text": false,
    "visual_concerns": []
  },
  "reliability_breakdown": {
    "Source Reputation": 6.5,
    "Editorial Standards": 5.0,
    "Fact-Check Consistency": 7.2,
    "Absence of Bias": 4.8
  },
  "content_analysis_balance": [
    {
      "Indicator": "Clickbait",
      "True Likelihood": 15,
      "Fake Likelihood": 85
    },
    {
      "Indicator": "Sensationalist",
      "True Likelihood": 22,
      "Fake Likelihood": 78
    },
    {
      "Indicator": "Source Credibility",
      "True Likelihood": 65,
      "Fake Likelihood": 35
    },
    {
      "Indicator": "Fabricated Facts",
      "True Likelihood": 10,
      "Fake Likelihood": 90
    }
  ]
}

Rules:
- prediction must be either REAL or FAKE.
- confidence must be 0-100.
- model_score must be 0-1.
- reliability_breakdown values must be 0-10.
- likelihood values must be 0-100.
- image_analysis.present must be true only when an image was provided.
- image_analysis.supports_text must be true, false, or null.
- image_analysis.contradicts_text must be true, false, or null.
- Return ONLY valid JSON.
"""

    try:

        # -----------------------------
        # TEXT ONLY
        # -----------------------------

        if not image_data:

            user_content = [
                {
                    "type": "text",
                    "text": f"""
Analyze this news content:

{full_text}
"""
                }
            ]

        # -----------------------------
        # TEXT + IMAGE
        # -----------------------------

        else:

            user_content = [
                {
                    "type": "text",
                    "text": f"""
Analyze this news content together with the attached image.

NEWS TEXT:
{full_text}

Evaluate whether the image supports, contradicts, or provides misleading
context for the news text.
"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data
                    }
                }
            ]

        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],

            response_format={
                "type": "json_object"
            },

            reasoning_effort="none",
            temperature=0.1,
            max_tokens=800
        )

        raw_content = response.choices[0].message.content.strip()

        if not raw_content:
            raise HTTPException(
                status_code=502,
                detail="Qwen returned an empty response."
            )

        parsed_data = json.loads(raw_content)

        # Normalize verdict
        verdict_str = str(
            parsed_data.get("prediction", "FAKE")
        ).upper()

        normalized_verdict = (
            "Reliable"
            if verdict_str == "REAL"
            else "Misinformation"
        )

        parsed_data["verdict"] = normalized_verdict

        parsed_data["confidence_score"] = parsed_data.get(
            "model_score",
            parsed_data.get("confidence", 75) / 100.0
        )

        return parsed_data

    except json.JSONDecodeError as e:

        raise HTTPException(
            status_code=502,
            detail=f"Failed to parse Qwen output: {str(e)}"
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Groq API Error: {str(e)}"
        )

    
# Multimodal Endpoint
@app.post("/predict_multimodal")
async def predict_multimodal(
    text: str = Form(""),
    model_type: str = Form("qwen"),

    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),

    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None)
):
    
    transcribed_context = await extract_transcription_from_media(audio=audio, video=video)
    extracted_file_text = await extract_text_from_file(file)
    image_data = await encode_image(image)

    text_arr=[text.strip(),extracted_file_text.strip(),transcribed_context.strip()]

    full_text = full_text = "\n\n".join(
        x for x in text_arr if x
    )
    
    if not full_text and not image_data:
        raise HTTPException(status_code=400, detail="Provide at least text, an image, audio, or video.")

    translation_toenglish = translate_to_english(full_text)
    
    english_text = translation_toenglish["text"]
    source_language = translation_toenglish["language"]
    was_translated = translation_toenglish["translated"]

    selected_model = model_type.lower().strip()

    if selected_model == "classical":
        if ensemble_predict is None:
            raise HTTPException(status_code=500, detail="Classical ensemble dependencies (`src`) are not configured.")
        try:

            result = ensemble_predict(english_text)
            shap_features = explain_news(english_text) if explain_news else []
            explanation = generate_explanation(
                result["prediction"],
                result["confidence"],
                shap_features
            ) if generate_explanation else "Classical ensemble prediction completed."

            conf_score = round(result["confidence"] / 100.0, 2)
            is_real = result["prediction"] == "REAL"

            return {
                "verdict": "Reliable" if is_real else "Misinformation",
                "confidence_score": conf_score,
                "individual_models": result.get("individual_models", {}),
                "shap_features": shap_features,
                "explanation": explanation,
                "reliability_breakdown": {
                    "Source Reputation": round((conf_score if is_real else 1 - conf_score) * 8.5, 1),
                    "Editorial Standards": round((conf_score if is_real else 1 - conf_score) * 7.0, 1),
                    "Fact-Check Consistency": round((conf_score if is_real else 1 - conf_score) * 9.2, 1),
                    "Absence of Bias": round((conf_score if is_real else 1 - conf_score) * 6.5, 1)
                },
                "content_analysis_balance": [
                    {"Indicator": "Clickbait", "True Likelihood": int((1 - conf_score)*100 if is_real else conf_score*100), "Fake Likelihood": int(conf_score*100 if is_real else (1 - conf_score)*100)},
                    {"Indicator": "Sensationalist", "True Likelihood": int((1 - conf_score)*90), "Fake Likelihood": int(conf_score*90)},
                    {"Indicator": "Source Credibility", "True Likelihood": int(conf_score*100 if is_real else 30), "Fake Likelihood": int(30 if is_real else conf_score*100)},
                    {"Indicator": "Fabricated Facts", "True Likelihood": int((1 - conf_score)*100), "Fake Likelihood": int(conf_score*100)}
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Classical pipeline error: {str(e)}")

    elif selected_model == "transformer":
        if model is None or tokenizer is None:
            raise HTTPException(status_code=500, detail="XLM-RoBERTa model checkpoint is not loaded on server.")

        probabilities = predict_proba(english_text)[0]

        fake_prob = float(probabilities[0])
        real_prob = float(probabilities[1])
        
        masker=shap.maskers.Text(tokenizer)
        explainer=shap.Explainer(predict_proba,masker,output_names=["Fake News","Real News"])

        prediction = "Reliable" if real_prob >= fake_prob else "Misinformation"
        confidence = round(real_prob if prediction == "Reliable" else fake_prob, 2)

        try:
            masker = shap.maskers.Text(tokenizer)

            explainer = shap.Explainer(predict_proba,masker,output_names=["Fake News", "Real News"])

            shap_values = explainer([english_text])

        # Extract SHAP values for the input
            shap_result = []

            tokens = shap_values.data[0]
            values = shap_values.values[0]

            for i, token in enumerate(tokens):

                shap_result.append({
                                "token": str(token),
                                "fake_contribution": round(float(values[i][0]), 6),
                                "real_contribution": round(float(values[i][1]), 6)
                                })

        # Sort by strongest contribution
            shap_result = sorted(shap_result,key=lambda x: max(
                                                    abs(x["fake_contribution"]),
                                                    abs(x["real_contribution"])),
                            reverse=True)
        except Exception as e:
                print(f"SHAP error: {e}")
                shap_result = []


        return {
            "verdict": prediction,
            "confidence_score": confidence,
            #"individual_models": {"xml-roberta_transformer": round(real_prob, 4)},
            "individual_models": {
            "roberta_transformer": {
                "fake_probability": round(fake_prob, 4),
                "real_probability": round(real_prob, 4)
                }
            },
            "shap_features":shap_result[:20],
            #"explanation": f"RoBERTa sequence classifier predicted {prediction} with a confidence score of {confidence}.",
            "explanation": (
                    f"XLM-RoBERTa sequence classifier predicted "
                    f"{prediction} with a confidence score of {confidence}."
            ),
            "reliability_breakdown": {
                "Source Reputation": round(real_prob * 9.0, 1),
                "Editorial Standards": round(real_prob * 8.0, 1),
                "Fact-Check Consistency": round(real_prob * 8.5, 1),
                "Absence of Bias": round(real_prob * 7.5, 1)
            },
            "content_analysis_balance": [
                {"Indicator": "Clickbait", "True Likelihood": int(real_prob * 100), "Fake Likelihood": int(fake_prob * 100)},
                {"Indicator": "Sensationalist", "True Likelihood": int(real_prob * 85), "Fake Likelihood": int(fake_prob * 100)},
                {"Indicator": "Source Credibility", "True Likelihood": int(real_prob * 90), "Fake Likelihood": int(fake_prob * 90)},
                {"Indicator": "Fabricated Facts", "True Likelihood": int(real_prob * 95), "Fake Likelihood": int(fake_prob * 95)}
            ]
        }

    elif selected_model == "qwen":
        return await run_qwen_prediction(full_text,image_data=image_data)

    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model_type: '{selected_model}'. Supported choices are 'classical', 'transformer', or 'qwen'."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)