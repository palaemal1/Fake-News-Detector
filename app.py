

import streamlit as st
import requests
import pandas as pd

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Misinformation Detector",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# Custom Styling
# ============================================================

st.markdown("""
    <style>
    .custom-card {
        background-color: #1e1e2f;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }

    .metric-title {
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
    }
    
    .probability-container {
        background-color: #1e1e2f;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }

    .progress-bar-background {
        background-color: #33334d;
        border-radius: 8px;
        height: 24px;
        width: 100%;
        display: flex;
        overflow: hidden;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .progress-bar-real {
        background-color: #28a745;
        height: 100%;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 12px;
        line-height: 24px;
        transition: width 0.5s ease-in-out;
    }

    .progress-bar-fake {
        background-color: #dc3545;
        height: 100%;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 12px;
        line-height: 24px;
        transition: width 0.5s ease-in-out;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# Title
# ============================================================

st.title("🛡️ Multimodal Misinformation Detection System")

st.markdown(
    "Analyze text using Classical Ensemble and RoBERTa Transformer, "
    "or expand to full multimodal analysis (text, audio, video, images) using Groq Qwen LLM."
)

# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Configuration")

model_choice = st.sidebar.selectbox(
    "Select Model Architecture",
    options=["qwen", "transformer", "classical"],
    format_func=lambda x: {
        "qwen": "Groq Qwen LLM (Recommended)",
        "transformer": "XML-R Transformer",
        "classical": "Classical Ensemble"
    }[x]
)

st.sidebar.markdown("---")

if model_choice == "qwen":
    st.sidebar.info(
        "Upload media sources and/or enter text "
        "to begin multimodal misinformation analysis."
    )
else:
    st.sidebar.info(
        "Text-Only Mode: Enter news text or article context "
        "to begin analysis."
    )

# ============================================================
# Text Input
# ============================================================

text_input = st.text_area(
    "📝 Enter News Text or Article Context:",
    placeholder="Type or paste news content here...",
    height=300
)

# ============================================================
# Conditional File Uploaders (Only for Qwen)
# ============================================================

image_file = None
document_file = None
audio_file = None
video_file = None

if model_choice == "qwen":
    st.markdown("### 📂 Multimodal Upload Options")
    col_file1, col_file2, col_file3 = st.columns(3)

    with col_file1:
        image_file = st.file_uploader(
            "🖼️ Upload Image (Optional)",
            type=["jpg", "jpeg", "png", "webp"]
        )

    with col_file2:
        document_file = st.file_uploader(
            "📄 Upload Text Document (Optional)",
            type=["txt", "pdf", "docx"]
        )

    with col_file3:
        audio_file = st.file_uploader(
            "🎙️ Upload Audio (Optional)",
            type=["mp3", "wav", "m4a"]
        )

    video_file = st.file_uploader(
        "🎬 Upload Video (Optional)",
        type=["mp4", "mov", "avi"]
    )
else:
    st.info("ℹ️ **Text-Only Mode Enabled:** Classical and Transformer architectures analyze text input directly.")

# ============================================================
# API Configuration
# ============================================================

API_URL = "http://127.0.0.1:8000/predict_multimodal"

# ============================================================
# Run Analysis
# ============================================================

if st.button(
    "🚀 Run Analysis",
    type="primary",
    use_container_width=True
):

    if not text_input.strip() and not (image_file or document_file or audio_file or video_file):
        st.warning(
            "⚠️ Please provide valid input text "
            "(and/or media sources if using Qwen model)."
        )

    else:

        with st.spinner(
            f"Analyzing via {model_choice.upper()} model pipeline... "
            "Please wait."
        ):

            try:

                files = {}

                if image_file:
                    files["image"] = (
                        image_file.name,
                        image_file.getvalue(),
                        image_file.type
                    )

                if document_file:
                    files["file"] = (
                        document_file.name,
                        document_file.getvalue(),
                        document_file.type
                    )

                if audio_file:
                    files["audio"] = (
                        audio_file.name,
                        audio_file.getvalue(),
                        audio_file.type
                    )

                if video_file:
                    files["video"] = (
                        video_file.name,
                        video_file.getvalue(),
                        video_file.type
                    )

                data = {
                    "text": text_input,
                    "model_type": model_choice
                }

                response = requests.post(
                    API_URL,
                    data=data,
                    files=files if files else None,
                    timeout=300
                )

                if response.status_code == 200:

                    result = response.json()

                    st.success(
                        "✅ Analysis Completed Successfully!"
                    )

                    # =================================================
                    # Top-Level Metrics & Dual Probability Bar
                    # =================================================

                    verdict = result.get(
                        "verdict",
                        "Unverified"
                    )

                    confidence = result.get(
                        "confidence_score",
                        0.0
                    )

                    if confidence <= 1.0:
                        confidence_pct = confidence * 100
                    else:
                        confidence_pct = confidence

                    is_real = verdict.lower() in ["reliable", "real"]
                    real_prob = confidence_pct if is_real else (100 - confidence_pct)
                    fake_prob = (100 - confidence_pct) if is_real else confidence_pct

                    m_col1, m_col2 = st.columns(2)

                    with m_col1:
                        if is_real:
                            st.metric(
                                label="🎯 Final Verdict",
                                value=verdict,
                                delta="Trusted Source",
                                delta_color="normal"
                            )
                        else:
                            st.metric(
                                label="🎯 Final Verdict",
                                value=verdict,
                                delta="Potential Risk",
                                delta_color="inverse"
                            )

                    with m_col2:
                        st.metric(
                            label="📊 Confidence Score",
                            value=f"{confidence_pct:.1f}%"
                        )

                    st.markdown(
                        f"""
                        <div class="probability-container">
                            <div style="display: flex; justify-content: space-between; font-weight: bold; margin-bottom: 5px;">
                                <span style="color: #28a745;">🟢 Reliable / Real: {real_prob:.1f}%</span>
                                <span style="color: #dc3545;">🔴 Fake / Risk: {fake_prob:.1f}%</span>
                            </div>
                            <div class="progress-bar-background">
                                <div class="progress-bar-real" style="width: {real_prob}%;">{' ' if real_prob < 10 else f'{real_prob:.1f}%'}</div>
                                <div class="progress-bar-fake" style="width: {fake_prob}%;">{' ' if fake_prob < 10 else f'{fake_prob:.1f}%'}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown("---")

                    # =================================================
                    # Qwen Image Analysis
                    # =================================================

                    image_analysis = result.get("image_analysis")

                    if image_analysis:

                        st.subheader("🖼️ Image Analysis")

                        if image_analysis.get("present", False):

                            img_col1, img_col2 = st.columns(2)

                            with img_col1:
                                st.write("**Image Description**")
                                st.info(
                                    image_analysis.get(
                                        "description",
                                        "No description returned."
                                    )
                                )

                            with img_col2:
                                supports = image_analysis.get("supports_text")
                                contradicts = image_analysis.get("contradicts_text")

                                st.write("**Relationship with News Text**")

                                if supports is True:
                                    st.success("✅ Image supports the news text.")
                                elif supports is False:
                                    st.warning("⚠️ Image does not clearly support the news text.")

                                if contradicts is True:
                                    st.error("❌ Image contradicts the news text.")
                                elif contradicts is False:
                                    st.success("✅ No direct contradiction detected.")

                            visual_concerns = image_analysis.get("visual_concerns", [])
                            if visual_concerns:
                                st.write("**⚠️ Visual Concerns**")
                                for concern in visual_concerns:
                                    st.warning(str(concern))

                        else:
                            st.info("No image was provided for analysis.")

                    # =================================================
                    # Explanation
                    # =================================================

                    st.subheader("💡 Model Explanation & Reasoning")

                    st.info(
                        result.get(
                            "explanation",
                            "No detailed explanation returned from API."
                        )
                    )

                    # =================================================
                    # Reliability Breakdown
                    # =================================================

                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:

                        st.subheader("📊 Reliability Score Breakdown")
                        st.markdown('<div class="custom-card">', unsafe_allow_html=True)

                        reliability_breakdown = result.get("reliability_breakdown", {})

                        if reliability_breakdown:
                            for metric_name, score in reliability_breakdown.items():
                                try:
                                    score = float(score)
                                    st.write(f"**{metric_name}** — Score: `{score:.1f}/10`")
                                    st.progress(min(max(int((score / 10.0) * 100), 0), 100))
                                except (ValueError, TypeError):
                                    st.write(f"**{metric_name}** — {score}")
                        else:
                            st.info("No reliability breakdown metrics found.")

                        st.markdown('</div>', unsafe_allow_html=True)

                    # =================================================
                    # Content Analysis Balance
                    # =================================================

                    with chart_col2:

                        st.subheader("📈 Content Analysis Balance")
                        st.markdown('<div class="custom-card">', unsafe_allow_html=True)

                        analysis_data = result.get("content_analysis_balance", [])

                        if analysis_data:
                            df_chart = pd.DataFrame(analysis_data)
                            if not df_chart.empty and "Indicator" in df_chart.columns:
                                df_chart = df_chart.set_index("Indicator")
                                st.bar_chart(df_chart)
                            else:
                                st.warning("Malformed content analysis chart data.")
                        else:
                            st.info("No content analysis balance statistics found.")

                        st.markdown('</div>', unsafe_allow_html=True)

                    # =================================================
                    # SHAP Features & Raw JSON
                    # =================================================

                    if "shap_features" in result and result["shap_features"]:
                        with st.expander("🔍 View SHAP Feature Importances"):
                            shap_data = result["shap_features"]
                            if isinstance(shap_data, list) and shap_data and isinstance(shap_data[0], dict):
                                shap_df = pd.DataFrame(shap_data)
                                st.dataframe(shap_df, use_container_width=True)
                            else:
                                st.json(shap_data)

                    if model_choice == "qwen":
                        with st.expander("🔍 View Complete Qwen JSON Response"):
                            st.json(result)

                else:
                    try:
                        error_data = response.json()
                        error_message = error_data.get("detail", response.text)
                    except Exception:
                        error_message = response.text

                    st.error(f"❌ Server Error [{response.status_code}]: {error_message}")

            except requests.exceptions.ConnectionError:
                st.error("🔌 Connection Error: Could not connect to the FastAPI server. Make sure main.py is running on http://127.0.0.1:8000")

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Video/audio processing or Qwen analysis may require more time.")

            except Exception as e:
                st.error(f"⚠️ An unexpected client error occurred: {str(e)}")