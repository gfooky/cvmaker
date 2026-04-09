import streamlit as st
import pandas as pd
import datetime
import os
import sys
import json
import re
import glob
import base64
import subprocess

from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==========================================
# 1. CONFIGURATION
# ==========================================
API_KEY = os.getenv("GEMINI_API_KEY")
CRM_FILE = "my_job_crm.csv"
MASTER_CV_FILE = "master-cv.yaml"

LOCAL_SCORE_THRESHOLD = 20     # Minimum TF-IDF similarity (%) to pass pre-filter
AI_SCORE_THRESHOLD = 70        # Minimum AI fit score (%) to generate the CV

client = genai.Client(api_key=API_KEY)

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

def read_cv_corpus() -> str:
    """Reads the synonym file for the TF-IDF filter."""
    try:
        with open("cv_corpus.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # If it does not find the file, fallback to the original Master CV
        return read_master_cv()
    
def display_pdf(file_path: str) -> None:
    """Renders a PDF file inline in the Streamlit app using an iframe."""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_html = (
        f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
        f'width="100%" height="800px" type="application/pdf"></iframe>'
    )
    st.markdown(pdf_html, unsafe_allow_html=True)


def load_crm() -> pd.DataFrame:
    """Loads the CRM CSV file, or returns an empty DataFrame if it doesn't exist."""
    if os.path.exists(CRM_FILE):
        return pd.read_csv(CRM_FILE)
    return pd.DataFrame(
        columns=["Date", "Company", "Local Fit (%)", "AI Fit (%)", "Status", "Link/Note"]
    )


def save_to_crm(new_entry: dict) -> None:
    """Appends a new job entry to the CRM CSV file."""
    df = load_crm()
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(CRM_FILE, index=False)


def read_master_cv() -> str | None:
    """Reads the master CV YAML file and returns its content as a string."""
    try:
        with open(MASTER_CV_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


# ==========================================
# 3. CORE ENGINE (LOCAL + AI)
# ==========================================

def local_pre_filter(cv_text: str, job_text: str) -> float:
    """
    Calculates TF-IDF cosine similarity between the CV and the job description.
    Returns a similarity score from 0.0 to 100.0.

    Note: stop_words is intentionally set to None to support non-English content.
    """
    def clean(text: str) -> str:
        return re.sub(r"[^\w\s]", " ", text).lower()

    cv_clean = clean(cv_text)
    job_clean = clean(job_text)

    vectorizer = TfidfVectorizer(stop_words=None)
    try:
        tfidf_matrix = vectorizer.fit_transform([cv_clean, job_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return round(float(similarity[0][0]) * 100, 2)
    except Exception:
        return 0.0


def build_prompt(job_description: str, master_cv_text: str, score_threshold: int) -> str:
    return f"""
You are a technical recruiter and ATS specialist.
Analyze the fit (0–100) between the job description and my CV.
If the fit >= {score_threshold}, return an adapted CV (highly ATS-optimized, mirroring the job's terminology).
If not, return null. NEVER invent or fabricate data.

CRITICAL REQUIREMENTS:
1. Identify the language of the JOB DESCRIPTION.
2. Translate and adapt all the generated CV content to match the JOB DESCRIPTION language exactly.
3. Keep the original YAML structure intact.
4. Always copy the "design:" block exactly as it appears at the end of MY CV. Do not omit it.

Return a JSON object: {{"percentual_fit": int, "reason": str, "adapted_cv_yaml": str or null}}

JOB DESCRIPTION:
{job_description}

MY CV:
{master_cv_text}
"""


def strip_markdown_fences(text: str) -> str:
    """Removes markdown code fences (```json ... ```) from a string if present."""
    text = text.strip()
    fence = "```"
    if text.startswith(fence):
        lines = text.splitlines()
        # Remove opening fence line (e.g. ```json)
        if lines[0].startswith(fence):
            lines = lines[1:]
        # Remove closing fence line
        if lines and lines[-1].strip() == fence:
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def evaluate_and_generate_ai(
    job_text: str, master_cv_text: str, score_threshold: int
) -> dict:
    """
    Calls the Gemini API to assess CV fit and generate an ATS-optimized CV.
    Returns a parsed JSON dict with keys: percentual_fit, reason, adapted_cv_yaml.
    """
    prompt = build_prompt(job_text, master_cv_text, score_threshold)

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    raw_text = strip_markdown_fences(response.text)

    try:
        # strict=False allows embedded newlines inside JSON strings (needed for YAML content)
        return json.loads(raw_text, strict=False)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing failed: {e}")
        print(f"[DEBUG] Raw response:\n{raw_text}")
        raise


def render_cv_and_save(
    yaml_content: str,
    company_name: str,
    job_link: str,
    local_fit: float,
    ai_fit: int | str,
) -> None:
    """
    Saves the adapted CV as a YAML file, compiles it to PDF via RenderCV,
    displays the result in the UI, and logs the entry to the CRM.
    """
    # Safety net: if the AI omitted the design block, inject a default
    if "design:" not in yaml_content:
        yaml_content += "\n\ndesign:\n  theme: engineeringresumes\n"

    safe_company_name = re.sub(r"[^\w]", "_", company_name).lower()
    yaml_filename = f"cv_{safe_company_name}.yaml"

    with open(yaml_filename, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    with st.expander("📄 View compiled YAML source"):
        st.code(yaml_content, language="yaml")

    with st.spinner("🎨 Running RenderCV to compile the PDF..."):
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            # Use sys.executable to ensure the correct Python interpreter is used
            result = subprocess.run(
                [sys.executable, "-m", "rendercv", "render", yaml_filename],
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
            )

            if result.returncode == 0:
                st.success("✨ PDF generated successfully!")
                pdf_files = glob.glob("rendercv_output/*.pdf")

                if pdf_files:
                    # Use modification time (mtime) for reliability across all OS
                    latest_pdf = max(pdf_files, key=os.path.getmtime)
                    st.subheader(f"CV ready for {company_name}:")
                    display_pdf(latest_pdf)

                    with open(latest_pdf, "rb") as f:
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=f,
                            file_name=f"CV_{company_name}_Adapted.pdf",
                            mime="application/pdf",
                        )
                else:
                    st.warning("⚠️ PDF not found in the output folder.")
            else:
                st.error("❌ RenderCV encountered an error in the YAML.")
                with st.expander("View RenderCV error log"):
                    st.code(result.stderr)

        except FileNotFoundError:
            st.error("🚨 RenderCV command not found. Is it installed? (`pip install rendercv[full]`)")

    save_to_crm(
        {
            "Date": datetime.date.today().strftime("%Y-%m-%d"),
            "Company": company_name,
            "Local Fit (%)": local_fit,
            "AI Fit (%)": ai_fit,
            "Status": "PDF Generated",
            "Link/Note": job_link,
        }
    )
    st.balloons()


# ==========================================
# 4. USER INTERFACE
# ==========================================
st.set_page_config(page_title="CV_maker", layout="wide")
st.title("🚀 CV-Maker — Auto-Apply & CV Optimizer")

master_cv_text = read_master_cv()
if not master_cv_text:
    st.error(
        f"⚠️ Master CV not found. Please create a file named '{MASTER_CV_FILE}' "
        "in the same directory as this script."
    )
    st.stop()

tab_analyze, tab_crm = st.tabs(["🔍 Analyze New Job", "📊 My Pipeline"])

# ------------------------------------------
# TAB 1: Job Analysis & CV Generation
# ------------------------------------------
with tab_analyze:
    with st.form("job_form"):
        col1, col2 = st.columns(2)
        with col1:
            company_name = st.text_input("Company Name")
        with col2:
            job_link = st.text_input("Job Link (optional)")
        job_description = st.text_area("Paste the job description here:", height=150)
        submit_btn = st.form_submit_button("Analyze Fit & Generate CV")

    if submit_btn and job_description and company_name:
        # Loads the enriched corpus (PT + EN) instead of the master in YAML
        cv_corpus_text = read_cv_corpus()
        
        # TF-IDF now compares the job with words in both languages
        local_fit = local_pre_filter(cv_corpus_text, job_description)

        if local_fit < LOCAL_SCORE_THRESHOLD:
            st.error(
                f"❌ Job rejected at pre-filter stage. "
                f"Fit: {local_fit}% (Minimum: {LOCAL_SCORE_THRESHOLD}%)"
            )
        else:
            st.success(f"✅ Passed local pre-filter ({local_fit}%)! Waking up the AI...")

            with st.spinner("AI is analyzing the job description word by word..."):
                try:
                    ai_result = evaluate_and_generate_ai(
                        job_description, master_cv_text, AI_SCORE_THRESHOLD
                    )
                    ai_fit = ai_result.get("percentual_fit", 0)
                    reason = ai_result.get("reason", "No reason provided.")
                    adapted_yaml = ai_result.get("adapted_cv_yaml")

                    if ai_fit >= AI_SCORE_THRESHOLD:
                        st.success(f"**AI Fit Score: {ai_fit}%** — {reason}")
                        render_cv_and_save(
                            adapted_yaml, company_name, job_link, local_fit, ai_fit
                        )
                    else:
                        st.warning(
                            f"⚠️ Job rejected by AI. Fit: {ai_fit}% "
                            f"(Minimum: {AI_SCORE_THRESHOLD}%) — {reason}"
                        )

                except Exception as e:
                    st.error(f"🚨 API call failed: {e}")
                    st.info(
                        "💡 **Plan B activated:** Copy the prompt below, paste it into "
                        "any AI chat interface, then paste the full JSON response into "
                        "the manual form at the bottom of this page."
                    )

                    st.session_state["failed_company"] = company_name
                    st.session_state["failed_link"] = job_link
                    st.session_state["failed_fit"] = local_fit

                    emergency_prompt = build_prompt(
                        job_description, master_cv_text, AI_SCORE_THRESHOLD
                    )
                    st.code(emergency_prompt, language="markdown")

    # ------------------------------------------
    # FALLBACK: Manual Plan B Form
    # ------------------------------------------
    st.divider()
    st.subheader("🛠️ Plan B: Manual Generator")
    st.write(
        "Use this form if the API above failed. "
        "Paste the full JSON response you got from an external AI here."
    )

    with st.form("manual_form"):
        col1_m, col2_m = st.columns(2)
        with col1_m:
            manual_company = st.text_input(
                "Company", value=st.session_state.get("failed_company", "")
            )
        with col2_m:
            manual_link = st.text_input(
                "Link", value=st.session_state.get("failed_link", "")
            )

        pasted_text = st.text_area(
            "Paste the AI-generated JSON response here:", height=250
        )
        manual_btn = st.form_submit_button("Parse JSON & Generate PDF")

    if manual_btn and pasted_text and manual_company:
        saved_local_fit = st.session_state.get("failed_fit", 0)
        cleaned_text = strip_markdown_fences(pasted_text)

        try:
            parsed_data = json.loads(cleaned_text, strict=False)
            adapted_yaml = parsed_data.get("adapted_cv_yaml")
            ai_fit = parsed_data.get("percentual_fit", "Manual")
            reason = parsed_data.get("reason", "")

            if not adapted_yaml:
                st.error(
                    "❌ The JSON is valid but does not contain the 'adapted_cv_yaml' key."
                )
            else:
                st.success(f"✅ JSON parsed successfully! AI Fit: {ai_fit}%")
                if reason:
                    st.info(f"**Reason:** {reason}")
                render_cv_and_save(
                    adapted_yaml, manual_company, manual_link, saved_local_fit, ai_fit
                )

        except json.JSONDecodeError:
            # Last resort: treat the pasted content as raw YAML
            st.warning(
                "⚠️ Input does not appear to be valid JSON. "
                "Attempting to use it directly as YAML..."
            )
            render_cv_and_save(
                pasted_text, manual_company, manual_link, saved_local_fit, "Manual (Raw YAML)"
            )

# ------------------------------------------
# TAB 2: CRM / Job Pipeline
# ------------------------------------------
with tab_crm:
    df_crm = load_crm()
    if not df_crm.empty:
        edited_df = st.data_editor(df_crm, use_container_width=True)
        if st.button("Save CRM"):
            edited_df.to_csv(CRM_FILE, index=False)
            st.success("CRM updated successfully!")
    else:
        st.info("Your pipeline is empty. Analyze a job to get started!")