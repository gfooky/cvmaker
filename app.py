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
import traceback

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

def generate_cv_corpus_via_ai(master_cv_text: str) -> str:
    """
    Calls the Gemini API to read the master CV and generate a dense text
    with the keywords, synonyms, and translations to English.
    """
    prompt = f"""
    You are an expert technical recruiter and ATS system developer.
    Analyze the following CV and extract all key terms, hard skills, job titles, and technologies.
    Generate a dense text block containing these keywords, including their common synonyms and exact English translations.
    
    CRITICAL RULE: Return ONLY the raw text with the keywords. Do not use markdown, do not write code fences (```), do not include greetings or explanations. Just the words separated by commas or newlines.

    MY CV:
    {master_cv_text}
    """
    
    print("\n" + "="*50)
    print("[DEBUG - AI CORPUS] Starting API call to generate corpus...")
    print(f"[DEBUG - AI CORPUS] Model selected: 'gemini-flash-latest'")
    
    try:
        # Trigger the AI model
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
        )
        print("[DEBUG - AI CORPUS] API call completed successfully.")
        
        # Safely get a preview of the response text (first 100 characters)
        raw_response_preview = repr(response.text[:100]) if response.text else "EMPTY_RESPONSE"
        print(f"[DEBUG - AI CORPUS] Raw response text preview: {raw_response_preview}...")
        
        # Cleans the generated text (removing ``` delimiters)
        print("[DEBUG - AI CORPUS] Cleaning markdown fences from response...")
        corpus_text = strip_markdown_fences(response.text)
        
        # Check current working directory to avoid saving it in a weird temp folder
        cwd = os.getcwd()
        file_path = os.path.join(cwd, "cv_corpus.txt")
        print(f"[DEBUG - AI CORPUS] Current Working Directory is: {cwd}")
        print(f"[DEBUG - AI CORPUS] Attempting to create/overwrite file at: {file_path}")
        
        # Saves to file for future use
        with open("cv_corpus.txt", "w", encoding="utf-8") as f:
            f.write(corpus_text)
            
        print(f"[DEBUG - AI CORPUS] Successfully wrote {len(corpus_text)} characters to the file.")
        
        # Immediate verification: Does the OS recognize the file?
        if os.path.exists(file_path):
            print("[DEBUG - AI CORPUS] OS Verification: TRUE (File exists on disk!).")
        else:
            print("[DEBUG - AI CORPUS] OS Verification: FALSE (File not found on disk after writing!).")
        
        print("="*50 + "\n")
        return corpus_text
        
    except Exception as e:
        # Detailed error catching to find exactly what went wrong
        print("\n[ERROR - AI CORPUS] An exception occurred during corpus generation!")
        print(f"[ERROR - AI CORPUS] Exception Type: {type(e).__name__}")
        print(f"[ERROR - AI CORPUS] Exception Details: {str(e)}")
        
        print("[ERROR - AI CORPUS] Full Stack Trace:")
        traceback.print_exc()  # This will print the exact file line that crashed
        
        # In case of failure, return the original master CV as a fallback
        print("[DEBUG - AI CORPUS] Returning the original master_cv_text as a fallback.")
        print("="*50 + "\n")
        return master_cv_text

def get_or_create_cv_corpus(master_cv_text: str) -> str:
    """
    Checks if the corpus file exists. If not, generates it with AI.
    """
    corpus_file = "cv_corpus.txt"
    if os.path.exists(corpus_file):
        with open(corpus_file, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return generate_cv_corpus_via_ai(master_cv_text)
    
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


def build_prompt(job_description: str, master_cv_text: str, score_threshold: int, target_language: str) -> str:
    return f"""
You are a technical recruiter and ATS specialist.
Analyze the fit (0–100) between the job description and my CV.
If the fit >= {score_threshold}, return an adapted CV (highly ATS-optimized, mirroring the job's terminology).
If not, return null. NEVER invent or fabricate data.

CRITICAL REQUIREMENTS:
1. Translate and adapt ALL the generated CV content to: {target_language}.
2. Keep the original YAML structure intact.
3. Always copy the "design:" block exactly as it appears at the end of MY CV. Do not omit it.

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
    job_text: str, master_cv_text: str, score_threshold: int, target_language: str
) -> dict:
    """
    Calls the Gemini API to assess CV fit and generate an ATS-optimized CV.
    Returns a parsed JSON dict with keys: percentual_fit, reason, adapted_cv_yaml.
    """
    prompt = build_prompt(job_text, master_cv_text, score_threshold, target_language)

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
st.set_page_config(page_title="CV-Maker Pipeline", layout="wide")

master_cv_text = read_master_cv()
if not master_cv_text:
    st.error(
        f"⚠️ Master CV not found. Please create a file named '{MASTER_CV_FILE}' "
        "in the same directory as this script."
    )
    st.stop()

# Load CRM Data
df_crm = load_crm()

# ------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------
with st.sidebar:
    st.title("📂 Job Pipeline")
    st.write("Manage your applications here.")
    
    # Build options list dynamically
    options = ["➕ New Application"]
    option_mapping = {} # To easily find the DataFrame index later
    
    if not df_crm.empty:
        # Iterate backwards to show the newest entries first
        for idx, row in df_crm.iloc[::-1].iterrows():
            opt_str = f"📄 {row['Company']} ({row['Date']}) [ID:{idx}]"
            options.append(opt_str)
            option_mapping[opt_str] = idx
            
    selected_option = st.radio("Navigation", options)
    
    # Keep the raw data editor available inside an expander
    st.divider()
    with st.expander("⚙️ Edit CRM Database"):
        if not df_crm.empty:
            edited_df = st.data_editor(df_crm, width="stretch")
            if st.button("Save CRM Changes"):
                edited_df.to_csv(CRM_FILE, index=False)
                st.success("CRM updated successfully!")
                st.rerun() # Refresh app to update sidebar list
        else:
            st.info("Pipeline is empty.")

# ------------------------------------------
# MAIN AREA
# ------------------------------------------
if selected_option == "➕ New Application":
    st.title("🚀 CV-Maker — Auto-Apply & CV Optimizer")
    
    with st.form("job_form"):
        col1, col2 = st.columns(2)
        with col1:
            company_name = st.text_input("Company Name")
        with col2:
            job_link = st.text_input("Job Link (optional)")
        
        target_language = st.selectbox("Generate CV in which language?", ["Português", "English", "Español"])
        
        job_description = st.text_area("Paste the job description here:", height=150)
        submit_btn = st.form_submit_button("Analyze Fit & Generate CV")

    # When the form is submitted, we save the data in Streamlit's memory
    if submit_btn and job_description and company_name:
        st.session_state["job_data"] = {
            "company": company_name,
            "link": job_link,
            "desc": job_description,
            "lang": target_language
        }
        st.session_state["force_ai"] = False  # Force analysis reset

    # If there are job data in the memory, process them
    if "job_data" in st.session_state:
        job_data = st.session_state["job_data"]
        
        # 2. Check/Verify/Generate the Corpus
        corpus_file = "cv_corpus.txt"
        if not os.path.exists(corpus_file):
            with st.spinner("🧠 Initializing... AI is generating your bilingual synonym dictionary. This only happens once!"):
                cv_corpus_text = get_or_create_cv_corpus(master_cv_text)
            
            if cv_corpus_text and os.path.exists(corpus_file):
                st.toast("✅ Dictionary created and saved as 'cv_corpus.txt'!")
            else:
                st.warning("⚠️ Could not generate dictionary. Falling back to original Master CV.")
                cv_corpus_text = master_cv_text
        else:
            cv_corpus_text = get_or_create_cv_corpus(master_cv_text)
        
        # 3. Local Filter
        local_fit = local_pre_filter(cv_corpus_text, job_data["desc"])

        # Decision Logic for Running the AI
        run_ai = False

        if local_fit >= LOCAL_SCORE_THRESHOLD:
            st.success(f"✅ Passed the local pre-filter ({local_fit}%)! Waking up the AI...")
            run_ai = True
        elif st.session_state.get("force_ai", False):
            st.info(f"⚡ Forced analysis manually by the user. (Original local fit: {local_fit}%)")
            run_ai = True
        else:
            st.warning(
                f"⚠️ The job did not meet the minimum score in the pre-filter.\n\n"
                f"**Local Fit:** {local_fit}% (Minimum required: {LOCAL_SCORE_THRESHOLD}%)"
            )
            # Button to ignore the block
            if st.button("🚀 Force Submission to the AI (Spend Tokens)"):
                st.session_state["force_ai"] = True
                st.rerun()

        # 4. AI execution (if approved or forced)
        if run_ai:
            with st.spinner(f"The AI is translating your CV to {job_data['lang']} and optimizing it..."):
                try:
                    ai_result = evaluate_and_generate_ai(
                        job_data["desc"], master_cv_text, AI_SCORE_THRESHOLD, job_data["lang"]
                    )
                    ai_fit = ai_result.get("percentual_fit", 0)
                    reason = ai_result.get("reason", "No reason provided.")
                    adapted_yaml = ai_result.get("adapted_cv_yaml")

                    if ai_fit >= AI_SCORE_THRESHOLD:
                        st.success(f"**AI Fit Score: {ai_fit}%** — {reason}")
                        render_cv_and_save(
                            adapted_yaml, job_data["company"], job_data["link"], local_fit, ai_fit
                        )
                        del st.session_state["job_data"]
                    else:
                        st.warning(
                            f"⚠️ Job rejected definitively by the AI. Fit: {ai_fit}% "
                            f"(Minimum: {AI_SCORE_THRESHOLD}%) — {reason}"
                        )
                        del st.session_state["job_data"]

                except Exception as e:
                    st.error(f"🚨 API call failed: {e}")
                    st.info(
                        "💡 **Plan B activated:** Copy the prompt below, paste it into "
                        "any AI chat interface, then paste the full JSON response into "
                        "the manual form at the bottom of this page."
                    )

                    st.session_state["failed_company"] = job_data["company"]
                    st.session_state["failed_link"] = job_data["link"]
                    st.session_state["failed_fit"] = local_fit

                    emergency_prompt = build_prompt(
                        job_data["desc"], master_cv_text, AI_SCORE_THRESHOLD, job_data["lang"]
                    )
                    st.code(emergency_prompt, language="markdown")
                    
                    if "job_data" in st.session_state:
                        del st.session_state["job_data"]

    # ------------------------------------------
    # FALLBACK: Manual Plan B Form
    # ------------------------------------------
    st.divider()
    st.subheader("🛠️ Plan B: Manual Generator")
    st.write("Use this form if the API above failed. Paste the full JSON response you got from an external AI here.")

    with st.form("manual_form"):
        col1_m, col2_m = st.columns(2)
        with col1_m:
            manual_company = st.text_input("Company", value=st.session_state.get("failed_company", ""))
        with col2_m:
            manual_link = st.text_input("Link", value=st.session_state.get("failed_link", ""))

        pasted_text = st.text_area("Paste the AI-generated JSON response here:", height=250)
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
                st.error("❌ The JSON is valid but does not contain the 'adapted_cv_yaml' key.")
            else:
                st.success(f"✅ JSON parsed successfully! AI Fit: {ai_fit}%")
                if reason:
                    st.info(f"**Reason:** {reason}")
                render_cv_and_save(adapted_yaml, manual_company, manual_link, saved_local_fit, ai_fit)

        except json.JSONDecodeError:
            st.warning("⚠️ Input does not appear to be valid JSON. Attempting to use it directly as YAML...")
            render_cv_and_save(pasted_text, manual_company, manual_link, saved_local_fit, "Manual (Raw YAML)")

else:
    # ------------------------------------------
    # VIEW SAVED JOB DETAILS
    # ------------------------------------------
    # Extract the correct index from the dictionary mapping
    job_idx = option_mapping[selected_option]
    job_data = df_crm.loc[job_idx]
    
    st.title("🏢 Application Details")
    st.header(job_data['Company'])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Local Fit (%)", job_data.get('Local Fit (%)', 'N/A'))
    col2.metric("AI Fit (%)", job_data.get('AI Fit (%)', 'N/A'))
    col3.metric("Status", job_data.get('Status', 'N/A'))
    
    st.divider()
    st.write(f"**📅 Date Applied:** {job_data.get('Date', 'N/A')}")
    
    link = job_data.get('Link/Note', '')
    if pd.notna(link) and str(link).strip():
        st.write(f"**🔗 Link/Note:** {link}")
    else:
        st.write("**🔗 Link/Note:** Not provided")
        
    st.divider()
    st.info("💡 You can find the generated PDF for this application inside the `rendercv_output` folder in your project directory.")