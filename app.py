import streamlit as st
from PIL import Image
import os
import base64
import io
import textwrap
from typing import Optional, Tuple
from dotenv import load_dotenv
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet

# ======================
# CONFIGURATION
# ======================
st.set_page_config(
    page_title="Smart Diet Analyzer",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

ALLOWED_FILE_TYPES = ['png', 'jpg', 'jpeg']
MODEL_NAME = "llama-3.2-11b-vision-preview"
MODEL_SETTINGS = {
    'temperature': 0.2,
    'max_tokens': 400,
    'top_p': 0.5
}
LOGO_PATH = "src/logo.png"

# ======================
# CACHED RESOURCES
# ======================
@st.cache_data
def get_logo_base64() -> Optional[str]:
    """Load and cache logo as base64 string"""
    try:
        with open(LOGO_PATH, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        st.error(f"Logo file not found at {LOGO_PATH}")
        return None

@st.cache_resource
def initialize_groq_client() -> Groq:
    """Initialize and cache Groq API client"""
    load_dotenv()
    if api_key := os.getenv("GROQ_API_KEY"):
        return Groq(api_key=api_key)
    st.error("GROQ_API_KEY not found in environment")
    st.stop()

# ======================
# CORE FUNCTIONALITY
# ======================
def process_image(uploaded_file: io.BytesIO) -> Optional[Tuple[str, str]]:
    """Process uploaded image to base64 string with format detection"""
    try:
        with Image.open(uploaded_file) as img:
            fmt = img.format or 'PNG'
            buffer = io.BytesIO()
            img.save(buffer, format=fmt)
            return base64.b64encode(buffer.getvalue()).decode('utf-8'), fmt
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

def generate_pdf_content(report_text: str, logo_b64: Optional[str]) -> io.BytesIO:
    """Generate PDF report with logo and analysis content"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Add logo if available
    if logo_b64:
        try:
            logo_data = base64.b64decode(logo_b64)
            with Image.open(io.BytesIO(logo_data)) as logo_img:
                aspect = logo_img.height / logo_img.width
                max_width = 150
                img_width = min(logo_img.width, max_width)
                img_height = img_width * aspect
                
            story.append(
                ReportLabImage(io.BytesIO(logo_data), width=img_width, height=img_height)
            )
            story.append(Spacer(1, 12))
        except Exception as e:
            st.error(f"Logo processing error: {str(e)}")

    # Add report content
    story.extend([
        Paragraph("<b>Nutrition Analysis Report</b>", styles['Title']),
        Spacer(1, 12),
        Paragraph(report_text.replace('\n', '<br/>'), styles['BodyText'])
    ])

    try:
        doc.build(story)
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
    
    buffer.seek(0)
    return buffer

def generate_ai_analysis(client: Groq, image_b64: str, img_format: str) -> Optional[str]:
    """Generate nutritional analysis using Groq's vision API"""
    vision_prompt = textwrap.dedent("""
    As an expert nutritionist with advanced image analysis capabilities, analyze the provided food image:

    1. Identify all visible food items
    2. Estimate calorie content considering:
       - Portion size
       - Cooking method
       - Food density
    3. Mark estimates as "approximate" when assumptions are needed
    4. Calculate total meal calories

    Output format:
    - Food Item 1: [Name] – Estimated Calories: [value] kcal
    - ...
    - **Total Estimated Calories:** [value] kcal

    Include confidence levels for unclear images and specify limitations.
    """)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/{img_format.lower()};base64,{image_b64}"
                    }}
                ]
            }],
            **MODEL_SETTINGS
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# ======================
# UI COMPONENTS
# ======================
def render_main_content(logo_b64: Optional[str]):
    """Main content layout and interactions"""
    st.markdown(f"""
        <div style="text-align: center;">
            {f'<img src="data:image/png;base64,{logo_b64}" width="100">' if logo_b64 else ''}
            <h2 style="color: #4CAF50;">Smart Diet Analyzer</h2>
            <p style="color: #FF6347;">AI-Powered Food & Nutrition Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    if analysis := st.session_state.get('analysis_result'):
        col1, col2 = st.columns(2)
        with col1:
            pdf_buffer = generate_pdf_content(analysis, logo_b64)
            st.download_button(
                "📄 Download Nutrition Report",
                data=pdf_buffer,
                file_name="nutrition_report.pdf",
                mime="application/pdf"
            )
        with col2:
            if st.button("Clear Analysis 🗑️"):
                del st.session_state.analysis_result
                st.rerun()
        
        st.markdown("### 🎯 Nutrition Analysis Report")
        st.info(analysis)

def render_sidebar(client: Groq):
    """Sidebar upload and processing functionality"""
    with st.sidebar:
        st.subheader("Meal Image Analysis")
        uploaded_file = st.file_uploader(
            "Upload Food Image", 
            type=ALLOWED_FILE_TYPES,
            help="Upload clear photo of your meal for analysis"
        )

        if not uploaded_file:
            return

        try:
            st.image(Image.open(uploaded_file), caption="Uploaded Meal Image")
        except Exception as e:
            st.error(f"Invalid image file: {str(e)}")
            return

        if st.button("Analyze Meal 🍽️", use_container_width=True):
            with st.spinner("Analyzing nutritional content..."):
                if img_data := process_image(uploaded_file):
                    analysis = generate_ai_analysis(client, *img_data)
                    if analysis:
                        st.session_state.analysis_result = analysis
                        st.rerun()

# ======================
# APPLICATION ENTRYPOINT
# ======================
def main():
    """Main application controller"""
    client = initialize_groq_client()
    logo_b64 = get_logo_base64()
    
    render_main_content(logo_b64)
    render_sidebar(client)

if __name__ == "__main__":
    main()
