import streamlit as st
from PIL import Image
import os
import base64
import io
from dotenv import load_dotenv
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ======================
# CONFIGURATION SETTINGS
# ======================
st.set_page_config(
    page_title="Smart Diet Analyzer",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

ALLOWED_FILE_TYPES = ['png', 'jpg', 'jpeg']

# ======================
# UTILITY FUNCTIONS
# ======================

def initialize_api_client():
    """Initialize Groq API client"""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("API key not found. Please verify .env configuration.")
        st.stop()
    return Groq(api_key=api_key)


def encode_image(image_path):
    """Encode an image to base64"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        return ""


def process_image(uploaded_file):
    """Convert image to base64 string"""
    try:
        image = Image.open(uploaded_file)
        buffer = io.BytesIO()
        image.save(buffer, format=image.format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8'), image.format
    except Exception as e:
        st.error(f"Image processing error: {e}")
        return None, None


def generate_pdf(report_text):
    """Generate a PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("<b>Nutrition Analysis Report</b>", styles['Title']), Spacer(1, 12),
             Paragraph(report_text.replace('\n', '<br/>'), styles['BodyText'])]
    doc.build(story)
    buffer.seek(0)
    return buffer


def generate_analysis(uploaded_file, client):
    """Generate AI-powered food analysis"""
    base64_image, img_format = process_image(uploaded_file)
    if not base64_image:
        return None
    
    image_url = f"data:image/{img_format.lower()};base64,{base64_image}"
    
    try:
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """
                        You are an expert nutritionist with advanced image analysis capabilities.  
                        Your task is to analyze the provided image, identify all visible food items, and estimate their calorie content as accurately as possible.  

                        **Instructions:**  
                        - List each identified food item separately.  
                        - Use known nutritional data to provide accurate calorie estimates.  
                        - Consider portion size, cooking method, and density of food.  
                        - Clearly specify if an item's calorie count is an estimate due to ambiguity.  
                        - Provide the total estimated calorie count for the entire meal.  

                        **Output Format:**  
                        - Food Item 1: [Name] – Estimated Calories: [value] kcal  
                        - Food Item 2: [Name] – Estimated Calories: [value] kcal  
                        - ...  
                        - **Total Estimated Calories:** [value] kcal  

                        If the image is unclear or lacks enough details, state the limitations and provide a confidence percentage for the estimation.
                        """},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=400,
            top_p=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API communication error: {e}")
        return None

# ======================
# UI COMPONENTS
# ======================

def display_main_interface():
    """Render primary application interface"""
    logo_b64 = encode_image("src/logo.png")
    
    # HTML with inline styles to change text colors
    st.markdown(f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{logo_b64}" width="100">
            <h2 style="color: #4CAF50;">Smart Diet Analyzer</h2>
            <p style="color: #FF6347;">AI-Powered Food & Nutrition Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.get('analysis_result'):
        # Create two columns: one for download and one for clear button
        col1, col2 = st.columns([1, 1])
        
        # Left column for the Download button
        with col1:
            pdf_report = generate_pdf(st.session_state.analysis_result)
            st.download_button("📄 Download Nutrition Report", data=pdf_report, file_name="nutrition_report.pdf", mime="application/pdf")
        
        # Right column for the Clear button
        with col2:
            if st.button("Clear Analysis 🗑️"):
                st.session_state.pop('analysis_result')
                st.rerun()
    
    if st.session_state.get('analysis_result'):
        st.markdown("### 🎯 Nutrition Analysis Report")
        st.info(st.session_state.analysis_result)


def render_sidebar(client):
    """Create sidebar UI elements"""
    with st.sidebar:
        st.subheader("Image Upload")
        uploaded_file = st.file_uploader("Upload Food Image", type=ALLOWED_FILE_TYPES)
        
        if uploaded_file:
            st.image(Image.open(uploaded_file), caption="Uploaded Food Image")
            if st.button("Analyze Meal 🍽️"):
                with st.spinner("Analyzing image..."):
                    report = generate_analysis(uploaded_file, client)
                    st.session_state.analysis_result = report
                    st.rerun()

# ======================
# APPLICATION ENTRYPOINT
# ======================

def main():
    """Primary application controller"""
    client = initialize_api_client()
    display_main_interface()
    render_sidebar(client)

if __name__ == "__main__":
    main()
