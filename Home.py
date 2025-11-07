
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="No-Show Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HOME PAGE CONTENT ---
st.title("🏥 Welcome to the Patient No-Show Predictor!")

st.markdown("""
This application is designed to help healthcare administrators predict patient appointment no-shows. 
By leveraging a machine learning model, we can identify high-risk appointments, allowing staff to 
take proactive measures to reduce no-show rates, optimize scheduling, and improve patient care.

### How to Use This App:
1.  Navigate to the **Upload & Predict** page using the sidebar.
2.  Upload a CSV file containing upcoming appointment data.
3.  The app will process the data and display a list of predictions.
4.  Go to the **Analytics** page to view an interactive dashboard with insights from the predictions.

This tool is a demonstration of a predictive analytics system built with Python, Streamlit, and Scikit-learn.
""")

st.info("Navigate to a page on the left to get started.", icon="👈")

# Add a separator
st.markdown("---")

# Optional: Add an image
# from PIL import Image
# image = Image.open('path/to/your/image.jpg')
# st.image(image, caption='Improving Healthcare with Data')