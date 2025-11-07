
import streamlit as st
import pandas as pd
from predictor import NoShowPredictor # Make sure predictor.py is in the root directory

st.set_page_config(page_title="Upload & Predict", page_icon="📈", layout="wide")

st.title("📈 Upload Data & Generate Predictions")

st.markdown("Upload a CSV file with appointment data to get no-show risk predictions.")

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="The CSV should contain columns like 'age', 'gender', 'days_between', etc."
)

if uploaded_file is not None:
    try:
        # Load raw data
        raw_data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        # Display a preview of the uploaded data
        if st.checkbox("Show a preview of the uploaded data"):
            st.write(raw_data.head())

        # --- Prediction Logic ---
        if st.button("Generate Predictions", type="primary"):
            with st.spinner('Running predictions... This may take a moment.'):
                predictor = NoShowPredictor()
                predictions_df = predictor.predict_batch(raw_data)
            
            st.success('Predictions complete!')

            # Combine original data with predictions
            full_results_df = pd.concat([raw_data.reset_index(drop=True), predictions_df], axis=1)

            # --- Store results in session state to share with other pages ---
            st.session_state['prediction_results'] = full_results_df
            
            # --- Display Predictions on this page ---
            st.subheader("Prediction Results")
            st.dataframe(full_results_df)

            st.info("Navigate to the 'Analytics' page to see a full dashboard of these results.", icon="📊")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please ensure your CSV file has the correct format and columns.")

else:
    # Clear session state if no file is uploaded
    if 'prediction_results' in st.session_state:
        del st.session_state['prediction_results']
    st.info("Please upload a CSV file to begin.")