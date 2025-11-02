import streamlit as st
import pandas as pd
from predictor import NoShowPredictor
from analyzer import DataAnalyzer

# --- Page Configuration ---
st.set_page_config(
    page_title="No-Show Prediction Dashboard",
    page_icon="🏥",
    layout="wide"
)

# --- Main Application ---
st.title("🏥 Automated Patient No-Show Prediction")
st.markdown("Upload your appointment data to predict no-shows and get actionable insights.")

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with appointment data",
        type="csv"
    )

# --- Core Logic ---
if uploaded_file is not None:
    try:
        # Load the raw data from the user's upload
        raw_data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        if st.checkbox("Show raw uploaded data"):
            st.write(raw_data.head())

        # --- Prediction Step ---
        with st.spinner('Running predictions... This may take a moment.'):
            # Create an instance of our predictor
            predictor = NoShowPredictor()
            
            # The predictor will handle its own preprocessing internally now
            predictions_df = predictor.predict_batch(raw_data)
        
        st.success('Predictions complete!')
        
        # --- Combine Original Data with Predictions for Analysis ---
        # This joins the original details with the new prediction columns
        full_results_df = pd.concat([raw_data.reset_index(drop=True), predictions_df], axis=1)

        # --- Dashboard Display ---
        st.header("Prediction Dashboard")
        
        # Pass the complete results to the analyzer
        analyzer = DataAnalyzer(full_results_df)

        # 1. Summary Metrics
        stats = analyzer.get_summary_statistics()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Appointments", stats.get('total_appointments', 0))
        col2.metric("High-Risk Appointments", stats.get('high_risk_count', 0))
        col3.metric("Predicted No-Show Rate", f"{stats.get('predicted_noshow_rate', 0):.1%}")

        # 2. Visualizations
        st.plotly_chart(analyzer.create_risk_distribution_plot(), use_container_width=True)
        
        # 3. High-Risk Report
        st.subheader("High-Risk Patient Report")
        high_risk_report = analyzer.get_high_risk_report()
        st.dataframe(high_risk_report)
        
        # Download button
        st.download_button(
           label="Download High-Risk Report as CSV",
           data=high_risk_report.to_csv(index=False).encode('utf-8'),
           file_name='high_risk_appointments.csv',
           mime='text/csv',
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to begin analysis.")