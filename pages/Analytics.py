
import streamlit as st
from analyzer import DataAnalyzer # Make sure analyzer.py is in the root directory

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")

st.title("📊 Analytics Dashboard")

# --- Check if prediction results exist in session state ---
if 'prediction_results' in st.session_state:
    results_df = st.session_state['prediction_results']
    
    st.markdown("This dashboard provides insights into the generated no-show predictions.")

    # --- Initialize Analyzer and Display Dashboard ---
    analyzer = DataAnalyzer(results_df)
    
    # 1. Summary Metrics
    st.subheader("Summary Metrics")
    stats = analyzer.get_summary_statistics()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Appointments", stats.get('total_appointments', 0))
    col2.metric("High-Risk Appointments", stats.get('high_risk_count', 0))
    col3.metric("Predicted No-Show Rate", f"{stats.get('predicted_noshow_rate', 0):.1%}")

    # 2. Visualizations
    st.subheader("Visualizations")
    risk_dist_fig = analyzer.create_risk_distribution_plot()
    st.plotly_chart(risk_dist_fig, use_container_width=True)

    # You can add more charts from your analyzer here
    # e.g., risk_by_day_fig = analyzer.create_risk_by_day_plot()
    # st.plotly_chart(risk_by_day_fig, use_container_width=True)
    
    # 3. High-Risk Report
    st.subheader("High-Risk Patient Report")
    high_risk_report = analyzer.get_high_risk_report()
    st.dataframe(high_risk_report)
    
    # Download button for the report
    st.download_button(
       label="Download High-Risk Report as CSV",
       data=high_risk_report.to_csv(index=False).encode('utf-8'),
       file_name='high_risk_appointments.csv',
       mime='text/csv',
    )
else:
    st.warning("No prediction data found. Please go to the 'Upload & Predict' page to generate predictions first.")
    st.page_link("pages/1_📈_Upload_&_Predict.py", label="Go to Upload Page", icon="📈")