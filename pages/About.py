
import streamlit as st

st.set_page_config(page_title="About", page_icon="📄", layout="wide")

st.title("📄 About This Project")

st.markdown("""
### Automated Patient Appointment Management & No-Show Prevention System

This project is a demonstration of a machine learning-powered application designed to tackle the critical issue of patient no-shows in healthcare facilities.

**Author:** Judah
**Version:** 1.0

#### Technology Stack:
- **Language:** Python 3.12
- **Web Framework:** Streamlit
- **Data Manipulation:** Pandas
- **Machine Learning:** Scikit-learn (using a `RandomForestClassifier`)
- **Data Visualization:** Plotly Express
- **Development Environment:** VS Code, Jupyter Notebooks

#### Project Goal:
The primary goal is to provide a user-friendly tool for healthcare staff to:
1.  Upload lists of upcoming appointments.
2.  Receive an AI-generated risk score for each appointment.
3.  Visualize trends and patterns in the prediction data.
4.  Identify and download a list of high-risk patients for proactive intervention.

This system aims to reduce revenue loss, optimize clinic schedules, and ultimately improve patient care by ensuring timely medical attention.
""")

st.info("For any questions or collaboration, please contact the project team.", icon="📧")