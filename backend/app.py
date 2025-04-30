# from flask import Flask, render_template, request
# import analysis
# import sys
# import os

# # Explicitly add the current directory to the Python path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     result = {}
#     if request.method == 'POST':
#         analysis_type = request.form['analysis']
#         if analysis_type == 'univariate':
#             mean_std, mode = analysis.univariate_analysis()
#             result = {'type': 'univariate', 'mean_std': mean_std, 'mode': mode}
#         elif analysis_type == 'bivariate':
#             plots, chi = analysis.bivariate_analysis()
#             result = {'type': 'bivariate', 'plots': plots, 'chi': chi}
#         elif analysis_type == 'multivariate':
#             plots, svm_r, nb_r, mlr = analysis.multivariate_analysis()
#             result = {'type': 'multivariate', 'plots': plots, 'svm': svm_r, 'nb': nb_r, 'mlr': mlr}
#     return render_template('index.html', result=result)

# if __name__ == '__main__':
#     app.run(debug=True)





import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Road Accident Data Analysis", layout="wide")
st.title("🚧 Road Accident Data Analysis (NCRB 2020)")

# Load your local CSV
df = pd.read_csv("traffic.csv")

# Display data preview
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# Analysis type
analysis_type = st.radio("Select Type of Analysis", ["Univariate", "Bivariate", "Multivariate"], horizontal=True)

if analysis_type == "Univariate":
    col = st.selectbox("Select column to analyze", df.columns[2:])
    st.subheader(f"Univariate Analysis of {col}")
    fig = px.bar(df, x="State/UT/City", y=col, title=f"{col} across States", labels={col: col, "State/UT/City": "Location"})
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Bivariate":
    x_col = st.selectbox("X-axis", df.columns[2:], key='biv_x')
    y_col = st.selectbox("Y-axis", df.columns[2:], key='biv_y')
    st.subheader(f"Bivariate Analysis: {x_col} vs {y_col}")
    fig = px.scatter(df, x=x_col, y=y_col, color="State/UT/City", hover_name="State/UT/City")
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Multivariate":
    x_col = st.selectbox("X-axis", df.columns[2:], key='multi_x')
    y_col = st.selectbox("Y-axis", df.columns[2:], key='multi_y')
    z_col = st.selectbox("Z-axis", df.columns[2:], key='multi_z')
    st.subheader(f"Multivariate Analysis: {x_col}, {y_col}, {z_col}")
    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color="State/UT/City")
    st.plotly_chart(fig, use_container_width=True)
