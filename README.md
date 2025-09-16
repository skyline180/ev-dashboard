# ⚡ EV Model Popularity & Trend Analysis Dashboard

An **interactive EV adoption analytics dashboard** built with **Streamlit** and **Plotly**—featuring **Year-over-Year growth**, **forecasting**, and **geospatial mapping**.

<p align="center">
  <img src="images/dashboard_overview.png" width="700" alt="Dashboard Overview">
</p>

---

## 🚀 Features

- 🔍 **Smart Filters**  
  Drill down by _State_, _Make_, _Model_, _Model Year_, and choose the **Top N models** for focused analysis.

- 📊 **Visual Exploration**
  - Top EV Makes and Make-Model combinations
  - Registration counts by state via **bar chart** and **U.S. choropleth map**
  - **Year-over-Year (YoY) growth** table
  - 🔮 **Registration Forecasting** using [**Facebook Prophet**](https://facebook.github.io/prophet/), with linear regression fallback
  - Heatmap of **Model × Year** (log-scaled for clarity)

- ⚡ **High Interactivity**  
  Optimized for fast rendering with large datasets and real-time visual updates.

- 📥 **Data Export**  
  Download CSV data for any chart—forecasts, growth stats, and more.

---

## 📂 Dataset

- **Source**: [Electric Vehicle Population Data – Washington State Department of Licensing](https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/7c2c-a9ih)
- **Scope**: Battery Electric Vehicles (BEVs) and Plug-in Hybrid Electric Vehicles (PHEVs) registered in Washington State
- **Key Fields**:
  - `Make`, `Model`, `Model Year`
  - `State`, `County`, `Electric Vehicle Type`, `VIN`, etc.

---

## 🖥️ Live Demo

🔗 **Check out the live app:**  
[https://share.streamlit.io/skyline180/EV-Dashboard-Streamlit](https://share.streamlit.io/skyline180/EV-Dashboard-Streamlit)

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/skyline180/EV-Dashboard-Streamlit.git
cd EV-Dashboard-Streamlit

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run streamlit_ev_dashboard.py

