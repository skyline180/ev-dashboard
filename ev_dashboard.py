# streamlit_ev_dashboard.py

"""
Streamlit + Plotly dashboard: EV Model Popularity, Trends & Forecasting
Filename: streamlit_ev_dashboard.py

What it does:
- Loads Electric_Vehicle_Population_Data.csv (path configurable)
- Sidebar filters: State, Make, Model, Model Year range, Top N models
- Visualizes:
    • Top Makes bar chart
    • Top Make-Model combinations bar chart
    • State-level EV counts bar chart
    • U.S. choropleth map of EV adoption by state
    • Year-over-Year growth percentage table for Make-Model trends
    • Forecasting future registrations using Prophet (or linear regression fallback)
    • Heatmap of registration counts (Make-Model vs. Model Year, log-scaled)
- Provides CSV downloads for all aggregate tables, growth data, forecasts, state counts, and heatmap data

How to run:
    streamlit run /path/to/streamlit_ev_dashboard.py

This self-contained dashboard skeleton is designed for rapid iteration and extensibility—add geospatial normalization, per-capita metrics, time-series decomposition, or forecasting enhancements as needed.
"""


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# Forecasting
try:
    from prophet import Prophet
except ImportError:
    Prophet = None
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title='EV Model Popularity & Trends', layout='wide')

@st.cache_data
def load_data(csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    for col in ['Make','Model','State','Model Year','County']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if 'Model Year' in df.columns:
        df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce').astype('Int64')
    df['Make_Model'] = (df.get('Make','').fillna('Unknown') + ' ' + df.get('Model','').fillna('')).str.strip()
    return df

@st.cache_data
def aggregate_counts(df):
    top_makes = df['Make'].value_counts().reset_index()
    top_makes.columns = ['Make','count']
    top_models = df['Make_Model'].value_counts().reset_index()
    top_models.columns = ['Make_Model','count']
    state_counts = df['State'].value_counts().reset_index()
    state_counts.columns = ['State','count']
    return top_makes, top_models, state_counts

# Sidebar
st.sidebar.header('Data & Filters')
default_path = 'data/Electric_Vehicle_Population_Data.csv'
csv_path = st.sidebar.text_input('CSV path', value=default_path)
data_file = Path(csv_path)
if not data_file.exists():
    st.sidebar.error(f'File not found: {csv_path}')
    st.stop()

with st.spinner('Loading data…'):
    df = load_data(csv_path)

# Filters
states = sorted(df['State'].dropna().unique().tolist())
selected_states = st.sidebar.multiselect('State', states)
makes = sorted(df['Make'].dropna().unique().tolist())
selected_makes = st.sidebar.multiselect('Make', makes)
if selected_makes:
    models = sorted(df[df['Make'].isin(selected_makes)]['Model'].dropna().unique().tolist())
else:
    models = sorted(df['Model'].dropna().unique().tolist())
selected_models = st.sidebar.multiselect('Model', models)
min_year = int(df['Model Year'].min(skipna=True)) if df['Model Year'].notna().any() else 2000
max_year = int(df['Model Year'].max(skipna=True)) if df['Model Year'].notna().any() else 2025
year_range = st.sidebar.slider('Year Range', min_year, max_year, (min_year, max_year))
top_n = st.sidebar.number_input('Top N models', min_value=5, max_value=50, value=10)

df_filtered = df.copy()
if selected_states:
    df_filtered = df_filtered[df_filtered['State'].isin(selected_states)]
if selected_makes:
    df_filtered = df_filtered[df_filtered['Make'].isin(selected_makes)]
if selected_models:
    df_filtered = df_filtered[df_filtered['Model'].isin(selected_models)]
df_filtered = df_filtered[(df_filtered['Model Year'] >= year_range[0]) & (df_filtered['Model Year'] <= year_range[1])]
st.sidebar.markdown('---')
st.sidebar.write(f'Filtered records: **{len(df_filtered):,}**')

# Main Layout
st.title('EV Model Popularity & Trend Analysis — Enhanced')
st.markdown('Extended dashboard with YoY growth %, forecasting & geospatial maps.')

col1, col2, col3, col4 = st.columns(4)
col1.metric('Total records', f'{len(df_filtered):,}')
col2.metric('Unique Makes', f'{df_filtered["Make"].nunique()}')
col3.metric('Unique Models', f'{df_filtered["Model"].nunique()}')
col4.metric('Year range', f'{year_range[0]} — {year_range[1]}')

# Aggregates
top_makes, top_models, state_counts = aggregate_counts(df_filtered)

# Bar Charts
r1c1, r1c2 = st.columns([1,2])
with r1c1:
    st.subheader('Top Makes')
    fig_makes = px.bar(top_makes.head(top_n), x='count', y='Make', orientation='h', title=f'Top {top_n} Makes')
    st.plotly_chart(fig_makes, use_container_width=True)
    st.download_button('Download makes CSV', data=top_makes.to_csv(index=False), file_name='top_makes.csv')
with r1c2:
    st.subheader('Top Make-Model Combos')
    fig_models = px.bar(top_models.head(top_n), x='count', y='Make_Model', orientation='h', title=f'Top {top_n} Make-Model Combos')
    st.plotly_chart(fig_models, use_container_width=True)
    st.download_button('Download models CSV', data=top_models.to_csv(index=False), file_name='top_models.csv')

# State Counts & Geospatial Map
st.subheader('EV Counts by State')
fig_states = px.bar(state_counts.head(20), x='count', y='State', orientation='h', title='Top 20 States by EV Registrations')
st.plotly_chart(fig_states, use_container_width=True)
st.download_button('Download state counts CSV', data=state_counts.to_csv(index=False), file_name='state_counts.csv')

st.subheader('U.S. Choropleth Map')
state_counts['StateCode'] = state_counts['State']  # assume state codes
fig_map = px.choropleth(state_counts, locations='StateCode', locationmode='USA-states',
                        color='count', scope='usa', title='EV Registrations by State')
st.plotly_chart(fig_map, use_container_width=True)

# YoY Growth
st.subheader('Year-over-Year Growth %')
trend_counts = df_filtered.groupby(['Model Year','Make_Model']).size().reset_index(name='count')
trend_pivot = trend_counts.pivot(index='Model Year', columns='Make_Model', values='count').fillna(0)
growth_pct = trend_pivot.pct_change().multiply(100).round(2)
if not growth_pct.empty:
    st.dataframe(growth_pct)
    st.download_button('Download YoY % CSV', data=growth_pct.to_csv(), file_name='yoy_growth_pct.csv')
else:
    st.info('No YoY growth data available.')

# Forecasting
st.subheader('Forecasting EV Registrations for a Model')
selected_for_forecast = st.selectbox('Select a model', top_models['Make_Model'].head(top_n).tolist())
if selected_for_forecast:
    df_model = trend_counts[trend_counts['Make_Model'] == selected_for_forecast][['Model Year','count']].rename(columns={'Model Year': 'ds', 'count': 'y'})
    df_model['ds'] = pd.to_datetime(df_model['ds'], format='%Y')
    future_years = st.number_input('Forecast horizon (years)', min_value=1, max_value=10, value=5)
    if Prophet:
        m = Prophet(yearly_seasonality=False)
        m.fit(df_model)
        future = m.make_future_dataframe(periods=future_years, freq='Y')
        forecast = m.predict(future)
        fig_forecast = px.line(forecast, x='ds', y=['yhat'], title=f'Forecast for {selected_for_forecast}', labels={'ds':'Year','yhat':'Forecast'})
        st.plotly_chart(fig_forecast, use_container_width=True)
        st.download_button('Download forecast CSV', data=forecast[['ds','yhat']].to_csv(index=False), file_name='forecast.csv')
    else:
        st.warning('Prophet not available—fallback to linear regression.')
        X = np.arange(len(df_model)).reshape(-1,1)
        y = df_model['y'].values
        lr = LinearRegression().fit(X, y)
        future_idx = np.arange(len(df_model) + future_years).reshape(-1,1)
        y_pred = lr.predict(future_idx)
        years = pd.date_range(start=df_model['ds'].min(), periods=len(future_idx), freq='Y').year
        df_fore = pd.DataFrame({'Year': years, 'Forecast': y_pred})
        fig_forecast = px.line(df_fore, x='Year', y='Forecast', title=f'Linear Forecast for {selected_for_forecast}')
        st.plotly_chart(fig_forecast, use_container_width=True)
        st.download_button('Download linear forecast CSV', data=df_fore.to_csv(index=False), file_name='linear_forecast.csv')

# Heatmap
st.subheader('Heatmap: Models vs Model Year (log-scaled)')
heat_top_n = st.slider('Number of models in heatmap', 5, top_n, min(10, top_n))
heat_models = top_models['Make_Model'].head(heat_top_n).tolist()
heat_df = df_filtered[df_filtered['Make_Model'].isin(heat_models)]
heat_counts = heat_df.groupby(['Make_Model','Model Year']).size().reset_index(name='count')
heat_pivot = heat_counts.pivot(index='Make_Model', columns='Model Year', values='count').fillna(0)
if not heat_pivot.empty:
    heat_matrix = np.log1p(heat_pivot)
    fig_heat = px.imshow(heat_matrix, labels=dict(x='Model Year', y='Make-Model', color='log1p(count)'),
                         x=heat_pivot.columns.astype(str).tolist(),
                         y=heat_pivot.index.tolist(),
                         aspect='auto', title='Heatmap (log1p counts)')
    st.plotly_chart(fig_heat, use_container_width=True)
    st.download_button('Download heatmap CSV', data=heat_pivot.to_csv(), file_name='heatmap.csv')
else:
    st.info('No heatmap data available.')

st.markdown('---')
st.caption('Enhanced skeleton with YoY growth%, forecasting & maps. Extend further as needed.')
