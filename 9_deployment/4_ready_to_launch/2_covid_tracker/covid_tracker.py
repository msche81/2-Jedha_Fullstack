import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Covid Tracker Exercice MSC",
    layout="wide"
)

### Partie supérieure (sur toute la largeur)
st.title("This is a CovidTracker Dashboard for the Jedha BootCamp journey by MSC!")

st.markdown("""
    Goal of this dashboard is to learn and practice Streamlit and to build a dashboard that follows the evolution of cases across Europe.

    Source: [European Centre for Disease Prevention and Control](https://www.ecdc.europa.eu/en/publications-data/data-daily-new-cases-covid-19-eueea-country)
""")

# Fonction pour charger les données
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    data['dateRep'] = pd.to_datetime(data['dateRep'], format='%d/%m/%Y', errors='coerce')  # Conversion en format datetime
    data['year'] = data['dateRep'].dt.year  # Extraire l'année
    return data

# Chargement des données
DATA_URL = 'https://opendata.ecdc.europa.eu/covid19/nationalcasedeath_eueea_daily_ei/csv'
data = load_data()

# Correction des données : forcer les morts à 42 pour le 1er septembre 2021 (Grèce)
data.loc[(data['dateRep'] == '2021-09-01') & (data['countriesAndTerritories'] == 'Greece'), 'deaths'] = 500

# Correction des données : forcer les cases à 15000 pour le 20 mai 2021 (France)
data.loc[(data['dateRep'] == '2021-05-20') & (data['countriesAndTerritories'] == 'France'), 'cases'] = 15000

# Correction des données : forcer les cases à 100000 pour le 08 février 2022 (Netherlands)
data.loc[(data['dateRep'] == '2022-02-08') & (data['countriesAndTerritories'] == 'Netherlands'), 'cases'] = 100000

# Extraction des années disponibles
data['year'] = data['dateRep'].dt.year

### Country Analysis
st.subheader("Country Analysis")
col1, col2 = st.columns(2)

with col1:
    countries = ['All'] + list(data['countriesAndTerritories'].unique())
    selected_country = st.selectbox("Select a country:", countries)

with col2:
    years = ['All'] + sorted(data['year'].dropna().unique())
    selected_year = st.selectbox("Select a year:", years)

# Filtrage des données en fonction des sélections
filtered_data = data.copy()
if selected_country != 'All':
    filtered_data = filtered_data[filtered_data['countriesAndTerritories'] == selected_country]
if selected_year != 'All':
    filtered_data = filtered_data[filtered_data['year'] == int(selected_year)]

# Calculs pour les widgets
total_cases = filtered_data['cases'].sum()
total_deaths = filtered_data['deaths'].sum()

# Affichage des widgets côte à côte
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div style="
            background-color: #e0f7fa;
            border: 2px solid #b2ebf2;
            border-radius: 5px;
            padding: 5px;
            text-align: center;
        ">
            <h3 style="font-size: 25px; margin-bottom: 5px;">Total Cases</h3>
            <p style="font-size: 40px; margin: 0;">{int(total_cases):,}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="
            background-color: #ffcccb;
            border: 2px solid #f2b2b2;
            border-radius: 5px;
            padding: 5px;
            text-align: center;
        ">
            <h3 style="font-size: 25px; margin-bottom: 5px;">Total Deaths</h3>
            <p style="font-size: 40px; margin: 0;">{int(total_deaths):,}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Regrouper les données par date pour le graphique
grouped_data = filtered_data.groupby('dateRep', as_index=False).sum()

### Graphique avec deux axes
fig = go.Figure()

# Ligne pour les cas
fig.add_trace(go.Scatter(
    x=grouped_data['dateRep'], 
    y=grouped_data['cases'], 
    name="Cases",
    mode='lines'
))

# Ligne pour les décès
fig.add_trace(go.Scatter(
    x=grouped_data['dateRep'], 
    y=grouped_data['deaths'], 
    name="Deaths",
    mode='lines',
    yaxis="y2"
))

# Configuration des axes
fig.update_layout(
    title="Covid-19 Cases and Deaths Over Time",
    xaxis=dict(title="Date"),  # Échelle dynamique basée sur les données
    yaxis=dict(title="Cases"),
    yaxis2=dict(
        title="Deaths",
        overlaying="y",
        side="right"
    ),
    template="plotly_white",
    legend=dict(
        title="Metric",
        orientation="h",
        yanchor="top",
        y=-0.2,
        xanchor="center",
        x=0.5
    )
)

# Affichage du graphique précédent
st.plotly_chart(fig, use_container_width=True)

### Map Analysis et tableau
st.subheader("European Data Overview")
map_col, table_col = st.columns([1, 1.5])

# Map dans col1
with map_col:
    st.subheader("Map Analysis")

    # Filtres pour la carte
    map_filters_col1, map_filters_col2 = st.columns(2)
    with map_filters_col1:
        map_year_filter = st.selectbox("Select a year for the map:", ['All'] + sorted(data['year'].unique()), index=0)
    with map_filters_col2:
        map_data_type = st.radio("Select data type for the map:", ["Cases", "Deaths"], horizontal=True)

    # Filtrer les données pour la carte
    map_filtered_data = data.copy()
    if map_year_filter != 'All':
        map_filtered_data = map_filtered_data[map_filtered_data['year'] == int(map_year_filter)]

    # Regrouper les données par pays
    map_grouped_data = map_filtered_data.groupby('countriesAndTerritories').agg({
        'cases': 'sum',
        'deaths': 'sum',
        'popData2020': 'mean'
    }).reset_index()

    # Ajouter les colonnes ISO pour les pays
    country_iso_map = data[['countriesAndTerritories', 'countryterritoryCode']].drop_duplicates()
    map_grouped_data = map_grouped_data.merge(country_iso_map, on='countriesAndTerritories', how='left')

    # Sélection des données pour la carte
    map_data_column = "cases" if map_data_type == "Cases" else "deaths"
    color_scale = "Blues" if map_data_type == "Cases" else "Reds"

    ### Création de la carte
    fig_map = px.choropleth(
        map_grouped_data,
        locations="countryterritoryCode",  # Utiliser les codes ISO pour les pays
        color=map_data_column,  # Colonne à utiliser pour colorier la carte
        hover_name="countriesAndTerritories",  # Nom des pays au survol
        hover_data={map_data_column: True, "popData2020": True},
        color_continuous_scale=color_scale,
        title=f"Covid-19 {map_data_type} in Europe ({map_year_filter})",
        scope="europe"  # Limiter la carte à l'Europe
    )

    fig_map.update_layout(height=800)  # Augmenter la taille de la carte

    # Affichage de la carte
    st.plotly_chart(fig_map, use_container_width=True)

# Tableau dans col2
with table_col:
    st.subheader("Data Table")

    # Filtre par année pour le tableau
    table_year_filter = st.selectbox("Select a year for the table:", ['All'] + sorted(data['year'].unique()), index=0)
    table_filtered_data = data.copy()
    if table_year_filter != 'All':
        table_filtered_data = table_filtered_data[table_filtered_data['year'] == int(table_year_filter)]

    # Regrouper les données par pays
    table_grouped_data = table_filtered_data.groupby('countriesAndTerritories').agg({
        'cases': 'sum',
        'deaths': 'sum',
        'popData2020': 'mean'
    }).reset_index()

    table_grouped_data['% Cases vs Pop'] = (table_grouped_data['cases'] / table_grouped_data['popData2020']) * 100
    table_grouped_data['% Deaths vs Cases'] = (table_grouped_data['deaths'] / table_grouped_data['cases']) * 100
    table_grouped_data['% Deaths vs Pop'] = (table_grouped_data['deaths'] / table_grouped_data['popData2020']) * 100

    # Trier les données par % Cases vs Pop
    table_grouped_data = table_grouped_data.sort_values(by='% Cases vs Pop', ascending=False)

    # Affichage
    st.dataframe(table_grouped_data)

### Bar Charts by Country
st.subheader("Bar Charts by Country")

# Filtrer les données par année avant le regroupement
bar_chart_year_filter = st.selectbox("Select a year for the charts:", ['All'] + sorted(data['year'].unique()), index=0)
bar_chart_filtered_data = data.copy()
if bar_chart_year_filter != 'All':
    bar_chart_filtered_data = bar_chart_filtered_data[bar_chart_filtered_data['year'] == int(bar_chart_year_filter)]

# Regrouper les données filtrées
bar_chart_grouped_data = bar_chart_filtered_data.groupby('countriesAndTerritories').agg({
    'cases': 'sum',
    'deaths': 'sum',
    'popData2020': 'mean'
}).reset_index()

# Calculer les pourcentages
bar_chart_grouped_data['% Cases vs Pop'] = (bar_chart_grouped_data['cases'] / bar_chart_grouped_data['popData2020']) * 100
bar_chart_grouped_data['% Deaths vs Cases'] = (bar_chart_grouped_data['deaths'] / bar_chart_grouped_data['cases']) * 100
bar_chart_grouped_data['% Deaths vs Pop'] = (bar_chart_grouped_data['deaths'] / bar_chart_grouped_data['popData2020']) * 100

# Trier pour chaque graphique
cases_vs_pop_sorted = bar_chart_grouped_data.sort_values(by='% Cases vs Pop', ascending=False)
deaths_vs_cases_sorted = bar_chart_grouped_data.sort_values(by='% Deaths vs Cases', ascending=False)

### Premier graphique : % Cases vs Pop
fig_cases_vs_pop = go.Figure()

fig_cases_vs_pop.add_trace(go.Bar(
    x=cases_vs_pop_sorted['countriesAndTerritories'],
    y=cases_vs_pop_sorted['% Cases vs Pop'],
    name='% Cases vs Pop',
    marker_color='blue'
))

fig_cases_vs_pop.update_layout(
    title="% Cases vs Pop (sorted)",
    xaxis=dict(title="Country"),
    yaxis=dict(title="% Cases vs Pop"),
    template="plotly_white",
    height=500
)

# Affichage du premier graphique
st.plotly_chart(fig_cases_vs_pop, use_container_width=True)

### Deuxième graphique : % Deaths vs Cases et % Deaths vs Pop (côte à côte sur une seule échelle)
fig_deaths_metrics = go.Figure()

# Barres pour % Deaths vs Cases
fig_deaths_metrics.add_trace(go.Bar(
    x=deaths_vs_cases_sorted['countriesAndTerritories'],
    y=deaths_vs_cases_sorted['% Deaths vs Cases'],
    name='% Deaths vs Cases',
    marker_color='red'
))

# Barres pour % Deaths vs Pop
fig_deaths_metrics.add_trace(go.Bar(
    x=deaths_vs_cases_sorted['countriesAndTerritories'],
    y=deaths_vs_cases_sorted['% Deaths vs Pop'],
    name='% Deaths vs Pop',
    marker_color='green'
))

# Configuration des axes
fig_deaths_metrics.update_layout(
    title="% Deaths Metrics (sorted by % Deaths vs Cases)",
    xaxis=dict(title="Country"),
    yaxis=dict(
        title="Percentage",
        titlefont=dict(color="black"),
        tickfont=dict(color="black"),
    ),
    barmode="group",  # Barres côte à côte
    template="plotly_white",
    height=600,
    legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center")
)

# Affichage du graphique
st.plotly_chart(fig_deaths_metrics, use_container_width=True)