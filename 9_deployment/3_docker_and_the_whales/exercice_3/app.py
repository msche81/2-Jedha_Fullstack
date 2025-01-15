import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np

### Config
st.set_page_config(
    page_title="E-commerce",
    page_icon="💸 ",
    layout="wide"
)

DATA_URL = ('https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/e-commerce_data.csv')

### App
st.title("Build dashboards with Streamlit 🎨")

st.markdown("""
    Welcome to this awesome `streamlit` dashboard. This library is great to build very fast and
    intuitive charts and application running on the web. Here is a showcase of what you can do with
    it. Our data comes from an e-commerce website that simply displays samples of customer sales. Let's check it out.

    Also, if you want to have a real quick overview of what streamlit is all about, feel free to watch the below video 👇
""")

with st.expander("⏯️ Watch this 15min tutorial"):
    st.video("https://youtu.be/B2iAodr0fOo")

st.markdown("---")


# Use `st.cache` when loading data is extremly useful
# because it will cache your data so that your app 
# won't have to reload it each time you refresh your app
@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    data["Date"] = data["Date"].apply(lambda x: pd.to_datetime(",".join(x.split(",")[-2:])))
    data["currency"] = data["currency"].apply(lambda x: pd.to_numeric(x[1:]))
    return data

st.subheader("Load and showcase data")
st.markdown("""

    You can use the usual Data Science libraries like `pandas` or `numpy` to load data. 
    Then simply use [`st.write()`](https://docs.streamlit.io/library/api-reference/write-magic/st.write) to showcase it on your web app. 

""")

data_load_state = st.text('Loading data...')
data = load_data(1000)
data_load_state.text("") # change text from "Loading data..." to "" once the the load_data function has run

## Run the below code if the check is checked ✅
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)    

## Simple bar chart
st.subheader("Simple bar chart built directly with Streamlit")
st.markdown("""
    You can build simple chart directly with streamlit using:

    * [`st.bar_chart`](https://docs.streamlit.io/library/api-reference/charts/st.bar_chart)
    * [`st.line_chart`](https://docs.streamlit.io/library/api-reference/charts/st.line_chart)
    * [`st.area_chart`](https://docs.streamlit.io/library/api-reference/charts/st.area_chart)

    Eventhough it doesn't offer great flexibility, it is still a very simple way to create graphs quickly since 
    streamlit since these methods accepts a pandas DataFrame or Numpy array as input.

""")
currency_per_country = data.set_index("country")["currency"]
st.bar_chart(currency_per_country)

## Bar chart built with plotly 
st.subheader("Simple bar chart built with Plotly")
st.markdown("""
    Now, the best thing about `streamlit` is its compatibility with other libraries. For example, you
    don't need to actually use built-in charts to create your dashboard, you can use :
    
    * [`plotly`](https://docs.streamlit.io/library/api-reference/charts/st.plotly_chart) 
    * [`matplotlib`](https://docs.streamlit.io/library/api-reference/charts/st.pyplot)
    * [`bokeh`](https://docs.streamlit.io/library/api-reference/charts/st.bokeh_chart)
    * ...

    This way, you have all the flexibility you need to build awesome dashboards. 🥰

""")
fig = px.histogram(data.sort_values("country"), x="country", y="currency", barmode="group")
st.plotly_chart(fig, use_container_width=True)


### Input data 
st.subheader("Input data")
st.markdown("""
    As a final note, you can use data that a user will insert when he/she interacts with your app.
    This is called *input data*. To collect these, you can do two things:
    * [Use any of the input widget](https://docs.streamlit.io/library/api-reference/widgets)
    * [Build a form](https://docs.streamlit.io/library/api-reference/control-flow/st.form)

    Depending on what you need to do, you will prefer one or the other. With a `form`, you actually group
    input widgets together and send the data right away, which can be useful when you need to filter
    by several variables.

""")

#### Create two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("**1️⃣ Example of input widget**")
    country = st.selectbox("Select a country you want to see all time sales", data["country"].sort_values().unique())
    
    country_sales = data[data["country"]==country]
    fig = px.histogram(country_sales, x="Date", y="currency")
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**2️⃣ Example of input form**")

    with st.form("average_sales_per_country"):
        country = st.selectbox("Select a country you want to see sales", data["country"].sort_values().unique())
        start_period = st.date_input("Select a start date you want to see your metric")
        end_period = st.date_input("Select an end date you want to see your metric")
        submit = st.form_submit_button("submit")

        if submit:
            avg_period_country_sales = data[(data["country"]==country)]
            start_period, end_period = pd.to_datetime(start_period), pd.to_datetime(end_period)
            mask = (avg_period_country_sales["Date"] > start_period) & (avg_period_country_sales["Date"] < end_period)
            avg_period_country_sales = avg_period_country_sales[mask].mean()
            st.metric("Average sales during selected period (in $)", np.round(avg_period_country_sales, 2))


### Side bar 
st.sidebar.header("Build dashboards with Streamlit")
st.sidebar.markdown("""
    * [Load and showcase data](#load-and-showcase-data)
    * [Charts directly built with Streamlit](#simple-bar-chart-built-directly-with-streamlit)
    * [Charts built with Plotly](#simple-bar-chart-built-with-plotly)
    * [Input Data](#input-data)
""")
e = st.sidebar.empty()
e.write("")
st.sidebar.write("Made with 💖 by [Jedha](https://jedha.co)")



### Footer 
empty_space, footer = st.columns([1, 2])

with empty_space:
    st.write("")

with footer:
    st.markdown("""
        🍇
        If you want to learn more, check out [streamlit's documentation](https://docs.streamlit.io/) 📖
    """)