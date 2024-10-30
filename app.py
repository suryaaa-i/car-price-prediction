import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.let_it_rain import rain
import pandas as pd
import plotly.express as px
import pickle
from PIL import Image

# Set up the app configuration
img = Image.open("C:/Users/imand/OneDrive/Desktop/CAPSTONE_PROJECT/cardekho/Screenshot_(1).png")
st.set_page_config(page_title='Cardekho Resale Price Prediction', page_icon=img, layout='wide')

# Define the multiapp class for navigation
class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, function):
        self.apps.append({'title': title, 'function': function})

    def run(self):
        with st.sidebar:
            app = option_menu(
                'Car Resale Price Prediction',
                ["Home", "Data Filtering", "Data Analysis", "Data Prediction"],
                icons=['house', 'search', "reception-4", "dice-5-fill"],
                menu_icon='cash-coin',
                default_index=0,
                orientation="vertical",
                styles={
                    "container": {"padding": "0!important", "background-color": "#A95C68"},
                    "icon": {"color": "violet", "font-size": "20px"},
                    "nav-link": {"font-size": "18px", "text-align": "left", "margin": "0px", "--hover-color": "#C4A484"},
                    "nav-link-selected": {"background-color": "#C04000"},
                }
            )
        if app == 'Home':
            home_page()
        elif app == 'Data Filtering':
            filtering_page()
        elif app == 'Data Analysis':
            analysis_page()
        elif app == 'Data Prediction':
            prediction_page()

# Define each page as separate functions

def home_page():
    colored_header(
        label='Welcome to :orange[Home] Page 👋🏼',
        color_name='orange-70'
    )
    with st.form(key='form', clear_on_submit=False):
        st.markdown("## :orange[*Project Title*:]")
        st.subheader(" &nbsp; &nbsp; *CarDekho Used Car Price Prediction*")
        st.markdown("## :orange[*Skills take away from this project*:]")
        st.subheader(" &nbsp; &nbsp; *Python scripting, Data Wrangling, EDA, Machine Learning, Streamlit.*")
        st.markdown("## :orange[*Domain*:]")
        st.subheader(" &nbsp; &nbsp; *Automobile*")
        st.markdown("## :orange[*Problem Statement:*]")
        st.subheader(" &nbsp; &nbsp; *Predicting used car prices accurately to aid in buying/selling decisions.*")
        st.markdown("## :orange[*Results:*]")
        st.subheader(" &nbsp; &nbsp; *An accurate machine learning model to predict used car prices.*")

        if st.form_submit_button('**Click here to get Data Set Link**'):
            st.markdown("## :orange[Dataset : [Data Link](https://drive.google.com/drive/folders/16U7OH7URsCW0rf91cwyDqEgd9UoeZAJh)]")


def filtering_page():
    colored_header(
        label='You are in Data :blue[Filtering] page',
        color_name='blue-70'
    )

    @st.cache_data
    def load_data():
        return pd.read_csv('Cleaned_Car_Dheko.csv')

    df = load_data()
    df['Car_Produced_Year'] = df['Car_Produced_Year'].astype('str')
    df['Registration_Year'] = df['Registration_Year'].astype('str')

    choice = st.sidebar.selectbox(
        label='*Select a column to know unique values:*',
        options=['Car_Model', 'Manufactured_By']
    )
    
    unique_values = df[choice].unique()
    unique_values_df = pd.DataFrame({f'{choice}': unique_values})
    st.sidebar.dataframe(unique_values_df, use_container_width=True)

    filter = dataframe_explorer(df)
    if st.button('**SUBMIT**'):
        st.dataframe(filter, use_container_width=True, hide_index=True)


def analysis_page():
    colored_header(
        label='You are in Data :green[Analysis] page',
        color_name='green-70'
    )

    @st.cache_data
    def load_data():
        return pd.read_csv('Cleaned_Car_Dheko.csv')

    df = load_data()

    choice = st.selectbox("**Select an option to Explore their data**", df.drop('Car_Price', axis=1).columns)
    st.plotly_chart(px.histogram(df, x=choice, y='Car_Price', width=950, height=500))

    st.plotly_chart(px.bar(df.groupby('Manufactured_By').count().reset_index(), x='Manufactured_By', y='Fuel_Type', width=950, height=500, labels={'Fuel_Type': 'Total Count of Car Brand'}))

    # Other plot sections, similar to the above examples...


# Inverse Transformation Function
def inv_trans(x):
    return 1/x if x != 0 else 0

def prepare_input_data(km_driven, transmission, car_model, model_year, engine_cc, mileage, location):
    # This function should transform the inputs into the format expected by the model
    # For example, you may need to encode categorical variables
    # Here's a basic structure (modify as needed):
    
    # Example encoding (update these mappings as necessary)
    transmission_encoded = 1 if transmission == 'Automatic' else 0  # Example encoding
    # Add other encodings as needed...

    # Return the prepared input in the format required by your model
    return [[inv_trans(km_driven), transmission_encoded, car_model, model_year, engine_cc, mileage, location]]

# Data Prediction Page
def prediction_page():
    colored_header(label='Welcome to Data :red[Prediction] page 👋🏼', color_name='red-70', description='CarDekho Used Cars Price Prediction')
    
    @st.cache_data
    def load_data():
        df = pd.read_csv('Cleaned_Car_Dheko.csv')
        df1 = pd.read_csv('Preprocessed_Car_Dheko.csv')
        return df, df1

    df, df1 = load_data()
    # Clean the data
    df.drop(['Manufactured_By', 'No_of_Seats', 'No_of_Owners', 'Fuel_Type', 'Registration_Year', 'Car_Age'], axis=1, inplace=True)
    df1.drop(['Manufactured_By', 'No_of_Seats', 'No_of_Owners', 'Fuel_Type', 'Registration_Year', 'Car_Age'], axis=1, inplace=True)

    # Encoding categorical variables
    encodings = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            encode_vals = df1[col].sort_values().unique()  # encoded values
            decode_vals = df[col].sort_values().unique()  # original values
            encodings[col] = dict(zip(decode_vals, encode_vals))

    with st.form(key='form', clear_on_submit=False):
        car_model = st.selectbox("**Select a Car Model**", options=df['Car_Model'].unique())
        model_year = st.selectbox("**Select a Car Produced Year**", options=df['Car_Produced_Year'].unique())
        transmission = st.radio("**Select a Transmission Type**", options=df['Transmission_Type'].unique(), horizontal=True)
        location = st.selectbox("**Select a location**", options=df['Location'].unique())
        km_driven = st.number_input(f"**Enter a Kilometer Driven**", min_value=df['Kilometers_Driven'].min(), max_value=df['Kilometers_Driven'].max())
        engine_cc = st.number_input(f"**Enter an Engine CC**", min_value=df['Engine_CC'].min(), max_value=df['Engine_CC'].max())
        mileage = st.number_input(f"**Enter a Mileage**", min_value=df['Mileage(kmpl)'].min(), max_value=df['Mileage(kmpl)'].max())

        with open('GradientBoost_model.pkl', 'rb') as file:
            model = pickle.load(file)

        button = st.form_submit_button('**Predict**', use_container_width=True)

        if button:
            try:
                # Prepare input for prediction
                input_data = [[km_driven, encodings['Transmission_Type'][transmission], encodings['Car_Model'][car_model],
                               model_year, engine_cc, mileage, encodings['Location'][location]]]
                result = model.predict(input_data)
                st.markdown(f"## :green[*Predicted Car Price is {result[0]:,.2f}*]")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Initialize and run the multi-app
app = MultiApp()
app.add_app("Home", home_page)
app.add_app("Data Filtering", filtering_page)
app.add_app("Data Analysis", analysis_page)
app.add_app("Data Prediction", prediction_page)
app.run()
