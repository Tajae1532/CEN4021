import streamlit as st
from streamlit_option_menu import option_menu
import requests
from datetime import date


# PAGE SETUP
st.set_page_config(
    page_title="NFL X Sports Betting",
    page_icon="https://www.google.com/url?sa=i&url=https%3A%2F%2Fpngimg.com%2Fimage%2F37762&psig=AOvVaw1UGJr9PfUTTFWp7e4W5363&ust=1698041913884000&source=images&cd=vfe&opi=89978449&ved=0CBAQjRxqFwoTCMjbypqBiYIDFQAAAAAdAAAAABAO",
    # menu_items =
    layout="wide",
)

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] 
{
background-image: url("https://images.unsplash.com/photo-1557174949-3b1f5b2e8fac?auto=format&fit=crop&q=80&w=1527&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");

background-position: cover;
background-size: 100%;
background-position: center; 

}


[data-testid="stHeader"] {
background: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
right: 2rem;
}

[class="block-container st-emotion-cache-z5fcl4 ea3mdgi4"]{
padding-top: 0;
}


</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


st.title("NFL X Sports Betting.")
st.subheader("A Hub for NFL Sports Betters")

# MENU SETUP
menu_selection = option_menu(
    menu_title=None,
    options=["Home", "News", "Schedule", "Prediction Bot"],
    icons=["house", "book", "calendar-event",  "circle"],
    default_index=0,
    orientation="horizontal",
)

# MENU
if menu_selection == "Home":
    st.header("Today's Games")

    todaysDate = date.today()
    dailyScoreboardURL = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLScoresOnly"
    todaysGameQuery = {"gameDate": "20221211", "topPerformers": "true"}
    headers = {
        "X-RapidAPI-Key": "078585a2e1msheaec933132a44a7p1c1c95jsn31046989ccf4",
        "X-RapidAPI-Host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
    }
    dailyScoreboardResponse = requests.get(dailyScoreboardURL, headers=headers, params=todaysGameQuery)
    st.write(dailyScoreboardResponse.json())

    for i in dailyScoreboardResponse.json["body"]:
        with st.container():
            text_container = st.container()
            with text_container:
                st.subheader(i["home"] + "VS" + i["away"])
                st.write("Time: " + i["gameTime"])
                st.write("SCORE: ")
                st.write(i["home"] + ": " + i["homePts"])
                st.write(i["away"] + ": " + i["awayPts"])

elif menu_selection == "News":
    st.write("Under Construction")

elif menu_selection == "Schedule":
    st.write("Under Construction")

elif menu_selection == "Prediction Bot":
    st.write("Under Construction")
