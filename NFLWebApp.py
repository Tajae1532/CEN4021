import streamlit as st
from streamlit_option_menu import option_menu
import requests
from datetime import date

# FUNCTIONS
# Returns Team Logo Given Team Abbreviation
def teamLogo(team):
    if team == "ARI":
        return "https://static.www.nfl.com/image/private/f_auto/league/u9fltoslqdsyao8cpm0k"
    elif team == "ATL":
        return "https://static.www.nfl.com/image/private/f_auto/league/d8m7hzpsbrl6pnqht8op"
    elif team == "BAL":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/ucsdijmddsqcj1i9tddd"
    elif team == "BUF":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/giphcy6ie9mxbnldntsf"
    elif team == "CAR":
        return "https://static.www.nfl.com/image/private/f_auto/league/ervfzgrqdpnc7lh5gqwq"
    elif team == "CHI":
        return "https://static.www.nfl.com/image/private/f_auto/league/ijrplti0kmzsyoaikhv1"
    elif team == "CIN":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/okxpteoliyayufypqalq"
    elif team == "CLE":
        return "https://static.www.nfl.com/image/private/f_auto/league/fgbn8acp4opvyxk13dcy"
    elif team == "DAL":
        return "https://static.www.nfl.com/image/private/f_auto/league/ieid8hoygzdlmzo0tnf6"
    elif team == "DEN":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/t0p7m5cjdjy18rnzzqbx"
    elif team == "DET":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/ocvxwnapdvwevupe4tpr"
    elif team == "GB":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/gppfvr7n8gljgjaqux2x"
    elif team == "HOU":
        return "https://static.www.nfl.com/image/private/f_auto/league/bpx88i8nw4nnabuq0oob"
    elif team == "IND":
        return "https://static.www.nfl.com/image/private/f_auto/league/ketwqeuschqzjsllbid5"
    elif team == "JAX":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/qycbib6ivrm9dqaexryk"
    elif team == "KC":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/ujshjqvmnxce8m4obmvs"
    elif team == "LAC":
        return "https://static.www.nfl.com/image/private/f_auto/league/ayvwcmluj2ohkdlbiegi"
    elif team == "LAR":
        return "https://static.www.nfl.com/image/private/f_auto/league/ayvwcmluj2ohkdlbiegi"
    elif team == "LV":
        return "https://static.www.nfl.com/image/private/f_auto/league/gzcojbzcyjgubgyb6xf2"
    elif team == "MIA":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/lits6p8ycthy9to70bnt"
    elif team == "MIN":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/teguylrnqqmfcwxvcmmz"
    elif team == "NE":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/moyfxx3dq5pio4aiftnc"
    elif team == "NO":
        return "https://static.www.nfl.com/image/private/f_auto/league/grhjkahghjkk17v43hdx"
    elif team == "NYG":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/t6mhdmgizi6qhndh8b9p"
    elif team == "NYJ":
        return "https://static.www.nfl.com/t_headshot_desktop_2x/f_auto/league/api/clubs/logos/NYJ"
    elif team == "PHI":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/puhrqgj71gobgdkdo6uq"
    elif team == "PIT":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/xujg9t3t4u5nmjgr54wx"
    elif team == "SEA":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/gcytzwpjdzbpwnwxincg"
    elif team == "SF":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/dxibuyxbk0b9ua5ih9hn"
    elif team == "TB":
        return "https://static.www.nfl.com/image/private/f_auto/league/v8uqiualryypwqgvwcih"
    elif team == "TEN":
        return "https://static.www.nfl.com/image/private/f_auto/league/pln44vuzugjgipyidsre"
    elif team == "WSH":
        return "https://static.www.nfl.com/image/private/f_auto/league/xymxwrxtyj9fhaemhdyd"


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
    icons=["house", "book", "calendar-event", "circle"],
    default_index=0,
    orientation="horizontal",
)

# MENU
if menu_selection == "Home":
    st.header("Today's Games")

    todaysDate = date.today()
    # API CALL FOR TODAY'S GAMES
    dailyScoreboardURL = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLScoresOnly"
    todaysGamesQuery = {"gameDate": "20221211", "topPerformers": "true"}
    todaysGamesHeaders = {
        "X-RapidAPI-Key": "078585a2e1msheaec933132a44a7p1c1c95jsn31046989ccf4",
        "X-RapidAPI-Host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
    }
    # API CALL FOR NFL TEAMS
    nflTeamsURL = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLTeams"
    nflTeamsQuery = {"rosters": "true", "schedules": "true", "topPerformers": "true", "teamStats": "true"}
    teamHeaders = {
        "X-RapidAPI-Key": "078585a2e1msheaec933132a44a7p1c1c95jsn31046989ccf4",
        "X-RapidAPI-Host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
    }

    dailyScoreboardResponse = requests.get(dailyScoreboardURL, headers=todaysGamesHeaders, params=todaysGamesQuery).json()
    nflTeamsResponse = requests.get(nflTeamsURL, headers=teamHeaders, params=nflTeamsQuery).json()

    # Loops Through All of Today's Games
    for game in dailyScoreboardResponse["body"]:
        with st.container():
            with open('styles.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

            homeTeamColumn, gameInfoColumn, awayTeamColumn = st.columns((2, 1, 2))

            # DISPLAYS AWAY TEAM'S LOGO
            with awayTeamColumn:
                st.image(teamLogo(dailyScoreboardResponse["body"][game]["away"]), use_column_width="auto")

            # DISPLAYS GAME'S INFO
            with gameInfoColumn:
                # Finds Away Team's Name Given Team's Abbreviation
                for team in nflTeamsResponse:
                    if dailyScoreboardResponse["body"][game]["away"] == nflTeamsResponse["body"][team]["teamAbv"]:
                        awayTeam = nflTeamsResponse["body"][team]["teamName"]
                        break
                # Finds Away Team's Name Given Team's Abbreviation
                for team in nflTeamsResponse:
                    if dailyScoreboardResponse["body"][game]["home"] == nflTeamsResponse["body"][team]["teamAbv"]:
                        homeTeam = nflTeamsResponse["body"][team]["teamName"]
                        break
                # Scores
                awayTeamScore = dailyScoreboardResponse["body"][game]["awayPts"]
                homeTeamScore = dailyScoreboardResponse["body"][game]["homePts"]
                # Game Status
                gameStatus = dailyScoreboardResponse["body"][game]["gameStatus"]

                st.header(awayTeam + " @ " + homeTeam)
                st.subheader("(" + awayTeamScore + " - " + homeTeamScore + ")")
                if gameStatus == "Completed":
                    if(awayTeamScore > homeTeamScore):
                        st.subheader("WINNER: " + awayTeam)
                    elif(homeTeamScore > awayTeamScore):
                        st.subheader("WINNER: " + homeTeam)
                    else:
                        st.subheader("DRAW")
                else:
                    st.subheader("LIVE")

            # DISPLAYS HOME TEAM'S LOGO
            with homeTeamColumn:
                st.image(teamLogo(dailyScoreboardResponse["body"][game]["home"]), use_column_width="auto")



elif menu_selection == "News":
    st.write("Under Construction")

elif menu_selection == "Schedule":
    st.write("Under Construction")

elif menu_selection == "Prediction Bot":
    st.write("Under Construction")
