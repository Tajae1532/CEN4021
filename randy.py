import streamlit as st
from streamlit_option_menu import option_menu
import requests
from datetime import date

from sportsipy.nfl.boxscore import Boxscores, Boxscore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hvplot.pandas
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#from cairo.main import displaySchedule

pd.set_option('display.max_columns', None)


def prepare_data(pred_week):
    df = pd.read_csv(r"2021_week_2_through_14.csv")
    comp_games_df = df[df['week'] < pred_week]
    pred_games_df = df[df['week'] == pred_week]
    train_df = comp_games_df
    test_df = pred_games_df
    X_test = test_df.drop(columns=['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
    return df, X_test, test_df

def logistic_regression_predict(df, X_test, pred_week):
    comp_games_df = df[df['week'] < pred_week]
    train_df = comp_games_df
    X_train = train_df.drop(columns = ['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
    y_train = train_df[['result']]

    # For unscaled predictions
    clf_unscaled = LogisticRegression()
    clf_unscaled.fit(X_train, np.ravel(y_train.values))
    y_pred_unscaled = clf_unscaled.predict_proba(X_test)
    y_pred_unscaled = y_pred_unscaled[:, 1]

    # For scaled predictions
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    clf_scaled = LogisticRegression(penalty='l1', dual=False, tol=0.001, C=1.0, fit_intercept=True,
                                   intercept_scaling=1, class_weight='balanced', random_state=None,
                                   solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0)
    clf_scaled.fit(X_train_scaled, np.ravel(y_train.values))

    y_pred_scaled = clf_scaled.predict_proba(X_test_scaled)
    y_pred_scaled = y_pred_scaled[:, 1]

    return y_pred_unscaled, y_pred_scaled


def neural_network_predict(df, X_test, pred_week):
    comp_games_df = df[df['week'] < pred_week]
    train_df = comp_games_df
    X_train = train_df.drop(columns = ['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
    y_train = train_df[['result']]

    number_input_features = X_train.shape[1]
    hidden_nodes_layer1 = (number_input_features + 1) // 2
    hidden_nodes_layer2 = (hidden_nodes_layer1 + 1) // 2
    hidden_nodes_layer3 = (hidden_nodes_layer2 + 1) // 2

    nn = Sequential()
    nn.add(Dense(units=hidden_nodes_layer1, activation='relu', input_dim=number_input_features))
    nn.add(Dense(units=hidden_nodes_layer2, activation='relu'))
    nn.add(Dense(units=hidden_nodes_layer3, activation='relu'))
    nn.add(Dense(units=1, activation='sigmoid'))
    nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    fit_model = nn.fit(X_train, y_train, epochs=500, verbose=0)

    y_pred = nn.predict(X_test)
    return y_pred

def random_forest_predict(df, X_test, pred_week):
    comp_games_df = df[df['week'] < pred_week]
    train_df = comp_games_df
    X_train = train_df.drop(columns = ['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
    y_train = train_df[['result']]

    rf_model = RandomForestClassifier(n_estimators=100, random_state=1)
    rf_fitted = rf_model.fit(X_train, np.ravel(y_train.values))

    y_pred = rf_fitted.predict_proba(X_test)
    y_pred = y_pred[:, 1]

    return y_pred


def display_results(y_pred, test_df):
    with st.container():
        with open('styles.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


        for g in range(len(y_pred)):
            win_prob = int(y_pred[g] * 100)
            away_team = test_df.reset_index().drop(columns = 'index').loc[g,'away_name']
            home_team = test_df.reset_index().drop(columns = 'index').loc[g,'home_name']

            predictionColumn, winnerColumn, loserColumn, gap = st.columns((5, 3, 3, 1), gap="large")
            with predictionColumn:
                st.subheader(f'The {away_team} have a probability of {win_prob}% of beating the {home_team}.')
            if win_prob > 50:
                with winnerColumn:
                    st.header("WIN")
                    st.image(teamLogo(away_team), use_column_width="auto")
                with loserColumn:
                    st.header("LOSE")
                    st.image(teamLogo(home_team), use_column_width="auto")
            elif win_prob < 50:
                with winnerColumn:
                    st.header("WIN")
                    st.image(teamLogo(home_team), use_column_width="auto")
                with loserColumn:
                    st.header("LOSE")
                    st.image(teamLogo(away_team), use_column_width="auto")
            else:
                with winnerColumn:
                    st.header("DRAW")
                    st.image(teamLogo(home_team), use_column_width="auto")
                with loserColumn:
                    st.header("DRAW")
                    st.image(teamLogo(away_team), use_column_width="auto")


# FUNCTIONS
# Returns Team Logo
def teamLogo(team):
    if team == "ARI" or team == "Arizona Cardinals":
        return "https://static.www.nfl.com/image/private/f_auto/league/u9fltoslqdsyao8cpm0k"
    elif team == "ATL" or team == "Atlanta Falcons":
        return "https://static.www.nfl.com/image/private/f_auto/league/d8m7hzpsbrl6pnqht8op"
    elif team == "BAL" or team == "Baltimore Ravens":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/ucsdijmddsqcj1i9tddd"
    elif team == "BUF" or team == "Buffalo Bills":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/giphcy6ie9mxbnldntsf"
    elif team == "CAR" or team == "Carolina Panthers":
        return "https://static.www.nfl.com/image/private/f_auto/league/ervfzgrqdpnc7lh5gqwq"
    elif team == "CHI" or team == "Chicago Bears":
        return "https://static.www.nfl.com/image/private/f_auto/league/ijrplti0kmzsyoaikhv1"
    elif team == "CIN" or team == "Cincinnati Bengals":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/okxpteoliyayufypqalq"
    elif team == "CLE" or team == "Cleveland Browns":
        return "https://static.www.nfl.com/image/private/f_auto/league/fgbn8acp4opvyxk13dcy"
    elif team == "DAL" or team == "Dallas Cowboys":
        return "https://static.www.nfl.com/image/private/f_auto/league/ieid8hoygzdlmzo0tnf6"
    elif team == "DEN" or team == "Denver Broncos":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/t0p7m5cjdjy18rnzzqbx"
    elif team == "DET" or team == "Detroit Lions":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/ocvxwnapdvwevupe4tpr"
    elif team == "GB" or team == "Green Bay Packers":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/gppfvr7n8gljgjaqux2x"
    elif team == "HOU" or team == "Houston Texans":
        return "https://static.www.nfl.com/image/private/f_auto/league/bpx88i8nw4nnabuq0oob"
    elif team == "IND" or team == "Indianapolis Colts":
        return "https://static.www.nfl.com/image/private/f_auto/league/ketwqeuschqzjsllbid5"
    elif team == "JAX" or team == "Jacksonville Jaguars":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/qycbib6ivrm9dqaexryk"
    elif team == "KC" or team == "Kansas City Chiefs":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/ujshjqvmnxce8m4obmvs"
    elif team == "LAC" or team == "Los Angeles Chargers":
        return "https://static.www.nfl.com/image/private/f_auto/league/ayvwcmluj2ohkdlbiegi"
    elif team == "LAR" or team == "Los Angeles Rams":
        return "https://static.www.nfl.com/image/private/f_auto/league/ayvwcmluj2ohkdlbiegi"
    elif team == "LV" or team == "Las Vegas Raiders":
        return "https://static.www.nfl.com/image/private/f_auto/league/gzcojbzcyjgubgyb6xf2"
    elif team == "MIA" or team == "Miami Dolphins":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/lits6p8ycthy9to70bnt"
    elif team == "MIN" or team == "Minnesota Vikings":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/teguylrnqqmfcwxvcmmz"
    elif team == "NE" or team == "New England Patriots":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/moyfxx3dq5pio4aiftnc"
    elif team == "NO" or team == "New Orleans Saints":
        return "https://static.www.nfl.com/image/private/f_auto/league/grhjkahghjkk17v43hdx"
    elif team == "NYG" or team == "New York Giants":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/t6mhdmgizi6qhndh8b9p"
    elif team == "NYJ" or team == "New York Jets":
        return "https://static.www.nfl.com/t_headshot_desktop_2x/f_auto/league/api/clubs/logos/NYJ"
    elif team == "PHI" or team == "Philadelphia Eagles":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/puhrqgj71gobgdkdo6uq"
    elif team == "PIT" or team == "Pittsburgh Steelers":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/xujg9t3t4u5nmjgr54wx"
    elif team == "SEA" or team == "Seattle Seahawks":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/gcytzwpjdzbpwnwxincg"
    elif team == "SF" or team == "San Francisco 49ers":
        return "https://res.cloudinary.com/nflleague/image/private/f_auto/league/dxibuyxbk0b9ua5ih9hn"
    elif team == "TB" or team == "Tampa Bay Buccaneers":
        return "https://static.www.nfl.com/image/private/f_auto/league/v8uqiualryypwqgvwcih"
    elif team == "TEN" or team == "Tennessee Titans":
        return "https://static.www.nfl.com/image/private/f_auto/league/pln44vuzugjgipyidsre"
    elif team == "WSH" or team == "Washington Football Team":
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
    options=["Home", "News", "Schedule", "Team Stats", "Standings", "Predictions"],
    icons=["house", "book", "calendar-event", "triangle", "app", "circle"],
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

            awayTeamColumn, gameInfoColumn, homeTeamColumn = st.columns((1, 2, 1))

            # DISPLAYS AWAY TEAM'S LOGO
            with awayTeamColumn:
                st.image(teamLogo(dailyScoreboardResponse["body"][game]["away"]), use_column_width="auto")

            # DISPLAYS GAME'S INFO
            with gameInfoColumn:
                # Finds Away Team's Name Given Team's Abbreviation
                for team in nflTeamsResponse["body"]:
                    if dailyScoreboardResponse["body"][game]["away"] == team["teamAbv"]:
                        awayTeam = team["teamName"]
                        break
                # Finds Away Team's Name Given Team's Abbreviation
                for team in nflTeamsResponse["body"]:
                    if dailyScoreboardResponse["body"][game]["home"] == team["teamAbv"]:
                        homeTeam = team["teamName"]
                        break
                # Scores
                awayTeamScore = dailyScoreboardResponse["body"][game]["awayPts"]
                homeTeamScore = dailyScoreboardResponse["body"][game]["homePts"]
                # Game Status
                gameStatus = dailyScoreboardResponse["body"][game]["gameStatus"]

                st.header(awayTeam + " @ " + homeTeam)
                st.subheader("(" + awayTeamScore + " - " + homeTeamScore + ")")
                if gameStatus == "Completed":
                    if (awayTeamScore > homeTeamScore):
                        st.subheader("WINNER: " + awayTeam)
                    elif (homeTeamScore > awayTeamScore):
                        st.subheader("WINNER: " + homeTeam)
                    else:
                        st.subheader("DRAW")
                else:
                    st.subheader("LIVE")

            # DISPLAYS HOME TEAM'S LOGO
            with homeTeamColumn:
                st.image(teamLogo(dailyScoreboardResponse["body"][game]["home"]), use_column_width="auto")


# NEWS
elif menu_selection == "News":

    newsURL = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLNews"

    querystring = {"recentNews": "true", "maxItems": "10"}

    headers = {
        "X-RapidAPI-Key": "078585a2e1msheaec933132a44a7p1c1c95jsn31046989ccf4",
        "X-RapidAPI-Host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
    }


    newsResponse = requests.get(newsURL, headers=headers, params=querystring).json()

    # Loops Through all the Most Recent News Articles
    for news in newsResponse["body"]:
        with st.container():
            with open('styles.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

            imageColumn, headlineColumn = st.columns((1, 2))

            # Prints Article's Image
            with imageColumn:
                st.image(news["image"], use_column_width="auto")

            # Prints Article's Title
            with headlineColumn:
                st.subheader(news["title"] + "[[...]](" + news["link"] + ")")


# SCHEDULE
elif menu_selection == "Schedule":
    #displaySchedule()
    st.write("under construction")

# STATS
elif menu_selection == "Team Stats":
    st.header("Teams Stats")
    
     # API CALL FOR NFL TEAMS
    nflTeamsURL = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLTeams"
    nflTeamsQuery = {"rosters": "true", "schedules": "true", "topPerformers": "true", "teamStats": "true"}
    teamHeaders = {
        "X-RapidAPI-Key": "8917d83bacmshb156cbb3d6abfa3p1c23c0jsnc18a9dcfd536",
        "X-RapidAPI-Host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
    }
    
    nflTeams_Response = requests.get(nflTeamsURL, params = nflTeamsQuery, headers = teamHeaders).json()
    
    teamData = []
    
    for index in nflTeams_Response["body"]:
        
        if index["currentStreak"]["result"] == "W":
            streak_Type = "Win"
        else:
            streak_Type = "Lose"
        
        teamData.append(
            {
                'logo': teamLogo(index["teamAbv"]),
                'name': index["teamCity"] + " " + index["teamName"],
                'wins': index["wins"],
                'losses': index["loss"],
                'tie': index["tie"],
                'points for': index["pf"],
                'points against': index["pa"],
                'streak': streak_Type,
                'streak length': index["currentStreak"]["length"]
                
            }
        )
   
    df = pd.DataFrame(teamData)
      
     
    
    st.dataframe(
        df, 
        column_config ={
            "logo": st.column_config.ImageColumn(
                "Logo",help = "Team Logo"
            ),
            "name": st.column_config.TextColumn("Name"),
            "wins": st.column_config.NumberColumn("Wins"),
            "Losses": st.column_config.NumberColumn("Losses"),
            "tie": st.column_config.NumberColumn("Ties"),
            "points for": st.column_config.NumberColumn("Points for"),
            "points against": st.column_config.NumberColumn("Points Against"),
            "streak": st.column_config.TextColumn("Streak"),
            "streak length": st.column_config.NumberColumn("Length")
            
        },
        height = 1158,
        hide_index = True,
        use_container_width = True
        )

# STANDINGS
elif menu_selection == "Standings":
    st.header("Standings")
    
    # Requesting NFL Data 
    nflTeamsURL = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLTeams"
    nflTeamsQuery = {"rosters": "true", "schedules": "true", "topPerformers": "true", "teamStats": "true"}
    teamHeaders = {
        "X-RapidAPI-Key": "8917d83bacmshb156cbb3d6abfa3p1c23c0jsnc18a9dcfd536",
        "X-RapidAPI-Host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
    }
    
    
    nflTeams_Response = requests.get(nflTeamsURL, params = nflTeamsQuery, headers = teamHeaders).json()
    
    # Creating a variable to shorten the name length
    body = nflTeams_Response["body"]
    
    # Creating the Dataframes for American Football Conference Teams
    AFC_East = pd.DataFrame(columns= ['Logo', 'Name','Wins','Losses', 'Ties'])
    AFC_West = pd.DataFrame(columns= ['Logo', 'Name','Wins','Losses', 'Ties'])
    AFC_North = pd.DataFrame(columns= ['Logo', 'Name','Wins','Losses', 'Ties'])
    AFC_South = pd.DataFrame(columns= ['Logo', 'Name','Wins','Losses', 'Ties'])
    
    
    #Filling in all the data for the AFC Teams based on division
    #Logos
    AFC_East['Logo'] = [
        teamLogo(body[19]["teamAbv"]),teamLogo(body[3]["teamAbv"]),
        teamLogo(body[24]["teamAbv"]),teamLogo(body[21]["teamAbv"])
    ]
    
    #Names
    AFC_East['Name'] = [
        body[19]["teamCity"] + " " + body[19]["teamName"],
        body[3]["teamCity"] + " " + body[3]["teamName"],
        body[24]["teamCity"] + " " + body[24]["teamName"],
        body[21]["teamCity"] + " " + body[21]["teamName"]
    ]
    
    #Wins
    AFC_East['Wins'] = [
        body[19]["wins"], body[3]["wins"], body[24]["wins"], body[21]["wins"]
    ]
    
    #Losses
    AFC_East['Losses'] = [
        body[19]["loss"], body[3]["loss"], body[24]["loss"], body[21]["loss"]
    ]
    
    #Ties
    AFC_East['Ties'] = [
        body[19]["tie"], body[3]["tie"], body[24]["tie"], body[21]["tie"]
    ]
    
    AFC_West['Logo'] = [
        teamLogo(body[15]["teamAbv"]), teamLogo(body[16]["teamAbv"]),
        teamLogo(body[17]["teamAbv"]), teamLogo(body[9]["teamAbv"])
    ]
    
    AFC_West['Name'] = [
        body[15]["teamCity"] + " " + body[15]["teamName"],
        body[16]["teamCity"] + " " + body[16]["teamName"],
        body[17]["teamCity"] + " " + body[17]["teamName"],
        body[9]["teamCity"] + " " + body[9]["teamName"]
    ]
    
    AFC_West['Wins'] = [
        body[15]["wins"], body[16]["wins"], body[17]["wins"], body[9]["wins"]
    ]
    
    AFC_West['Losses'] = [
        body[15]["loss"], body[16]["loss"], body[17]["loss"], body[9]["loss"]
    ]
    
    AFC_West['Ties'] = [
        body[15]["tie"], body[16]["tie"], body[17]["tie"], body[9]["tie"]
    ]
    
    AFC_North['Logo'] = [
        teamLogo(body[2]["teamAbv"]), teamLogo(body[25]["teamAbv"]),
        teamLogo(body[7]["teamAbv"]), teamLogo(body[6]["teamAbv"])
    ]
    
    AFC_North['Name'] = [
        body[2]["teamCity"] + " " + body[2]["teamName"],
        body[25]["teamCity"] + " " + body[25]["teamName"],
        body[7]["teamCity"] + " " + body[7]["teamName"],
        body[6]["teamCity"] + " " + body[6]["teamName"]
    ]
    
    AFC_North['Wins'] = [
        body[2]["wins"], body[25]["wins"], body[7]["wins"], body[6]["wins"]
    ]
    
    AFC_North['Losses'] = [
        body[2]["loss"], body[25]["loss"], body[7]["loss"], body[6]["loss"]
    ]
    
    AFC_North['Ties'] = [
        body[2]["tie"], body[25]["tie"], body[7]["tie"], body[6]["tie"]
    ]
     
    AFC_South['Logo'] = [
        teamLogo(body[14]["teamAbv"]),teamLogo(body[12]["teamAbv"]),
        teamLogo(body[13]["teamAbv"]),teamLogo(body[30]["teamAbv"])
    ]
    
    AFC_South['Name'] = [
        body[14]["teamCity"] + " " + body[14]["teamName"],
        body[12]["teamCity"] + " " + body[12]["teamName"],
        body[13]["teamCity"] + " " + body[13]["teamName"],
        body[30]["teamCity"] + " " + body[30]["teamName"]
    ]
    
    AFC_South['Wins'] = [
        body[14]["wins"], body[12]["wins"], body[13]["wins"], body[30]["wins"]
    ]
    
    AFC_South['Losses'] = [
        body[14]["loss"], body[12]["loss"], body[13]["loss"], body[30]["loss"]
    ]
    
    AFC_South['Ties'] = [
        body[14]["tie"], body[12]["tie"], body[13]["tie"], body[30]["tie"]
    ]
    
    #Creating Dataframes for the National Football Conference
    NFC_East = pd.DataFrame(columns= ['Logo', 'Name','Wins','Losses', 'Ties'])
    NFC_West = pd.DataFrame(columns= ['Logo', 'Name','Wins','Losses', 'Ties'])
    NFC_North = pd.DataFrame(columns= ['Logo', 'Name','Wins','Losses', 'Ties'])
    NFC_South = pd.DataFrame(columns= ['Logo', 'Name','Wins','Losses', 'Ties'])
    
    #Filling out the data for the NFC teams
    #Logo
    NFC_East['Logo'] = [
        teamLogo(body[26]["teamAbv"]),teamLogo(body[8]["teamAbv"]),
        teamLogo(body[31]["teamAbv"]),teamLogo(body[23]["teamAbv"])
    ]
    
    #Name
    NFC_East['Name'] = [
        body[26]["teamCity"] + " " + body[26]["teamName"],
        body[8]["teamCity"] + " " + body[8]["teamName"],
        body[31]["teamCity"] + " " + body[31]["teamName"],
        body[23]["teamCity"] + " " + body[23]["teamName"]
    ]
    
    #Wins
    NFC_East['Wins'] = [
        body[26]["wins"], body[8]["wins"], body[31]["wins"], body[23]["wins"]
    ]
    
    #Losses
    NFC_East['Losses'] = [
        body[26]["loss"], body[8]["loss"], body[31]["loss"], body[23]["loss"]
    ]
    
    #Ties
    NFC_East['Ties'] = [
        body[26]["tie"], body[8]["tie"], body[31]["tie"], body[23]["tie"]
    ]
    
    NFC_West['Logo'] = [
        teamLogo(body[27]["teamAbv"]), teamLogo(body[28]["teamAbv"]),
        teamLogo(body[18]["teamAbv"]), teamLogo(body[0]["teamAbv"])
    ]
    
    NFC_West['Name'] = [
        body[27]["teamCity"] + " " + body[27]["teamName"],
        body[28]["teamCity"] + " " + body[28]["teamName"],
        body[18]["teamCity"] + " " + body[18]["teamName"],
        body[0]["teamCity"] + " " + body[0]["teamName"]
    ]
    
    NFC_West['Wins'] = [
        body[27]["wins"], body[28]["wins"], body[18]["wins"], body[0]["wins"]
    ]
    
    NFC_West['Losses'] = [
        body[27]["loss"], body[28]["loss"], body[18]["loss"], body[0]["loss"]
    ]
    
    NFC_West['Ties'] = [
        body[27]["tie"], body[28]["tie"], body[18]["tie"], body[0]["tie"]
    ]
    
    NFC_North['Logo'] = [
        teamLogo(body[10]["teamAbv"]), teamLogo(body[20]["teamAbv"]),
        teamLogo(body[11]["teamAbv"]), teamLogo(body[5]["teamAbv"])
    ]
    
    NFC_North['Name'] = [
        body[10]["teamCity"] + " " + body[10]["teamName"],
        body[20]["teamCity"] + " " + body[20]["teamName"],
        body[11]["teamCity"] + " " + body[11]["teamName"],
        body[5]["teamCity"] + " " + body[5]["teamName"]
    ]
    
    NFC_North['Wins'] = [
        body[10]["wins"], body[20]["wins"], body[11]["wins"], body[5]["wins"]
    ]
    
    NFC_North['Losses'] = [
        body[10]["loss"], body[20]["loss"], body[11]["loss"], body[5]["loss"]
    ]
    
    NFC_North['Ties'] = [
        body[10]["tie"], body[20]["tie"], body[11]["tie"], body[5]["tie"]
    ]
     
    NFC_South['Logo'] = [
        teamLogo(body[22]["teamAbv"]),teamLogo(body[29]["teamAbv"]),
        teamLogo(body[1]["teamAbv"]),teamLogo(body[4]["teamAbv"])
    ]
    
    NFC_South['Name'] = [
        body[22]["teamCity"] + " " + body[22]["teamName"],
        body[29]["teamCity"] + " " + body[29]["teamName"],
        body[1]["teamCity"] + " " + body[1]["teamName"],
        body[4]["teamCity"] + " " + body[4]["teamName"]
    ]
    
    NFC_South['Wins'] = [
        body[22]["wins"], body[29]["wins"], body[1]["wins"], body[4]["wins"]
    ]
    
    NFC_South['Losses'] = [
        body[22]["loss"], body[29]["loss"], body[1]["loss"], body[4]["loss"]
    ]
    
    NFC_South['Ties'] = [
        body[22]["tie"], body[29]["tie"], body[1]["tie"], body[4]["tie"]
    ]
    
    #Sorting the Dataframes by the number of wins each teams have
    AFC_East.sort_values("Wins")
    AFC_West.sort_values("Wins")
    AFC_North.sort_values("Wins")
    AFC_South.sort_values("Wins")
    NFC_East.sort_values("Wins")
    NFC_West.sort_values("Wins")
    NFC_North.sort_values("Wins")
    NFC_South.sort_values("Wins")
    
    st.subheader("AFC EAST")
    
    #Converting the pandas dataframe into a streamlit dataframe for presentation
    st.dataframe(AFC_East, 
            column_config ={
                "Logo": st.column_config.ImageColumn(
                    "Logo", help= "Team Logo"
                ),
            "Name": st.column_config.TextColumn("Name"),
            "Wins": st.column_config.NumberColumn("Wins"),
            "Losses": st.column_config.NumberColumn("Losses"),
            "Ties": st.column_config.NumberColumn("Ties"),
                
            },
            hide_index = True
            )
    
    st.subheader("AFC WEST")
    
    st.dataframe(AFC_West, 
            column_config ={
                "Logo": st.column_config.ImageColumn(
                    "Logo", help= "Team Logo"
                ),
            "Name": st.column_config.TextColumn("Name"),
            "Wins": st.column_config.NumberColumn("Wins"),
            "Losses": st.column_config.NumberColumn("Losses"),
            "Ties": st.column_config.NumberColumn("Ties"),
                
            },
            hide_index = True
            )
    
    st.subheader("AFC NORTH")
    
    st.dataframe(AFC_North, 
            column_config ={
                "Logo": st.column_config.ImageColumn(
                    "Logo", help= "Team Logo"
                ),
            "Name": st.column_config.TextColumn("Name"),
            "Wins": st.column_config.NumberColumn("Wins"),
            "Losses": st.column_config.NumberColumn("Losses"),
            "Ties": st.column_config.NumberColumn("Ties"),
                
            },
            hide_index = True
            )
    
    st.subheader("AFC SOUTH")
    
    st.dataframe(AFC_South, 
            column_config ={
                "Logo": st.column_config.ImageColumn(
                    "Logo", help= "Team Logo"
                ),
            "Name": st.column_config.TextColumn("Name"),
            "Wins": st.column_config.NumberColumn("Wins"),
            "Losses": st.column_config.NumberColumn("Losses"),
            "Ties": st.column_config.NumberColumn("Ties"),
                
            },
            hide_index = True
            )
    
    st.subheader("NFC_EAST")
    
    st.dataframe(NFC_East, 
            column_config ={
                "Logo": st.column_config.ImageColumn(
                    "Logo", help= "Team Logo"
                ),
            "Name": st.column_config.TextColumn("Name"),
            "Wins": st.column_config.NumberColumn("Wins"),
            "Losses": st.column_config.NumberColumn("Losses"),
            "Ties": st.column_config.NumberColumn("Ties"),
                
            },
            hide_index = True
            )
    
    st.subheader("NFC WEST")
    
    st.dataframe(NFC_West, 
            column_config ={
                "Logo": st.column_config.ImageColumn(
                    "Logo", help= "Team Logo"
                ),
            "Name": st.column_config.TextColumn("Name"),
            "Wins": st.column_config.NumberColumn("Wins"),
            "Losses": st.column_config.NumberColumn("Losses"),
            "Ties": st.column_config.NumberColumn("Ties"),
                
            },
            hide_index = True
            )
    
    st.subheader("NFC NORTH")
    
    st.dataframe(NFC_North, 
            column_config ={
                "Logo": st.column_config.ImageColumn(
                    "Logo", help= "Team Logo"
                ),
            "Name": st.column_config.TextColumn("Name"),
            "Wins": st.column_config.NumberColumn("Wins"),
            "Losses": st.column_config.NumberColumn("Losses"),
            "Ties": st.column_config.NumberColumn("Ties"),
                
            },
            hide_index = True
            )
    
    st.subheader("NFC SOUTH")
    
    st.dataframe(NFC_South, 
            column_config ={
                "Logo": st.column_config.ImageColumn(
                    "Logo", help= "Team Logo"
                ),
            "Name": st.column_config.TextColumn("Name"),
            "Wins": st.column_config.NumberColumn("Wins"),
            "Losses": st.column_config.NumberColumn("Losses"),
            "Ties": st.column_config.NumberColumn("Ties"),
            },
            hide_index = True
            )    

# PREDICTIONS
elif menu_selection == "Predictions":
    st.header("NFL Game Prediction")

    with st.container():
        with open('styles.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

        spacingColumn, radioColumn, sliderColumn, secondspacingColumn = st.columns((1, 5, 5, 1))
        # Let user select the prediction method
        with radioColumn:
            method = st.radio("Select a prediction method", ["Logistic Regression", "Neural Network", "Random Forest"])
        with sliderColumn:
            pred_week = st.slider("Select week for prediction", 3, 14)
    # Load and preprocess data
    df, X_test, test_df = prepare_data(pred_week)


    if method == "Logistic Regression":
        # Run your logistic regression code and display the results
        y_pred_unscaled, y_pred_scaled = logistic_regression_predict(df, X_test, pred_week)
        st.subheader("Logistic Regression - Unscaled\n")
        display_results(y_pred_unscaled, test_df)
        st.subheader("\nLogistic Regression - Scaled\n")
        display_results(y_pred_scaled, test_df)

    elif method == "Neural Network":
        # Run your neural network code and display the results
        y_pred = neural_network_predict(df, X_test, pred_week)
        st.subheader("Neural Network Predictions\n")
        display_results(y_pred, test_df)

    elif method == "Random Forest":
        # Run your random forest code and display the results
        y_pred = random_forest_predict(df, X_test, pred_week)
        st.subheader("Random Forest Predictions\n")
        display_results(y_pred, test_df)