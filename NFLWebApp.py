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

pd.set_option('display.max_columns', None)


def prepare_data():
    df = pd.read_csv(r"2021_week_2_through_14.csv")
    pred_week = 14
    comp_games_df = df[df['week'] < pred_week]
    pred_games_df = df[df['week'] == pred_week]
    train_df = comp_games_df
    test_df = pred_games_df
    X_test = test_df.drop(columns=['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
    return df, X_test, test_df, pred_week

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

    #dailyScoreboardResponse = requests.get(dailyScoreboardURL, headers=todaysGamesHeaders, params=todaysGamesQuery).json()
    #nflTeamsResponse = requests.get(nflTeamsURL, headers=teamHeaders, params=nflTeamsQuery).json()

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
    st.header("NFL Game Prediction")

    with st.container():
        with open('styles.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

        spacingColumn, radioColumn, sliderColumn, secondspacingColumn = st.columns((1, 5, 5, 1))
        # Let user select the prediction method
        with radioColumn:
            method = st.radio("Select a prediction method", ["Logistic Regression", "Neural Network", "Random Forest"])
        with sliderColumn:
            st.slider("Select week for prediction", 1, 14, 7)
    # Load and preprocess data
    df, X_test, test_df, pred_week = prepare_data()


    if method == "Logistic Regression":
        # Run your logistic regression code and display the results
        y_pred_unscaled, y_pred_scaled = logistic_regression_predict(df, X_test, pred_week)
        st.write("Logistic Regression - Unscaled\n")
        display_results(y_pred_unscaled, test_df)
        st.write("\nLogistic Regression - Scaled\n")
        display_results(y_pred_scaled, test_df)

    elif method == "Neural Network":
        # Run your neural network code and display the results
        y_pred = neural_network_predict(df, X_test, pred_week)
        st.write("Neural Network Predictions\n")
        display_results(y_pred, test_df)

    elif method == "Random Forest":
        # Run your random forest code and display the results
        y_pred = random_forest_predict(df, X_test, pred_week)
        st.write("Random Forest Predictions\n")
        display_results(y_pred, test_df)
