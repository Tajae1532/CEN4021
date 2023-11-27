import streamlit
from cairo.TeamLogo import teamLogo

class Date:
    month = 0
    day = 0
    year = 0

#converts gameDate from JSON response into MM/DD/YYYY
def convertDate(date):
    gameDate = Date()
    gameDate.year = date[0:4]
    gameDate.month = date[4:6]
    gameDate.day = date[6:]

    return f"{gameDate.month}/{gameDate.day}/{gameDate.year}"

# extract and display info from response
def extractAndDisplayInfo(response):

    # extract info from response
    games_info = []

    if response['body'] is None or response['body'] == []:
        streamlit.subheader("No Games Scheduled.")
    else:
        for game in response['body']:
            game_info = {
                'gameID': game['gameID'],
                'seasonType': game['seasonType'],
                'away': game['away'],
                'gameDate': game['gameDate'],
                'espnID': game['espnID'],
                'teamIDHome': game['teamIDHome'],
                'gameStatus': game['gameStatus'],
                'gameWeek': game['gameWeek'],
                'teamIDAway': game['teamIDAway'],
                'home': game['home'],
                'espnLink': game['espnLink'],
                'cbsLink': game['cbsLink'],
                'gameTime': game['gameTime'],
                'season': game['season'],
                'neutralSite': game['neutralSite']
            }

            games_info.append(game_info)

        printedGameIDs = []

        for game in games_info:
            if game['gameID'] not in printedGameIDs:

                # Get team logos
                home_logo = teamLogo(game['home'])
                away_logo = teamLogo(game['away'])

                with streamlit.container():
                    with open('styles.css') as f:
                        streamlit.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

                    # Display in multiple columns
                    col1, col2, col3 = streamlit.columns([1, 4, 1])

                    # Logo and information in columns
                    with col1:
                        streamlit.image(away_logo, use_column_width="auto")

                    with col2:
                        streamlit.header(f"{game['away']} @ {game['home']}")
                        streamlit.write(f"Date: {convertDate(game['gameDate'])}")
                        streamlit.write(f"Time: {game['gameTime']}m")
                        streamlit.write(f"Status: {game['gameStatus']}")
                        streamlit.write(f"Home Team: {game['home']}")
                        streamlit.write(f"Away Team: {game['away']}")
                        streamlit.write(f"ESPN Link: {game['espnLink']}")
                        streamlit.write(f"CBS Link: {game['cbsLink']}")

                    with col3:
                        streamlit.image(home_logo, use_column_width="auto")

                    streamlit.write("-----")

                    printedGameIDs.append(game['gameID'])

        printedGameIDs.clear()
