import streamlit
from cairo.DisplayNameLogoSchedule import displayLogoSubheader
from cairo.convertGameDate import convertDate


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
            if game_info['gameID'] not in printedGameIDs:
                displayLogoSubheader(game_info['home'], game_info['away'])
                streamlit.write(f"Date: {convertDate(game_info['gameDate'])}")
                streamlit.write(f"Time: {game_info['gameTime']}m")
                streamlit.write(f"Status: {game_info['gameStatus']}")
                streamlit.write(f"Home Team: {game_info['home']}")
                streamlit.write(f"Away Team: {game_info['away']}")
                streamlit.write(f"ESPN Link: {game_info['espnLink']}")
                streamlit.write(f"CBS Link: {game_info['cbsLink']}")
                streamlit.write("-----")

                printedGameIDs.append(game_info['gameID'])

        printedGameIDs = []

