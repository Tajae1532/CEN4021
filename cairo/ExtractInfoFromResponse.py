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
            if game['gameID'] not in printedGameIDs:
                displayLogoSubheader(game['home'], game['away'])
                streamlit.write(f"Date: {convertDate(game['gameDate'])}")
                streamlit.write(f"Time: {game['gameTime']}m")
                streamlit.write(f"Status: {game['gameStatus']}")
                streamlit.write(f"Home Team: {game['home']}")
                streamlit.write(f"Away Team: {game['away']}")
                streamlit.write(f"ESPN Link: {game['espnLink']}")
                streamlit.write(f"CBS Link: {game['cbsLink']}")
                streamlit.write("-----")

                printedGameIDs.append(game['gameID'])

        printedGameIDs.clear()

