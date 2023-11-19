import datetime

from ScheduleRequest import getSchedule
import streamlit
from convert import convertDate

def displaySchedule():
    # week selection
    weekOption = streamlit.selectbox(
        "Select a week you would like to see",
        ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'),
        index=None,
        placeholder="Select week...",
    )

    # year selection
    yearOption = streamlit.number_input("Insert a year:", step=1, min_value=1989, max_value=datetime.MAXYEAR)

    streamlit.write('You selected week:', weekOption, 'of the ', yearOption, " season.")

    # sending request to rapid api for schedule
    response = getSchedule(weekOption, "reg", yearOption)

    # extract info
    games_info = []

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

    for game in games_info:
        streamlit.subheader(game_info['gameID'])
        streamlit.write(f"Date: {convertDate(game_info['gameDate'])}")
        streamlit.write(f"Time: {game_info['gameTime']}m")
        streamlit.write(f"Status: {game_info['gameStatus']}")
        streamlit.write(f"Home Team: {game_info['home']}")
        streamlit.write(f"Away Team: {game_info['away']}")
        streamlit.write(f"ESPN Link: {game_info['espnLink']}")
        streamlit.write(f"CBS Link: {game_info['cbsLink']}")
        streamlit.write("-----")
