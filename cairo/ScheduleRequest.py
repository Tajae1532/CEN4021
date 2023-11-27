import requests

global APIkey
APIkey = "41d4fe3980msh72a2378007665cap10ef7bjsn531723c27c11"

#sending a request to the API for the schedule of the chosen week
def getWeeklySchedule(week, sType, sYear):
    url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLGamesForWeek"

    querystring = {"week": week, "seasonType": sType, "season": sYear}

    headers = {
        "X-RapidAPI-Key": APIkey,
        "X-RapidAPI-Host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    return response.json()


#sending a request to the API for the schedule of the chosen date
def getDailySchedule(gameDate):
    url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLGamesForDate"

    querystring = {"gameDate": {gameDate}}

    headers = {
        "X-RapidAPI-Key": APIkey,
        "X-RapidAPI-Host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    return response.json()

