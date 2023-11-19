import requests

#sending a request to the API for the schedule of the chosen week
def getSchedule(week, sType, sYear):
    #API url
    url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLGamesForWeek"

    querystring = {"week": week, "seasonType": sType, "season": sYear}

    headers = {
        "X-RapidAPI-Key": "PUT SOMETHING HERE LATER DONT FORGET",
        "X-RapidAPI-Host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    print(response)
    return response.json()

