import requests

def getSchedule(week, sType, sYear):
    url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLGamesForWeek"

    querystring = {"week": week, "seasonType": sType, "season": sYear}

    headers = {
        "X-RapidAPI-Key": "31a0367006msh826038b6be6e9c0p10628fjsna8d24c75286b",
        "X-RapidAPI-Host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    print(response)
    return response.json()

