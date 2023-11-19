class Date:
    month = 0
    day = 0
    year = 0

def convertDate(date):
    gameDate = Date()
    gameDate.year = date[0:4]
    gameDate.month = date[4:6]
    gameDate.day = date[6:]

    return f"{gameDate.month}/{gameDate.day}/{gameDate.year}"