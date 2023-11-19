import streamlit
from cairo.DailyOrWeekly import ScheduleOptions
from cairo.ExtractInfoFromResponse import extractAndDisplayInfo


def displaySchedule():
    # initializing scheduleOptions object
    scheduleOptions = ScheduleOptions()

    # selection for daily or weekly schedule
    dayOption = streamlit.toggle(
        "Daily Schedule"
    )

## Change to returning to object named response
    # daily schedule
    if dayOption:
        response = scheduleOptions.daily()
    # weekly schedule
    else:
        response = scheduleOptions.weekly()

    # extract info
    games_info = extractAndDisplayInfo(response)
