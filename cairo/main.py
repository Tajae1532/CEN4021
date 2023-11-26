import streamlit
from cairo.DailyOrWeekly import ScheduleOptions
from cairo.ExtractInfoFromResponse import extractAndDisplayInfo


def displaySchedule():
    # initializing scheduleOptions object
    scheduleOptions = ScheduleOptions()

    # selection for daily or weekly schedule
    dayOption = streamlit.selectbox(
        "Choose a timeframe",
        ("Weekly Schedule", "Daily Schedule")
    )

## Change to returning to object named response
    # daily schedule
    if dayOption == "Daily Schedule":
        response = scheduleOptions.daily()
    # weekly schedule
    elif dayOption == "Weekly Schedule":
        response = scheduleOptions.weekly()

    # extract and display info
    extractAndDisplayInfo(response)
