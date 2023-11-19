import streamlit
import datetime
from cairo.ScheduleRequest import getDailySchedule, getWeeklySchedule


#allows user to choose between daily and weekly schedule then returns a JSON response with the data
class ScheduleOptions:

    # requests the daily schedule
    def daily(self):
        # date selection and formatting
        date = streamlit.date_input(label="Choose a date")
        gameDate = f"{date.year}{date.month}{date.day}"

        # sending request to rapid api for schedule
        response = getDailySchedule(gameDate)

        return response


    #requests the weekly schedule
    def weekly(self):
        # week selection
        weekOption = streamlit.selectbox(
            "Select a week you would like to see",
            ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'),
            index=None,
            placeholder="Select week...",
        )

        # year selection. placeholder doesn't work for some reason?
        yearOption = streamlit.number_input("Insert a year:",
                                            step=1,
                                            min_value=1989,
                                            max_value=datetime.date.today().year,
                                            placeholder=datetime.date.today().year.__str__())

        streamlit.write('You selected week:', weekOption, 'of the ', yearOption, " season.")

        # sending request to rapid api for schedule
        return getWeeklySchedule(weekOption, "reg", yearOption)