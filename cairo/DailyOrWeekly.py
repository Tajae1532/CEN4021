import streamlit
import datetime

#allows user to choose between daily and weekly schedule then returns a JSON response with the data
class ScheduleOptions:

    #requests the daily schedule. datetime.date.today() doesnt work for some reason???
    def daily(self):
        date = streamlit.date_input("Choose a date", datetime.date.today())


    #requests the weekly schedule
    def weekly(self):
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
        # return getSchedule(weekOption, "reg", yearOption)