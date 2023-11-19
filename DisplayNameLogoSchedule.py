import streamlit
from NFLWebApp import teamLogo

#display team names and logos for schedule
def displayLogoSubheader(home, away):
    # made extra columns to help with sizing logo images
    col1, col2, col3, col4, col5 = streamlit.columns(5, gap='small')

    # Display team logos in each column
    with col1:
        streamlit.image(teamLogo(away), use_column_width="auto")

    with col2:
        streamlit.image(teamLogo(home), use_column_width="auto")

    # formatting team names under logo images
    formatted_text = "&nbsp;&nbsp;" \
                     f"{home}" \
                     f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" \
                     "@" \
                     "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" \
                     f"{away}"

    # display formatted team names
    streamlit.subheader(formatted_text)



