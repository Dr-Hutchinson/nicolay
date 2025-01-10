import streamlit as st

st.set_page_config(
    page_title="Nicolay: Exploring the Speeches of Abraham Lincoln with AI (version 0.2)",
    layout='wide',
    page_icon='🎩'
)

st.header("Sources & Acknowledgements:")

st.subheader("**Data Sources:**")

st.write("The project relies on the speeches of Abraham Lincoln produced and curated by the [University of Virginia's Miller Center](https://millercenter.org/). Their [corpus of presidential speeches](https://millercenter.org/presidential-speeches-downloadable-data) is an important primary source for studying American history.")
st.write("Data used for linguistic analysis of the Lincoln corpus was created using [Voyant.](https://voyant-tools.org/)")

st.write("Additional data sources used by the app can be found in the [project Github page.](https://github.com/Dr-Hutchinson/nicolay)")


st.subheader("**Who was Nicolay?**")

st.write("[John George Nicolay](https://en.wikipedia.org/wiki/John_George_Nicolay) was a German-born American author and diplomat who served as the private secretary to US President Abraham Lincoln. Nicolay collaborated with his fellow Lincoln secretary, John Hay, to compose a 10-volume biography of Lincoln's life. Their substantial research played an important role in the future scholarship of the life and times of Lincoln. This app has been named Nicolay to honor his role in documenting and preserving the words and legacy of Abraham Lincoln for future generations.")

st.subheader("**Acknowledgements:**")

st.write("Thanks to the Miller Center at the University of Virginia for preserving and maintaining the collection of presidential speeches, which serve as the data source for this project, and to [Dr. Abraham Gibson](http://history.utsa.edu/faculty/gibson-abe) at the University of Texas-San Antonio for the opportunity to contribute to [Honest Abe's Information Emporium](https://honestabes.info/fireside-chats/).")
