import streamlit as st
import langchain_app as lp
import textwrap
st.title("DocsBot 🤖")

with st.sidebar:
    with st.form(key = "my form"):
        # taking directory path
        directory_path = st.text_area(label="Enter the path of directory format : C:/Users/ASUS/Desktop/sample_data/ ", max_chars=50)   
        # taking query from user                                    
        query = st.text_area(label="Enter the query related to documents", max_chars=100) 
        # sumbitting the form                                         
        submit_button = st.form_submit_button(label="submit")                                                                      

if directory_path and query:

    db = lp.file_to_vdb(directory_path,lp.embeddings)
    result = lp.main_runner(db,query)
    st.subheader("Answer : ")
    st.write(textwrap.fill(result))
