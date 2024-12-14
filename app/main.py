import streamlit as st

from app.rag import GRAPH

config = {"configurable": {"thread_id": "abc123"}}

st.title("Abteilung6 AI")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Wie kann ich dir heute helfen?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Warum kann mein Vermieter die Kaution behalten?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    response = GRAPH.invoke(
        {"messages": st.session_state.messages[1:]},
        config={"configurable": {"thread_id": 42}},
    )
    response_message = response["messages"][-1].content
    with st.chat_message("assistant"):
        st.markdown(response_message)
    st.session_state.messages.append({"role": "assistant", "content": response_message})