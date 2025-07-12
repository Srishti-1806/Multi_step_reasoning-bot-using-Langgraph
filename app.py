
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import asyncio
from rag_graph import graph  # Import your compiled LangGraph

st.set_page_config(page_title="Medical RAG Assistant", layout="wide")
st.title("🩺 Medical RAG Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask a medical question...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Display chat history so far
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    async def run_graph():
        input_data = {
            "question": HumanMessage(content=user_input),
            "messages": st.session_state.chat_history[:-1],  # previous messages
        }
        result = await graph.ainvoke(input=input_data, config={"configurable": {"thread_id": 1}})
        return result

    with st.spinner("Generating response..."):
        result = asyncio.run(run_graph())

    if "messages" in result:
        ai_msg = result["messages"][-1]
        if isinstance(ai_msg, AIMessage):
            st.session_state.chat_history.append(ai_msg)
            with st.chat_message("assistant"):
                st.markdown(ai_msg.content)
