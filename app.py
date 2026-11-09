import uuid
import pandas as pd
import streamlit as st
import data_analysis_agent_langchain_v2 as agent_mod

st.set_page_config(page_title="Data Analysis Agent", layout="wide")
st.title("Data Analysis Agent")


# --- Upload ---
uploaded = st.sidebar.file_uploader("Upload a CSV file", type="csv")
if uploaded is None:
    st.info("Upload a CSV file to start asking questions.")
    st.stop()

# New file -> new dataset and a fresh conversation (new thread_id)
if st.session_state.get("file_name") != uploaded.name:
    st.session_state.df = pd.read_csv(uploaded)
    st.session_state.file_name = uploaded.name
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat = []



agent_mod.set_data(st.session_state.df)   # every rerun, so tools always see it

df = st.session_state.df
st.sidebar.write(f"**{uploaded.name}**: {df.shape[0]} rows × {df.shape[1]} columns")
st.sidebar.dataframe(df.head())

# --- Chat history ---
for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        for p in turn.get("plots", []):
            st.image(p)

# --- New question ---
if prompt := st.chat_input("Ask a question about your data"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            answer, tool_log, plots = agent_mod.run_agent(prompt, st.session_state.thread_id)
        st.markdown(answer)
        for p in plots:
            st.image(p)
        with st.expander("Tool calls"):
            st.code("\n".join(tool_log) or "No tools called")

    st.session_state.chat.append({"role": "assistant", "content": answer, "plots": plots})    