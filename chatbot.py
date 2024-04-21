#
# Copyright IBM Corp. 2024
# SPDX-License-Identifier: Apache-2.0
#

import streamlit as st
import utils
import time
import os

st.set_page_config(
    page_title="GitPal",
    page_icon="images/gitpal.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.image("images/gitpal.png", width=200)


def main():
    if "genai_api_key" not in st.session_state:
        genai_api_key_placeholder = st.empty()
        genai_api_key = genai_api_key_placeholder.text_input(
            "IBM WatsonX API Key", type="password"
        )
        if not genai_api_key:
            st.info("Please add your IBM WatsonX API key to continue.")
            st.stop()
        genai_api_key_placeholder.empty()
        st.session_state.genai_api_key = genai_api_key

    with st.sidebar:
        st.image("images/gitpal.png", width=300)
        st.title(""":orange[Welcome to GitPal!]""")

        st.markdown(
            "ISC-CodeConnect is an advanced platform built using IBM WatsonX. "
            "It answers questions related to ISC Github Repositories."
            "Choose ISC repositories, and let CodeConnect hatch profound insights from repositories, all powered by cutting-edge IBM WatsonX Gen AI"
        )

        if "user_repo" not in st.session_state:
            user_repo = st.text_input(
                "Github Link to public repository",
                "",
            )
            if not user_repo:
                st.info("Please add Github repository link to continue.")
                st.stop()
            st.session_state.user_repo = user_repo

        if "embedder" not in st.session_state:
            st.toast("Cloning your repository...")
            time.sleep(2)

            ## Load the Github Repo
            embedder = utils.Embedder(st.session_state.user_repo)
            embedder.clone_repo()
            st.session_state.embedder = embedder
            st.toast("Your repo has been cloned")
            time.sleep(2)

            ## Chunk and Create DB
            with st.spinner("Processing your repository. This may take some time.."):
                st.session_state.conversation_chain = (
                    st.session_state.embedder.get_conversation_chain(
                        gen_ai_key=st.session_state.genai_api_key
                    )
                )
                st.success("Processing completed. Ready to take your questions")

    st.markdown(""":orange[Selected Repository] - """ + st.session_state.user_repo)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello. I am GitPal - Your personalized Github assistant! How may I assist you today?",
            }
        ]
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def clear_chat_history():
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello. I am GitPal - Your personalized Github assistant! How may I assist you today?",
            }
        ]

    st.sidebar.button("Clear Chat History", on_click=clear_chat_history, type="primary")

    if prompt := st.chat_input("Ask question from this repository."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking...."):
            # Display assistant response in chat message container
            response = st.session_state.embedder.retrieve_results(
                prompt, st.session_state.conversation_chain
            )
            # Add assistant response to chat history
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)

        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    main()
