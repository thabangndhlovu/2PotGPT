def main():
    try:
        __import__("pysqlite3")
        import sys
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except:
        pass

    import re
    import time
    import streamlit as st
    from llm_model import initialize_conversation_chain

    disclaimer = """
    ### Disclaimer:

    2PotGPT is a Retrieval-Augmented Generation (RAG) AI model providing information 
    about South Africa's Two-Pot Retirement system. While it retrieves data from 
    curated sources, this information is for general purposes only and may not be 
    current or complete. It does not constitute financial advice. Always verify 
    information with official sources and consult a qualified financial advisor for 
    personalised guidance.

    **Users are responsible for decisions made based on this tool's output.**

    Written by: [Thabang Ndhlovu](https://www.linkedin.com/in/thabangndhlovu)😄

    """
    st.set_page_config(
        page_title="2PotGPT", page_icon=".streamlit/icon.jpg", layout="centered"
    )
    st.markdown(
        """<h1 style="text-align: center; font-size: 50px;"><strong>2Pot<span style='color: #5EC6F5;'>GPT</span></strong></h2>""",
        unsafe_allow_html=True,
    )

    def get_conversation_chain():
        if "conversation_chain" not in st.session_state:
            st.session_state.conversation_chain = initialize_conversation_chain()
        return st.session_state.conversation_chain

    def reset_conversation():
        st.session_state.messages = []
        st.session_state.conversation_chain = initialize_conversation_chain()
        st.session_state.conversation = None
        st.session_state.chat_history = None

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "What would you like to know about the Two-Pot System?",
            }
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the Two Pot System"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            full_response = ""

            response = get_conversation_chain()({"question": prompt})["answer"]
            chunks = re.findall(r"(\S+\s*|\n|$)", response)

            # Simulate streaming
            for chunk in chunks:
                full_response += chunk
                time.sleep(0.053)  # Add a small delay between chunks

                # Clean up the displayed text
                display_text = re.sub(r"\n+", "\n\n", full_response.strip())
                message_placeholder.markdown(display_text + "▌")

            message_placeholder.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        st.button(
            "Reset Chat",
            type="primary",
            use_container_width=True,
            on_click=reset_conversation,
        )

    st.sidebar.markdown(disclaimer)


if __name__ == "__main__":
    main()