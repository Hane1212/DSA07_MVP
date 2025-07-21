import streamlit.components.v1 as components
import os
import requests
import streamlit as st

# gemini_model = "gemini-1.5-flash"
gemini_model = "gemini-2.5-pro"

def respond_to_chat(user_msg):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [
                    {
                        "parts": [{"text": user_msg}],
                    }
                ]
            },
        )
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        st.error(f"⚠️ Gemini API error: {e}")
        return "⚠️ Sorry, I couldn't reach the Gemini server."

def show_chatbox():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"sender": "bot", "text": "👩‍🌾 Hello! How can I help you with agriculture today?"}
        ]

    with st.expander("💬 Open AgriBot Chat"):
        st.markdown("#### AgriBot Chat")

        for msg in st.session_state.chat_history:
            if msg["sender"] == "user":
                st.markdown(f"<div style='text-align:right; color:#2e7d32'><b>You:</b> {msg['text']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:left; color:#37474f'><b>Bot:</b> {msg['text']}</div>", unsafe_allow_html=True)

        user_input = st.text_input("Ask something:", key="chat_input", label_visibility="collapsed")
        if st.button("Send"):
            if user_input.strip():
                st.session_state.chat_history.append({"sender": "user", "text": user_input})
                reply = respond_to_chat(user_input)
                st.session_state.chat_history.append({"sender": "bot", "text": reply})
                st.rerun()