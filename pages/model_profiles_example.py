import json

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

description = """
以下のようなモデル名を入力してください。

- openai:gpt-5.1
- anthropic:claude-sonnet-4-5-20250929
"""


def app() -> None:
    load_dotenv(override=True)

    st.title("Model Profiles Example")

    st.info(description)

    model_name = st.text_input("Model Name")
    if model_name:
        model = init_chat_model(model=model_name)
        model_profile = model.profile
        st.write(model_profile)
        print(json.dumps(model_profile, indent=2))


app()
