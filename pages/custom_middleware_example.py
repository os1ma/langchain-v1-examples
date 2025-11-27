from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain.messages import HumanMessage

from app.custom_middleware_example import CustomMiddlewareExampleAgent
from app.streamlit_components.show_message import show_message

load_dotenv()


class UIState:
    def __init__(self, user_id: str) -> None:
        self.agent = CustomMiddlewareExampleAgent()
        self.new_thread()
        self.user_id = user_id

    def new_thread(self) -> None:
        self.thread_id = uuid4().hex


def app() -> None:
    load_dotenv(override=True)

    # UIStateを初期化
    if "custom_middleware_example_ui_state" not in st.session_state:
        st.session_state.custom_middleware_example_ui_state = UIState(user_id="123")
    ui_state: UIState = st.session_state.custom_middleware_example_ui_state

    with st.sidebar:
        # 新規スレッドボタン
        clicked = st.button("新規スレッド")
        if clicked:
            ui_state.new_thread()
            st.rerun()

    st.title("Custom Middleware Example")
    st.write(f"thread_id: {ui_state.thread_id}")

    # 会話履歴を表示
    for m in ui_state.agent.get_messages(ui_state.thread_id):
        show_message(m)

    # ユーザーの指示を受け付ける
    human_input = st.chat_input()
    if human_input:
        show_message(HumanMessage(content=human_input))

        with st.spinner():
            for chunk in ui_state.agent.stream(
                message=human_input,
                user_id=ui_state.user_id,
                thread_id=ui_state.thread_id,
            ):
                show_message(chunk)


app()
