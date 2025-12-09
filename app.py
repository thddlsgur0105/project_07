import streamlit as st
from langchain_core.messages import ChatMessage

from langchain_ollama import ChatOllama      # 최신 import 방식
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate


class ChatWeb:
    def __init__(self, llm, page_title='Gazzi Chatbot', page_icon=':books:'):
        self.page_title = page_title
        self.page_icon = page_icon
        self.llm = llm
        
    def print_message(self):
        if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
            for chat_message in st.session_state["messages"]:
                st.chat_message(chat_message.role).write(chat_message.content)

    def run(self):
        st.set_page_config(
            page_title=self.page_title,
            page_icon=self.page_icon,
        )
        st.title(self.page_title)
        
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        
        self.print_message()
        
        if user_input := st.chat_input("질문을 입력해주세요."):
            st.chat_message("user").write(user_input)
            st.session_state["messages"].append(ChatMessage(role="user", content=user_input))

            # LLM 호출
            response = self.llm.invoke(user_input)

            with st.chat_message("assistant"):
                st.write(response)
                st.session_state["messages"].append(ChatMessage(role="assistant", content=response))


class ChatLLM:
    def __init__(self):
        # 최신 Ollama 모델 호출
        self.model = ChatOllama(model="gemma2:2b", temperature=3)
        
        self._template = """주어진 질문에 짧고 간결하게 한글로 답변을 제공해주세요.

        Question: {question}
        """

        # 최신 ChatPromptTemplate
        self._prompt = ChatPromptTemplate.from_template(self._template)

        # LCEL 체인 구성
        self._chain = (
            {"question": RunnablePassthrough()}
            | self._prompt
            | self.model
            | StrOutputParser()
        )

    def invoke(self, user_input: str):
        return self._chain.invoke(user_input)


if __name__ == "__main__":
    llm = ChatLLM()
    web = ChatWeb(llm=llm)
    web.run()
