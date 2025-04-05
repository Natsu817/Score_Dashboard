import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
import pandas as pd

# セッションステートの初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# OpenAI APIキーを取得（環境変数から）
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# データ読み込み
df = pd.read_csv('./Rag/Score_table.csv')
df = df.iloc[:, 0:11]

# ベクトルデータベースを読み込み
embedding = OpenAIEmbeddings(model='text-embedding-3-small')
db = FAISS.load_local(
    "./Rag/Score_table",
    embedding,
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever(search_kwargs={"k": 2})

# プロンプトテンプレート
prompt_form = ChatPromptTemplate.from_template(
    '''
    以下の文脈だけを踏まえて質問に回答してください。

    文脈: """
    {context}
    """

    質問:"""
    {question}
    """
    '''
)

# GPTモデルの設定（gpt-4o-mini）
model = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)

# Streamlit UI
st.title("🔍 わくわくさん チャットボット")
st.write("わくわくさんチャットボットへようこそ。きみは、どんなワクワクが欲しいかな。")

# 📊 データ表示
df['年月'] = df['年月'].astype(str)
df['類似のワクワクの発生年月'] = df['類似のワクワクの発生年月'].astype(str)

st.write("わくわくさんが答えられる情報はこれだけだよ")
st.dataframe(df)

# チャット履歴の表示
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ユーザー入力
if query := st.chat_input("いもむしカンパニー 2023年の情報..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # チェーン構築と応答生成
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_form
        | model
        | StrOutputParser()
    )
    response_content = chain.invoke(query)

    st.session_state.messages.append({"role": "assistant", "content": response_content})
    with st.chat_message("assistant"):
        st.markdown(response_content)
