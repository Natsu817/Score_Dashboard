
import os
import streamlit as st
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd

# セッションステートの初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# OpenAIクライアントを生成
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # 環境変数に設定したAPIキーを取得
client = OpenAI(api_key=OPENAI_API_KEY) # APIキーの指定


# dfを読み込む
df = pd.read_csv('./Rag/Score_table.csv')
df = df.iloc[:,0:11]

# dbを読み込む
embedding = OpenAIEmbeddings(model='text-embedding-3-small')
db = FAISS.load_local(
    "./Rag/Score_table",
    embedding,
    allow_dangerous_deserialization=True
)
# retrieverを作成
retriever = db.as_retriever(search_kwargs={"k": 1})

# プロンプトのひな型を作成
prompt_form = ChatPromptTemplate.from_template(
    '''
    以下の文脈だけを踏まえて質問に回答してください。

    文脈: """
    {context}
    """

    質問:"""
    {question}
    """
    
''')

# 生成AIモデルを定義
model = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)


# StreamlitチャットUIの表示
st.title("🔍 わくわくさん チャットボット")
st.write("わくわくさんチャットボットへようこそ。きみは、どんなワクワクが欲しいかな。")


# 📊 データフレームの表示をここで追加
df['年月'] = df['年月'].astype(str)
df['類似のワクワクの発生年月'] = df['類似のワクワクの発生年月'].astype(str)

st.write("わくわくさんが答えられる情報はこれだけだよ")
st.dataframe(df)

# チャット履歴を表示
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ユーザーの入力受付
if query := st.chat_input("いもむしカンパニー 2023年の情報..."):
    # ユーザーの入力を履歴に追加
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # RAGのチェーン処理を構築
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_form
        | model
        | StrOutputParser()
    )

    # 回答生成
    response_content = chain.invoke(query)

    # 回答を履歴に追加し表示
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    with st.chat_message("assistant"):
        st.markdown(response_content)


