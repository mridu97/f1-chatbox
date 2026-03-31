import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
 
import os
load_dotenv()
if "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
 
st.set_page_config(
    page_title="F1 Chatbox",
    page_icon="🏎️",
    layout="centered"
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500&display=swap');
 
html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif !important;
}
 
/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
 
.block-container {
    padding-top: 2rem;
    padding-bottom: 5rem;
    max-width: 780px;
}
 
/* ── HEADER ── */
.f1-header {
    text-align: center;
    padding: 1.5rem 1rem 0.5rem;
}
.f1-redbar {
    width: 48px;
    height: 4px;
    background: #e10600;
    margin: 0 auto 1.2rem;
    border-radius: 2px;
}
.f1-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #0d0d0d;
    line-height: 1;
    margin: 0;
}
.f1-title span { color: #e10600; }
.f1-subtitle {
    font-size: 0.72rem;
    font-weight: 400;
    color: #999;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.f1-divider {
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, transparent, #ddd, transparent);
    margin: 1.2rem 0;
}
 
/* ── STATUS BAR ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.68rem;
    color: #bbb;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    justify-content: center;
    margin-bottom: 0.3rem;
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #22c55e;
    display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; }
    50%      { opacity:0.3; }
}
 
/* ── CHIPS LABEL ── */
.chips-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #bbb;
    margin-bottom: 0.4rem;
}
 
/* ── SUGGESTION CHIP BUTTONS ── */
.stButton > button {
    background: #ffffff !important;
    border: 1px solid #e8e8e8 !important;
    color: #444 !important;
    border-radius: 20px !important;
    font-family: 'Barlow', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    padding: 0.28rem 0.85rem !important;
    transition: all 0.18s ease !important;
    white-space: nowrap !important;
    width: 100% !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
.stButton > button:hover {
    border-color: #e10600 !important;
    color: #e10600 !important;
    background: #fff5f5 !important;
    box-shadow: 0 2px 8px rgba(225,6,0,0.1) !important;
}
 
/* ── CHAT MESSAGES ── */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: 0.15rem 0 !important;
}
[data-testid="stChatMessageContent"] {
    background: transparent !important;
}
 
/* user bubble */
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: #f0f0f0 !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 14px 14px 2px 14px !important;
    padding: 0.7rem 1rem !important;
    color: #1a1a1a !important;
}
 
/* assistant bubble */
.stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
    background: #ffffff !important;
    border: 1px solid #ebebeb !important;
    border-left: 3px solid #e10600 !important;
    border-radius: 2px 14px 14px 14px !important;
    padding: 0.7rem 1rem !important;
    color: #1a1a1a !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}
 
/* avatar size */
.stChatMessage .stAvatar {
    width: 26px !important;
    height: 26px !important;
    min-width: 26px !important;
}
 
/* ── CHAT INPUT ── */
[data-testid="stChatInputSubmitButton"] > button,
[data-testid="stChatInputSubmitButton"] {
    background: #e10600 !important;
    border-radius: 6px !important;
    border: none !important;
}
 
/* ── SPINNER ── */
.stSpinner > div { border-top-color: #e10600 !important; }
 
/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #f5f5f5; }
::-webkit-scrollbar-thumb { background: #ddd; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #e10600; }
 
/* ── ACCENT CARD for welcome message ── */
.welcome-card {
    background: #fff;
    border: 1px solid #eee;
    border-left: 4px solid #e10600;
    border-radius: 2px 12px 12px 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
    color: #333;
    font-size: 0.92rem;
    line-height: 1.6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)
 
 
# ── HEADER ──
st.markdown("""
<div class="f1-header">
    <div class="f1-redbar"></div>
    <h1 class="f1-title">F1 <span>Chatbox</span></h1>
    <p class="f1-subtitle">Formula 1 Intelligence Agent</p>
</div>
<div class="status-bar">
    <span class="status-dot"></span>
    <span>Agent Online &nbsp;·&nbsp; Powered by Claude &nbsp;·&nbsp; FIA Regulations Loaded</span>
</div>
<div class="f1-divider"></div>
""", unsafe_allow_html=True)
 
 
# ── LOAD CHAIN ──
@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
 
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0
    )
 
    prompt = ChatPromptTemplate.from_template("""
You are a F1 chatbox, a passionate and knowledgeable Formula 1 expert assistant.
Your goal is to help fans especially new ones understand and love the sport.
You explain regulations clearly, share trivia enthusiastically, and make F1 accessible to everyone.
Use the context below from official F1 documents to answer accurately.
If the context does not cover the question, say so honestly and share what you do know.
Keep answers engaging, clear, and friendly. Use bullet points for complex answers.
 
Context:
{context}
 
Question: {question}
 
Answer:
""")
 
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
 
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
 
 
chain = load_chain()
 
 
# ── SESSION STATE ──
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Welcome to F1 Chatbox 🏎️\n\nI'm your Formula 1 assistant. Ask me anything — regulations, trivia, driver history, technical rules, race procedures, or anything else about the sport. Whether you're a new fan or a seasoned one, I'm here to help!"
    })
 
if "chip_question" not in st.session_state:
    st.session_state.chip_question = None
 
 
# ── SUGGESTION CHIPS ──
st.markdown('<p class="chips-label">Try asking</p>', unsafe_allow_html=True)
 
chips = [
     "🏁 Points scoring?",
    "🔵 What is DRS?",
    "⚠️ Safety Car rules?",
    "🔧 Tyre compounds?",
    "🏆 Most championships?",
    "📋 What is parc fermé?",
]
 
cols = st.columns(3)
for i, chip in enumerate(chips):
    with cols[i % 3]:
        if st.button(chip, key=f"chip_{i}"):
            st.session_state.chip_question = chip
 
st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)
 
 
# ── CHAT HISTORY ──
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
 
# ── HANDLE CHIP CLICK ──
if st.session_state.chip_question:
    question = st.session_state.chip_question
    st.session_state.chip_question = None
 
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
 
    with st.chat_message("assistant"):
        with st.spinner("On it..."):
            response = chain.invoke(question)
            st.markdown(response)
 
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
 
 
# ── CHAT INPUT ──
if user_input := st.chat_input("Ask anything about Formula 1..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
 
    with st.chat_message("assistant"):
        with st.spinner("On it..."):
            response = chain.invoke(user_input)
            st.markdown(response)
 
    st.session_state.messages.append({"role": "assistant", "content": response})
