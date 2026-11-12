import streamlit as st
from sentence_transformers import SentenceTransformer, util
import os
from groq import Groq

st.set_page_config(page_title="المساعد الذكي للمحل", page_icon="🤖")
st.title("🤖 المساعد الذكي للمحل")
st.write("مرحباً! اسألني عن الأسعار، الضمان، أو أي شيء يخص المحل.")

qa_data = [
    {"q": "الضمان كم؟", "a": "الضمان سنة كاملة ضد عيوب الصناعة."},
    {"q": "بتقبلوا أقساط؟", "a": "نعم، لدينا تقسيط عبر شركات محددة، يرجى المرور للمحل للتفاصيل."},
    {"q": "في توصيل لحمص؟", "a": "نعم، التوصيل متاح داخل دمشق فقط، ورسومه 20 ألف ليرة."},
    {"q": "الجهاز جديد ولا مستعمل؟", "a": "جميع الأجهزة المعروضة جديدة ومغلقة بضمان الوكيل."},
    {"q": "مرحبا، كيفك، أهلاً", "a": "أهلاً وسهلاً بك! كيف يمكنني مساعدتك اليوم؟"}
]

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
questions = [item["q"] for item in qa_data]
question_embeddings = model.encode(questions)

def get_bot_response(message):
    msg_embedding = model.encode(message)
    cos_scores = util.cos_sim(msg_embedding, question_embeddings)[0]
    best_idx = cos_scores.argmax()
    best_score = cos_scores[best_idx].item()

    # تم رفع الحد إلى 0.6 لمنع الإجابات العشوائية
    if best_score > 0.6:
        retrieved_answer = qa_data[best_idx]["a"]
        try:
            api_key = st.secrets["GROQ_API_KEY"]
            client = Groq(api_key=api_key)
            prompt = f"أنت مساعد ودود لمحل جوالات. أجب على سؤال الزبون بأسلوب طبيعي ومختصر بناءً على هذه المعلومة فقط: {retrieved_answer}\n\nإذا كانت المعلومة لا تجيب على السؤال، قل: 'عذراً، لا أملك هذه المعلومة.'\n\nسؤال الزبون: {message}"
            
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
            )
            return chat_completion.choices[0].message.content
        except Exception:
            # إذا فشل الاتصال، يعرض الجواب المباشر
            return retrieved_answer
    else:
        return "عذراً، لا أملك معلومات كافية عن هذا السؤال. يرجى التواصل معنا عبر واتساب للتفاصيل."

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("اكتب سؤالك هنا..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = get_bot_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
