import streamlit as st
from streamlit.runtime.state import session_state
import utils

st.set_page_config(page_title="Chatbot básico", page_icon="🤖", layout="wide")
st.title("ChatBot Básico 166")

# Historial
if "history" not in st.session_state:
    st.session_state.history = []
# Contexto
if "Context" not in st.session_state:
    st.session_state.context = []

# Construimos el espacio, emisor-mensaje
for sender, msg in st.session_state.history:
    if sender == "Tú":
        st.markdown(f'**👨{sender}: ** {msg}')
    else:
         st.markdown(f'**🤖{sender}: ** {msg}')

# Si no hay entrada 

if "user_input" not in st.session_state:
    st.session_state.user_input= ""

# procesamiento de la entrada
def send_msg():
    user_input = st.session_state.user_input.strip()
    if user_input:
        tag = utils.predict_class(user_input)
        st.session_state.context.append(tag)
        responses = utils.get_response(tag, st.session_state.context)
        st.session_state.history.append(('Tú', user_input))
        st.session_state.history.append(('bot', responses))
        st.session_state.user_input = ''

# creamos el campo de texto
st.text_input("Escribe tu mensaje: ", key= "user_input", on_change=send_msg)

