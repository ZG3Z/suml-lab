import streamlit as st
from transformers import pipeline

st.image("NLP.jpeg", use_column_width=True)
st.header('Przetwarzanie języka naturalnego')

option = st.selectbox(
    "Opcje",
    [
        "Translator: Angielski ➡️ Niemiecki",
        "Wydźwięk emocjonalny tekstu (eng)",
    ],
    
)

if option == "Translator: Angielski ➡️ Niemiecki":
    st.write("""
    Wpisz tekst w pole poniżej, a następnie naciśnij przycisk 'Translate'.
    """)
    text = st.text_area('Wpisz tekst do przetłumaczenia')
    
    if st.button('Translate'):
        if text:
            with st.spinner('Tłumaczenie...'):
                translator = pipeline("translation_en_to_de")
                translated_text = translator(text, max_length=40)
                st.success("Tłumaczenie: " + translated_text[0]['translation_text'])
        else:
            st.warning("Wprowadź tekst do przetłumaczenia.")

if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        classifier = pipeline("sentiment-analysis")
        answer = classifier(text)
        st.write(answer)


st.subheader("Numer indeksu: s23570")
