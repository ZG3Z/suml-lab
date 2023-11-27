import streamlit as st
import joblib

sex_d = {0: 'Kobieta', 1: 'Mężczyzna'}
pclass_d = {0: 'Pierwsza', 1: 'Druga', 2: 'Trzecia'}
embarked_d = {0: 'Cherbourg', 1: 'Queenstown', 2: 'Southampton'}

filename = "model"
model = model = joblib.load(filename)

def main():
    st.set_page_config(page_title='Czy przeżyłbyś katastrofę?')
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://media1.popsugar-assets.com/files/thumbor/7CwCuGAKxTrQ4wPyOBpKjSsd1JI/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2017/04/19/743/n/41542884/5429b59c8e78fbc4_MCDTITA_FE014_H_1_.JPG")
    

    with overview:
        st.title('Czy przeżyłbyś katastrofę?')

    with left:
        sex_ratio = st.radio('Płeć', list(sex_d.keys()), format_func= lambda x: sex_d[x])
        pclass_ratio = st.radio('Klasa', list(pclass_d.keys()), format_func = lambda x: pclass_d[x])
        embarked_ratio = st.radio('Port', list(embarked_d.keys()), index=2, format_func = lambda x: embarked_d[x])

    with right:
        age_slider = st.slider('Wiek', value=50, min_value=1, max_value=100)
        sibsp_slider = st.slider('# Liczba rodzeństwa i/lub partnera', min_value=0, max_value=8)
        parch_slider = st.slider('# Liczba rodziców i/lub dzieci', min_value=0, max_value=6)
        fare_slider = st.slider('Cena biletu', min_value=0, max_value=500, step=10)

    data = {
        'Pclass': pclass_ratio,
        'Age': age_slider,
        'SibSp': sibsp_slider,
        'Parch': parch_slider,
        'Fare': fare_slider,
        'Embarked': embarked_ratio,
        'male': sex_ratio,
    }

    data_for_prediction = [[data[key] for key in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'male']]]

    survival = model.predict(data_for_prediction)[0]
    s_confidence = model.predict_proba(data_for_prediction)[0]  

    with prediction:
        st.header('Czy dana osoba przeżyje?')
        if survival == 1:
            st.success('Tak')
        else:
            st.error('Nie')

        st.subheader("Pewność predykcji {0:.2f} %".format(s_confidence[survival] * 100))

if __name__ == '__main__':
    main()
