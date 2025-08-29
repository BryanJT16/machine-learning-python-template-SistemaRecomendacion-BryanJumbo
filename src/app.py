import streamlit as st
import pandas as pd
from pickle import load

st.title("Ingresos futuros según la Data")

model = load(open("modelo_recomendacion.sav", "rb"))

age = st.slider("Edad", min_value=0, max_value=100, value=25)
education = st.selectbox(
    "Educación", 
    [
        "Preschool",
        "1st-4th",
        "5th-6th",
        "7th-8th",
        "9th",
        "10th",
        "11th",
        "12th",
        "HS-grad",
        "Some-college",
        "Assoc-voc",
        "Assoc-acdm",
        "Bachelors",
        "Masters",
        "Prof-school",
        "Doctorate"
    ]
)

capital_gain = st.number_input("Ganancia de Capital", min_value=0, max_value=100000, value=0)
capital_loss = st.number_input("Pérdida de Capital", min_value=0, max_value=100000, value=0)
hours_per_week = st.slider("Horas por Semana", min_value=0, max_value=168, value=40)

workclass = st.selectbox("Clase de Trabajo", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])   
marital_status = st.selectbox("Estado Civil", ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
occupation = st.selectbox("Ocupación", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
relationship = st.selectbox("Relación", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
race = st.selectbox("Raza", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
sex = st.selectbox("Sexo", ("Male", "Female"))
native_country = st.selectbox("País de Origen", ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"])


df_mapping = pd.read_csv("src/education_mapping.csv")
education_dict = dict(zip(df_mapping["education"], df_mapping["education-num"]))


education_num = education_dict[education]

row = [[age, education_num, capital_gain, capital_loss, hours_per_week, workclass, marital_status, occupation, relationship, race, sex, native_country]]


def recomendar_mejoras(profile, threshold_top=0.80, threshold_mid=0.60):
    df_profile = pd.DataFrame([profile])
    prob = model.predict_proba(df_profile)[0, 1]
    
    salida = ""
    salida += f"Probabilidad actual: {prob * 100:.2f}%\n"
    
    if prob >= threshold_top:
        salida += ("✔ Alta probabilidad: ya tienes un perfil favorable")
    elif prob >= threshold_mid:
        salida += ("- Probabilidad media: podrías mejorar en algunos aspectos")
    else:
        salida += ("✖ Baja probabilidad: necesitas mejoras para superar los 50K")
    
    salida += "\nRecomendaciones:\n"

    
    # Aumentar educación
    if profile["education-num"] < 16:
        diferencia = 16 - profile["education-num"]
        perfil_mejorado = profile.copy()
        perfil_mejorado["education-num"] += diferencia # Simula obtener un título universitario
        
        prob_mejorada = model.predict_proba(pd.DataFrame([perfil_mejorado]))[0, 1]
        salida += f"- Aumentar educación a nivel universitario: probabilidad = {prob_mejorada*100:.2f}%\n"
    
    # Aumentar horas trabajadas
    if profile["hours-per-week"] < 40:
        diferencia = 40 - profile["hours-per-week"]
        perfil_mejorado = profile.copy()
        perfil_mejorado["hours-per-week"] += diferencia # Simula trabajar 40 horas/semana

        prob_mejorada = model.predict_proba(pd.DataFrame([perfil_mejorado]))[0, 1]
        salida += f"- Trabajar hasta 40 horas semanales: probabilidad = {prob_mejorada*100:.2f}%\n"
    
    # Cambiar a ocupación ejecutiva 
    if profile["occupation"] != "Exec-managerial":
        ocupacion = profile["occupation"]
        perfil_mejorado = profile.copy()
        perfil_mejorado["occupation"] = "Exec-managerial"
        prob_mejorada = model.predict_proba(pd.DataFrame([perfil_mejorado]))[0, 1]
        salida += f"- Cambiar de {ocupacion} a ocupación ejecutiva: probabilidad = {prob_mejorada*100:.2f}%\n"
    
    st.text_area("Resultados y recomendaciones", value=salida, height=300)

st.divider()

if st.button("Predecir Ingresos"):
    recomendar_mejoras({
        "age": age,
        "education-num": education_num,
        "workclass": workclass,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country
    })