# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib, os, io, base64, unicodedata, re
from pathlib import Path

# ==================== CONFIGURACIÓN GENERAL ====================
st.set_page_config(page_title="Gestantes - Puno", layout="wide", page_icon="🤰")

# ---------- FUNCIONES DE APOYO ----------
def set_background(image_file):
    """Agrega imagen de fondo"""
    if not Path(image_file).exists():
        return
    try:
        bin_str = Path(image_file).read_bytes()
        base64_img = base64.b64encode(bin_str).decode()
        css = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpg;base64,{base64_img}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center center;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"], [data-testid="stToolbar"] {{ background: rgba(0,0,0,0); }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        pass

def normalize_colname(s: str) -> str:
    """Limpia y normaliza nombres de columnas"""
    s = str(s).strip().lstrip('\ufeff').lstrip('\ufffe')
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    return s.strip('_')

# ---------- ESTILO GLOBAL ----------
st.markdown("""
<style>
h1, h2, h3, h4 { color: white !important; font-size: 30px !important; font-weight: bold; }
.section-title { color: #fff; font-size: 26px; margin-top: 20px; background-color: rgba(90,24,154,0.5); padding: 10px; border-radius: 10px; text-align:center; }
.stButton>button { background-color:#7b2cbf; color:white; border-radius:10px; height:3em; font-size:16px; }
.stButton>button:hover { background-color:#5a189a; color:#fff; }
div[data-testid="stMetricValue"]{ font-size:22px !important; color:#fff; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ---------- MENÚ ----------
st.sidebar.title("📚 Navegación")
menu = st.sidebar.radio("Selecciona una opción:", [
    "🏠 Inicio", "📈 Análisis de Datos", "⚙️ Entrenamiento", "🔮 Predicción", "📊 Estadísticas por Provincia"
])

# ---------- INICIO ----------
if menu == "🏠 Inicio":
    set_background("fondo_inicio.jpg")
    st.markdown("<h1 style='text-align:center;'>GESTANTES PRIMER SEMESTRE - PUNO</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:white; font-size:20px; background:rgba(0,0,0,0.5); padding:20px; border-radius:15px;">
    Sistema inteligente para el <b>análisis y predicción de anemia en gestantes</b> utilizando <b>Regresión Logística</b>.
    <br><br>Desarrollado por <b>Kely Zulema Ponce Quispe 💜</b>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ---------- CARGA DE ARCHIVO ----------
uploaded_file = st.sidebar.file_uploader("📤 Cargar archivo CSV", type=["csv"])
if uploaded_file is None:
    st.info("👆 Sube tu archivo CSV para comenzar el análisis.")
    st.stop()

# ---------- LECTURA ROBUSTA ----------
content = uploaded_file.read()
df = None
for sep in [",", ";", "\t", "|"]:
    try:
        df_try = pd.read_csv(io.BytesIO(content), sep=sep, encoding='utf-8')
        if df_try.shape[1] > 1:
            df = df_try
            break
    except Exception:
        continue

if df is None:
    st.error("⚠️ No se pudo leer el archivo CSV. Verifica el formato o separador.")
    st.stop()

df.columns = [normalize_colname(c) for c in df.columns]
mapping = {"edad":"Edad", "edad_gest":"Edad_Gest", "provincia":"Provincia",
           "hemoglobina":"Hemoglobina", "dx_anemia":"Dx_Anemia"}
df = df.rename(columns={c: mapping[c] for c in mapping if c in df.columns})

required = ["Edad", "Edad_Gest", "Provincia", "Hemoglobina", "Dx_Anemia"]
missing = [r for r in required if r not in df.columns]
if missing:
    st.error(f"❌ Faltan columnas requeridas: {missing}")
    st.stop()

df["Provincia"] = df["Provincia"].fillna("Desconocido").astype(str)
for col in ["Edad", "Edad_Gest", "Hemoglobina"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------- ANÁLISIS DE DATOS ----------
if menu == "📈 Análisis de Datos":
    set_background("analisis.jpg")
    st.markdown('<div class="section-title">📊 ANÁLISIS EXPLORATORIO DE DATOS</div>', unsafe_allow_html=True)
    st.dataframe(df.head())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de Gestantes", len(df))
    c2.metric("Provincias", df["Provincia"].nunique())
    c3.metric("Promedio de Hemoglobina", f"{df['Hemoglobina'].mean():.2f}")

    # Gráfico: Gestantes por provincia
    prov_counts = df["Provincia"].value_counts().reset_index()
    prov_counts.columns = ["Provincia", "Gestantes"]
    st.plotly_chart(px.bar(prov_counts, x="Provincia", y="Gestantes", color="Provincia",
                           title="Distribución de Gestantes por Provincia"), use_container_width=True)

    # Gráfico: Tipos de anemia
    anemia_counts = df["Dx_Anemia"].value_counts().reset_index()
    anemia_counts.columns = ["Diagnóstico", "Frecuencia"]
    st.plotly_chart(px.pie(anemia_counts, names="Diagnóstico", values="Frecuencia",
                           title="Distribución de Tipos de Anemia"), use_container_width=True)

    # Gráfico: Hemoglobina por provincia
    st.plotly_chart(px.box(df, x="Provincia", y="Hemoglobina", color="Provincia",
                           title="Distribución de Hemoglobina por Provincia"), use_container_width=True)

    # Gráfico: Anemia por provincia
    anemia_prov = df.groupby(["Provincia", "Dx_Anemia"]).size().reset_index(name="Cantidad")
    st.plotly_chart(px.bar(anemia_prov, x="Provincia", y="Cantidad", color="Dx_Anemia",
                           title="Casos de Anemia por Provincia"), use_container_width=True)

# ---------- ENTRENAMIENTO ----------
elif menu == "⚙️ Entrenamiento":
    set_background("entrenamiento.jpg")
    st.markdown('<div class="section-title">⚙️ ENTRENAMIENTO DEL MODELO (Regresión Logística)</div>', unsafe_allow_html=True)
    st.markdown("""
    **La Regresión Logística** permite predecir la probabilidad de un evento categórico (en este caso,
    el tipo de anemia) en función de variables como la edad, edad gestacional, provincia y nivel de hemoglobina.
    """)

    X = df[["Edad", "Edad_Gest", "Provincia", "Hemoglobina"]]
    y = df["Dx_Anemia"]

    num_cols = ["Edad", "Edad_Gest", "Hemoglobina"]
    cat_cols = ["Provincia"]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, multi_class="multinomial"))
    ])

    test_size = st.slider("📏 Tamaño del conjunto de prueba (%)", 10, 40, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    if st.button("🚀 Entrenar Modelo"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        st.success("✅ Modelo entrenado correctamente.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1-Score", f"{f1:.3f}")

        cm = confusion_matrix(y_test, y_pred)
        st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Viridis",
                                  title="Matriz de Confusión"), use_container_width=True)
        os.makedirs("models_saved", exist_ok=True)
        joblib.dump(model, "models_saved/model_logistic.joblib")
        st.info("💾 Modelo guardado correctamente.")

# ---------- PREDICCIÓN ----------
elif menu == "🔮 Predicción":
    set_background("prediccion.jpg")
    st.markdown('<div class="section-title">🔮 PREDICCIÓN DE ANEMIA EN NUEVA GESTANTE</div>', unsafe_allow_html=True)

    model_path = "models_saved/model_logistic.joblib"
    if not os.path.exists(model_path):
        st.warning("⚠️ Primero entrena el modelo.")
        st.stop()
    model = joblib.load(model_path)

    c1, c2, c3, c4 = st.columns(4)
    edad = c1.number_input("Edad", 10, 50, 25)
    edad_gest = c2.number_input("Edad Gestacional", 5, 45, 20)
    provincia = c3.selectbox("Provincia", sorted(df["Provincia"].unique()))
    hemoglobina = c4.number_input("Nivel de Hemoglobina", 5.0, 20.0, 13.5)

    if st.button("🔍 Predecir Diagnóstico"):
        new_data = pd.DataFrame([[edad, edad_gest, provincia, hemoglobina]],
                                columns=["Edad", "Edad_Gest", "Provincia", "Hemoglobina"])
        pred = model.predict(new_data)
        proba = model.predict_proba(new_data)
        st.success(f"Diagnóstico Predicho: **{pred[0]}**")
        prob_df = pd.DataFrame(proba, columns=model.classes_)
        st.plotly_chart(px.bar(prob_df.melt(var_name="Diagnóstico", value_name="Probabilidad"),
                               x="Diagnóstico", y="Probabilidad", color="Diagnóstico", text_auto=".2f",
                               title="Probabilidad de cada diagnóstico"), use_container_width=True)

# ---------- ESTADÍSTICAS ----------
elif menu == "📊 Estadísticas por Provincia":
    set_background("estadistica.jpg")
    st.markdown('<div class="section-title">📊 ESTADÍSTICAS POR PROVINCIA</div>', unsafe_allow_html=True)

    prov_sel = st.selectbox("Selecciona una provincia", sorted(df["Provincia"].unique()))
    df_prov = df[df["Provincia"] == prov_sel]
    st.metric("Gestantes registradas", len(df_prov))
    st.metric("Promedio de Hemoglobina", f"{df_prov['Hemoglobina'].mean():.2f}")

    st.plotly_chart(px.histogram(df_prov, x="Dx_Anemia", color="Dx_Anemia",
                                 title=f"Distribución de Anemia en {prov_sel}"), use_container_width=True)
    st.plotly_chart(px.box(df_prov, y="Hemoglobina", color="Dx_Anemia",
                           title=f"Hemoglobina por Diagnóstico en {prov_sel}"), use_container_width=True)

st.markdown("---")
st.caption("Desarrollado con ❤️ por **Kely Zulema Ponce Quispe** | Streamlit + Scikit-learn")
