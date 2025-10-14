# ============================================================
# app.py — Sistema Inteligente para Análisis y Predicción de Anemia en Gestantes
# Desarrollado por: Kely Zulema Ponce Quispe 💜
# ============================================================

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
import joblib, os, io, base64
from pathlib import Path

# ---------- CONFIGURACIÓN ----------
st.set_page_config(page_title="Gestantes - Puno", layout="wide", page_icon="🤰")

# ---------- FONDO DINÁMICO ----------
def set_background(image_file):
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
        [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background: rgba(0,0,0,0);
        }}
        h1, h2, h3, h4, h5 {{
            color: white !important;
            font-weight: bold;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        pass


# ---------- SIDEBAR ----------
st.sidebar.title("📚 Navegación")
st.sidebar.image("fondo_inicio.jpg", use_container_width=True)
menu = st.sidebar.radio(
    "Selecciona una opción:",
    ["🏠 Inicio", "📈 Análisis de Datos", "⚙️ Entrenamiento", "🔮 Predicción", "📊 Estadísticas por Provincia"]
)

# ---------- INICIO ----------
if menu == "🏠 Inicio":
    set_background("fondo_inicio.jpg")

    st.markdown("""
    <div style="background: rgba(0,0,0,0.6); padding:50px; border-radius:20px; text-align:center;">
        <h1>GESTANTES PRIMER SEMESTRE - PUNO</h1>
        <h3>Sistema Inteligente para el Análisis y Predicción de Anemia</h3>
        <p style="color:#e0e0e0; font-size:20px;">Basado en Regresión Logística Multiclase</p>
        <p style="color:#ffd6ff; font-size:18px;">Desarrollado por <b>Kely Zulema Ponce Quispe</b> 💜</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ---------- CARGA DE DATOS ----------
uploaded_file = st.sidebar.file_uploader("📤 Cargar archivo CSV", type=["csv"])
if uploaded_file is None:
    st.info("👆 Sube tu archivo CSV en la barra lateral para comenzar.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file, sep=";", encoding="utf-8")
except:
    df = pd.read_csv(uploaded_file, encoding="latin1")

df.columns = [c.strip() for c in df.columns]
required = ["Edad", "Edad_Gest", "Provincia", "Hemoglobina", "Dx_Anemia"]

missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"❌ Faltan columnas requeridas: {missing}")
    st.stop()

# Normalización de tipos
df["Provincia"] = df["Provincia"].fillna("Desconocido").astype(str)
for col in ["Edad", "Edad_Gest", "Hemoglobina"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

st.sidebar.success("✅ Archivo cargado correctamente")

# ---------- ANÁLISIS DE DATOS ----------
if menu == "📈 Análisis de Datos":
    set_background("analisis.jpg")
    st.header("📊 Análisis Exploratorio de Datos")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Gestantes", len(df))
    col2.metric("Provincias Registradas", df["Provincia"].nunique())
    col3.metric("Edad Gestacional Promedio", f"{df['Edad_Gest'].mean():.1f} sem.")

    # Distribución por provincia
    prov_counts = df["Provincia"].value_counts().reset_index()
    prov_counts.columns = ["Provincia", "N° Gestantes"]
    fig_prov = px.bar(prov_counts, x="Provincia", y="N° Gestantes", color="Provincia",
                      title="Distribución de Gestantes por Provincia")
    st.plotly_chart(fig_prov, use_container_width=True)

    # Distribución por diagnóstico
    anemia_counts = df["Dx_Anemia"].value_counts().reset_index()
    anemia_counts.columns = ["Diagnóstico", "Frecuencia"]
    fig_anemia = px.pie(anemia_counts, names="Diagnóstico", values="Frecuencia",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        title="Distribución General de Tipos de Anemia")
    st.plotly_chart(fig_anemia, use_container_width=True)

    # Niveles de hemoglobina
    st.subheader("💉 Distribución de Hemoglobina")
    fig_hemo = px.histogram(df, x="Hemoglobina", nbins=20, title="Distribución de Niveles de Hemoglobina")
    st.plotly_chart(fig_hemo, use_container_width=True)

    # Análisis de anemia por provincia
    st.subheader("📍 Casos de Anemia por Provincia")
    anemia_prov = df.groupby(["Provincia", "Dx_Anemia"]).size().reset_index(name="Cantidad")
    fig_anemia_prov = px.bar(anemia_prov, x="Provincia", y="Cantidad", color="Dx_Anemia",
                             barmode="group", title="Distribución de Anemia por Provincia")
    st.plotly_chart(fig_anemia_prov, use_container_width=True)

# ---------- ENTRENAMIENTO ----------
elif menu == "⚙️ Entrenamiento":
    st.header("⚙️ Entrenamiento del Modelo de Regresión Logística (Anemia vs No Anemia)")

    st.markdown("""
    <div style="background-color:#3b1f5c; padding:20px; border-radius:15px; color:white; font-size:18px;">
    <h3 style='color:#ffffff;'>📘 Objetivo del Entrenamiento</h3>
    Entrenar un modelo de <b>Regresión Logística</b> que determine si una gestante tiene <b>anemia</b> (leve o moderada) 
    o <b>no anemia</b> (normal), considerando las variables:
    <b>Edad</b>, <b>Edad Gestacional</b>, <b>Provincia</b> y <b>Hemoglobina</b>.
    <hr>
    <b>La Regresión Logística</b> es ideal para clasificar entre dos categorías: 
    1️⃣ Normal vs 2️⃣ Anemia.
    </div>
    """, unsafe_allow_html=True)

    required = ["Edad", "Edad_Gest", "Provincia", "Hemoglobina", "Dx_Anemia"]
    for req in required:
        if req not in df.columns:
            st.error(f"❌ Falta la columna requerida: {req}")
            st.stop()

    # Agrupar en binaria
    df["Dx_Binaria"] = df["Dx_Anemia"].apply(lambda x: "Anemia" if "Anemia" in x else "Normal")

    X = df[["Edad", "Edad_Gest", "Provincia", "Hemoglobina"]]
    y = df["Dx_Binaria"]

    X["Provincia"] = X["Provincia"].fillna("Desconocido").astype(str)
    X["Edad"] = X["Edad"].fillna(X["Edad"].median())
    X["Edad_Gest"] = X["Edad_Gest"].fillna(X["Edad_Gest"].median())
    X["Hemoglobina"] = X["Hemoglobina"].fillna(X["Hemoglobina"].median())

    num_cols = ["Edad", "Edad_Gest", "Hemoglobina"]
    cat_cols = ["Provincia"]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])

    pipeline = Pipeline([
        ("preproc", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    test_size = st.slider("📏 Tamaño del conjunto de prueba (%)", 10, 40, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    if st.button("🚀 Entrenar Modelo"):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label="Anemia", zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label="Anemia", zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label="Anemia", zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Precisión Global", f"{acc:.3f}")
        c2.metric("💉 Exactitud (Precision)", f"{prec:.3f}")
        c3.metric("❤️ Detección (Recall)", f"{rec:.3f}")
        c4.metric("⚖️ Equilibrio (F1)", f"{f1:.3f}")

        # Matriz de confusión 2x2
        cm = confusion_matrix(y_test, y_pred, labels=["Anemia", "Normal"])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Cálculos con validación
        total = tp + tn + fp + fn
        precision_val = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) != 0 else 0
        acc_val = (tp + tn) / total if total != 0 else 0
        f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) != 0 else 0

        st.markdown("""
        <div style="background-color:#f3e5f5; padding:15px; border-radius:10px; color:#3b1f5c; font-size:18px;">
        <h4>🧩 Matriz de Confusión (2x2)</h4>
        <p>Esta matriz muestra las predicciones del modelo frente a los valores reales:</p>
        <ul>
          <li>🟢 <b>TP (Verdaderos Positivos):</b> predijo anemia y realmente era anemia.</li>
          <li>🔵 <b>TN (Verdaderos Negativos):</b> predijo normal y realmente era normal.</li>
          <li>🟡 <b>FP (Falsos Positivos):</b> predijo anemia pero era normal (Error Tipo I).</li>
          <li>🔴 <b>FN (Falsos Negativos):</b> predijo normal pero era anemia (Error Tipo II).</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        cm_df = pd.DataFrame(
            [[tp, fn], [fp, tn]],
            index=["Real: Anemia", "Real: Normal"],
            columns=["Predicho: Anemia", "Predicho: Normal"]
        )

        fig_simple = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Purples",
            title="Matriz de Confusión Simplificada (2×2)",
            labels=dict(x="Predicción del Modelo", y="Valor Real")
        )
        st.plotly_chart(fig_simple, use_container_width=False, width=400, height=400)

        # Mostrar resultados interpretados
        st.markdown(f"""
        <div style="background-color:#ede7f6; padding:15px; border-radius:10px; color:#3b1f5c; font-size:17px;">
        <h4>📖 Interpretación y Cálculo de Métricas</h4>
        <ul>
          <li>🎯 <b>Precisión Global:</b> (TP + TN) / Total = {acc_val:.3f}</li>
          <li>💉 <b>Exactitud (Precision):</b> TP / (TP + FP) = {precision_val:.3f}</li>
          <li>❤️ <b>Detección (Recall):</b> TP / (TP + FN) = {recall_val:.3f}</li>
          <li>⚖️ <b>Equilibrio (F1):</b> 2×(Precision×Recall)/(Precision+Recall) = {f1_val:.3f}</li>
        </ul>
        <p>💡 Los colores más oscuros indican mayor cantidad de casos. 
        Idealmente, los valores altos deben concentrarse en la <b>diagonal principal</b> (TP y TN).</p>
        </div>
        """, unsafe_allow_html=True)

        os.makedirs("models_saved", exist_ok=True)
        joblib.dump(pipeline, "models_saved/model_logistic_binary.joblib")
        st.success("💾 Modelo guardado correctamente como modelo binario en 'models_saved/model_logistic_binary.joblib'")


# ---------- PREDICCIÓN ----------
elif menu == "🔮 Predicción":
    set_background("prediccion.jpg")
    st.header("🔮 Predicción para una Nueva Gestante")

    model_path = "models_saved/model_logistic.joblib"
    if not os.path.exists(model_path):
        st.warning("⚠️ Entrena el modelo primero en la sección anterior.")
    else:
        model = joblib.load(model_path)
        c1, c2, c3, c4 = st.columns(4)
        edad = c1.number_input("Edad", 10, 50, 25)
        edad_gest = c2.number_input("Edad Gestacional (semanas)", 5, 45, 20)
        provincia = c3.selectbox("Provincia", sorted(df["Provincia"].unique()))
        hemoglobina = c4.number_input("Hemoglobina (g/dL)", 5.0, 18.0, 13.5)

        if st.button("🔍 Predecir Diagnóstico"):
            new_df = pd.DataFrame([[edad, edad_gest, provincia, hemoglobina]],
                                  columns=["Edad", "Edad_Gest", "Provincia", "Hemoglobina"])
            pred = model.predict(new_df)
            proba = model.predict_proba(new_df)

            st.success(f"🩺 Diagnóstico Predicho: **{pred[0]}**")

            prob_df = pd.DataFrame(proba, columns=model.classes_)
            fig_proba = px.bar(prob_df.melt(var_name="Diagnóstico", value_name="Probabilidad"),
                               x="Diagnóstico", y="Probabilidad", color="Diagnóstico", text_auto=".2f",
                               title="Probabilidades de Diagnóstico")
            st.plotly_chart(fig_proba, use_container_width=True)

# ---------- ESTADÍSTICAS ----------
elif menu == "📊 Estadísticas por Provincia":
    set_background("estadistica.jpg")
    st.header("📊 Estadísticas por Provincia")

    prov_sel = st.selectbox("Selecciona una Provincia:", sorted(df["Provincia"].dropna().unique()))
    df_prov = df[df["Provincia"] == prov_sel]
    st.metric("Gestantes Registradas", len(df_prov))

    fig_anemia_prov = px.histogram(df_prov, x="Dx_Anemia", color="Dx_Anemia",
                                   title=f"Distribución de Tipos de Anemia en {prov_sel}")
    st.plotly_chart(fig_anemia_prov, use_container_width=True)

st.markdown("---")
st.caption("Desarrollado por 💜 **Kely Zulema Ponce Quispe** | Streamlit + Scikit-Learn + Plotly")
