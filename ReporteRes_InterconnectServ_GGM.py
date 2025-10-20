import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from PIL import Image

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="Dashboard de Deserción de Clientes")
warnings.filterwarnings('ignore')

# --- Título Principal ---
st.title("Proyecto Final: Análisis de Deserción de Clientes")

# --- Carga y Cacheo de Datos ---
@st.cache_data
def load_data():
    """Carga y fusiona todos los datos del proyecto."""
    try:
        # Cambiado a rutas relativas para que funcione en GitHub y Streamlit Cloud
        contact = pd.read_csv('data/contract.csv')
        internet = pd.read_csv('data/internet.csv')
        personal = pd.read_csv('data/personal.csv')
        phone = pd.read_csv('data/phone.csv')
    except FileNotFoundError:
        st.error("Error: Archivos CSV no encontrados. Asegúrate de que están en una carpeta llamada 'data'.")
        return None

    # Unir los dataframes como en el notebook
    df_universal = (internet
                    .merge(phone, on='customerID', how='inner')
                    .merge(personal, on='customerID', how='inner')
                    .merge(contact, on='customerID', how='inner'))
    
    # Crear la variable objetivo
    df_universal['tasa_permanencia'] = df_universal['EndDate'].apply(lambda x: 1 if x == 'No' else 0)
    return df_universal

# --- Entrenamiento y Cacheo de Modelos ---
@st.cache_data
def train_models(df):
    """Preprocesa los datos y entrena ambos modelos (LogReg y RF) usando GridSearchCV."""
    
    # 1. Preparación de datos para el modelo
    categoricas = [
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]
    df_modelo = pd.get_dummies(df[categoricas + ['tasa_permanencia']], drop_first=True)
    
    X = df_modelo.drop(columns=['tasa_permanencia'])
    y = df_modelo['tasa_permanencia']

    # Usar el mismo random_state que el notebook para consistencia
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=12345, stratify=y
    )

    # Diccionario para almacenar todos los resultados
    results = {}

    # --- Modelo 1: Regresión Logística ---
    
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # GridSearchCV para LogReg
    param_grid_log = {'C': np.logspace(-3, 2, 10)}
    grid_log = GridSearchCV(
        LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000, random_state=12345),
        param_grid_log, cv=5, scoring='f1', n_jobs=-1
    )
    grid_log.fit(X_train_scaled, y_train)
    best_model_log = grid_log.best_estimator_
    
    # Predicciones y Métricas LogReg
    y_prob_log = best_model_log.predict_proba(X_test_scaled)[:, 1]
    y_pred_log = (y_prob_log >= 0.5).astype(int)
    
    metrics_log = {
        "Accuracy": accuracy_score(y_test, y_pred_log),
        "Precision": precision_score(y_test, y_pred_log),
        "Recall": recall_score(y_test, y_pred_log),
        "F1 Score": f1_score(y_test, y_pred_log),
        "ROC-AUC": roc_auc_score(y_test, y_prob_log)
    }
    
    # Datos para gráficos LogReg
    fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
    cm_log = confusion_matrix(y_test, y_pred_log)
    
    results['log_reg'] = {
        "best_params": grid_log.best_params_,
        "metrics": metrics_log,
        "roc_data": (fpr_log, tpr_log, metrics_log["ROC-AUC"]),
        "cm_data": cm_log,
        "y_prob": y_prob_log,
        "y_pred": y_pred_log
    }

    # --- Modelo 2: Random Forest ---
    # Usar los mismos parámetros del notebook
    param_grid_rf = {
        'n_estimators': [200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10]
    }
    
    # Usar random_state=42 como en el notebook
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    
    grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
    grid_rf.fit(X_train, y_train) # RF no usa datos escalados
    best_model_rf = grid_rf.best_estimator_

    # Predicciones y Métricas RF
    y_prob_rf = best_model_rf.predict_proba(X_test)[:, 1]
    y_pred_rf = (y_prob_rf >= 0.5).astype(int)

    metrics_rf = {
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf),
        "Recall": recall_score(y_test, y_pred_rf),
        "F1 Score": f1_score(y_test, y_pred_rf),
        "ROC-AUC": roc_auc_score(y_test, y_prob_rf)
    }
    
    # Datos para gráficos RF
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    
    results['random_forest'] = {
        "best_params": grid_rf.best_params_,
        "metrics": metrics_rf,
        "roc_data": (fpr_rf, tpr_rf, metrics_rf["ROC-AUC"]),
        "cm_data": cm_rf,
        "y_prob": y_prob_rf,
        "y_pred": y_pred_rf
    }

    # --- Datos de prueba para comparación ---
    results['test_data'] = {
        "X_test_index": X_test.index,
        "y_test": y_test
    }

    return results

# --- Funciones de Gráficos (Helpers) ---

# Gráficos de EDA
def plot_count(df, column, title, xlabel, palette='pastel'):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(
        data=df,
        x=column,
        palette=palette,
        order=df[column].value_counts().index,
        edgecolor="black",
        ax=ax
    )
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Número de Clientes', fontsize=12)
    sns.despine()
    return fig

def plot_additional_services(df_internet):
    df_melt = df_internet.melt(
        id_vars=['customerID'],
        value_vars=['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    )
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.countplot(
        data=df_melt,
        x='value',
        hue='variable',
        palette='Set2',
        edgecolor="black",
        ax=ax
    )
    ax.set_title('Distribución de Clientes por Servicios Adicionales', fontsize=16, fontweight='bold')
    ax.set_xlabel('Estado del Servicio', fontsize=12)
    ax.set_ylabel('Número de Clientes', fontsize=12)
    ax.legend(title='Servicios', loc='upper right')
    sns.despine()
    return fig

# Gráficos de Modelo
def plot_roc_curve(fpr, tpr, auc, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("Tasa Positiva Falsa")
    ax.set_ylabel("Tasa Positiva Verdadera")
    ax.set_title(title)
    ax.legend()
    return fig

def plot_confusion_matrix(cm, labels, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title(title)
    return fig

# Gráficos de Comparación
def plot_probability_scatter(df_comp):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x="Prob_Logistica",
        y="Prob_RandomForest",
        hue="Real",
        palette={0: "tomato", 1: "seagreen"},
        alpha=0.7,
        edgecolor="black",
        data=df_comp,
        ax=ax
    )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Probabilidad - Regresión Logística", fontsize=12)
    ax.set_ylabel("Probabilidad - Random Forest", fontsize=12)
    ax.set_title("Comparativa de Probabilidades Predichas", fontsize=14, weight="bold")
    ax.legend(title="Real", labels=["Se va (0)", "Se queda (1)"])
    return fig

def plot_probability_kde(df_comp):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(
        x=df_comp["Prob_Logistica"],
        fill=True,
        color="steelblue",
        alpha=0.5,
        linewidth=2,
        label="Regresión Logística",
        ax=ax
    )
    sns.kdeplot(
        x=df_comp["Prob_RandomForest"],
        fill=True,
        color="darkorange",
        alpha=0.5,
        linewidth=2,
        label="Random Forest",
        ax=ax
    )
    ax.set_title("Comparativa de Distribuciones de Probabilidad", fontsize=14, weight="bold")
    ax.set_xlabel("Probabilidad de Permanencia", fontsize=12)
    ax.set_ylabel("Densidad", fontsize=12)
    ax.legend(title="Modelo")
    return fig


# --- Cuerpo Principal de la App ---

# Cargar datos y modelos
df_universal = load_data()

if df_universal is not None:
    model_results = train_models(df_universal)
    
    # Extraer resultados para facilitar el acceso
    log_reg_res = model_results['log_reg']
    rf_res = model_results['random_forest']
    test_data = model_results['test_data']

    # Crear el DataFrame de comparación
    df_comparacion = pd.DataFrame({
        "customerID": df_universal.loc[test_data["X_test_index"], "customerID"],
        "Prob_Logistica": log_reg_res["y_prob"],
        "Pred_Logistica": log_reg_res["y_pred"],
        "Prob_RandomForest": rf_res["y_prob"],
        "Pred_RandomForest": rf_res["y_pred"],
        "Real": test_data["y_test"].values
    })
    df_comparacion["Diferencia_Prob"] = (df_comparacion["Prob_RandomForest"] - df_comparacion["Prob_Logistica"]).round(3)


    # --- Definición de Pestañas ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Introducción", "Análisis Exploratorio", "Regresión Logística", 
        "Random Forest", "Comparación de Modelos", "Conclusiones"
    ])

    # --- Pestaña 1: Introducción ---
    with tab1:
        st.header("Introducción al Proyecto")
        st.write("Este dashboard presenta un análisis de deserción de clientes...")
        
        # --- MODIFICACIÓN AQUÍ ---
        # Sección del diagrama de flujo eliminada
        # --- FIN DE LA MODIFICACIÓN ---
        
        st.header("Datos Cargados")
        st.write("Vistazo al dataframe universal (primeras 100 filas):")
        st.dataframe(df_universal.head(100))


    # --- Pestaña 2: Análisis Exploratorio (EDA) ---
    with tab2:
        st.header("Análisis Exploratorio de Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.pyplot(plot_count(df_universal, 'EndDate', 'Distribución por Contrato (EndDate)', 'Tipo de Contrato'))
            st.pyplot(plot_count(df_universal, 'gender', 'Distribución por Género', 'Género'))
            st.pyplot(plot_count(df_universal, 'MultipleLines', 'Distribución por Líneas Múltiples', 'Líneas Múltiples'))

        with col2:
            st.pyplot(plot_count(df_universal, 'InternetService', 'Distribución por Tipo de Internet', 'Servicio de Internet', 'muted'))
            st.pyplot(plot_additional_services(df_universal))

    # --- Pestaña 3: Regresión Logística ---
    with tab3:
        st.header("Modelo: Regresión Logística")
        st.subheader("Mejores Hiperparámetros (GridSearchCV)")
        st.json(log_reg_res['best_params'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Métricas de Desempeño")
            metrics_df_log = pd.DataFrame.from_dict(log_reg_res['metrics'], orient='index', columns=['Valor'])
            st.dataframe(metrics_df_log.style.format("{:.4f}"))
            
            st.subheader("Primeras 20 Predicciones")
            st.dataframe(df_comparacion[['customerID', 'Prob_Logistica', 'Pred_Logistica', 'Real']].head(20))

        with col2:
            st.subheader("Curva ROC")
            fig_roc_log = plot_roc_curve(
                log_reg_res['roc_data'][0], 
                log_reg_res['roc_data'][1], 
                log_reg_res['roc_data'][2], 
                "Curva ROC - Regresión Logística"
            )
            st.pyplot(fig_roc_log)
            
            st.subheader("Matriz de Confusión")
            fig_cm_log = plot_confusion_matrix(
                log_reg_res['cm_data'], 
                ["Se va", "Se queda"], 
                "Matriz de Confusión - Regresión Logística"
            )
            st.pyplot(fig_cm_log)

    # --- Pestaña 4: Random Forest ---
    with tab4:
        st.header("Modelo: Random Forest")
        st.subheader("Mejores Hiperparámetros (GridSearchCV)")
        st.json(rf_res['best_params'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Métricas de Desempeño")
            metrics_df_rf = pd.DataFrame.from_dict(rf_res['metrics'], orient='index', columns=['Valor'])
            st.dataframe(metrics_df_rf.style.format("{:.4f}"))
            
            st.subheader("Primeras 20 Predicciones")
            st.dataframe(df_comparacion[['customerID', 'Prob_RandomForest', 'Pred_RandomForest', 'Real']].head(20))
            
        with col2:
            st.subheader("Curva ROC")
            fig_roc_rf = plot_roc_curve(
                rf_res['roc_data'][0], 
                rf_res['roc_data'][1], 
                rf_res['roc_data'][2], 
                "Curva ROC - Random Forest"
            )
            st.pyplot(fig_roc_rf)
            
            st.subheader("Matriz de Confusión")
            fig_cm_rf = plot_confusion_matrix(
                rf_res['cm_data'], 
                ["Se va", "Se queda"], 
                "Matriz de Confusión - Random Forest"
            )
            st.pyplot(fig_cm_rf)
            
    # --- Pestaña 5: Comparación de Modelos ---
    with tab5:
        st.header("Comparación de Modelos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.pyplot(plot_probability_scatter(df_comparacion))
        
        with col2:
            st.pyplot(plot_probability_kde(df_comparacion))
            
        st.subheader("Tabla Comparativa Detallada")
        st.write("Muestra las probabilidades y predicciones de ambos modelos lado a lado.")
        st.dataframe(df_comparacion)

    # --- Pestaña 6: Conclusiones ---
    with tab6:
        st.header("Conclusiones del proyecto y análisis de resultados")
        
        st.markdown("""
        A lo largo de este proyecto, se han explorado los resultados de dos modelos que son ampliamente utilizados en problemas de clasificación: Regresión Logística y Random Forest. La idea de aplicar 
                        
                        estos modelos al mismo tiempo es para comarar su desempeño y entender cuál se adapta mejor a nuestro conjunto de datos específico. En este dashboard se presentaron propiamente sus 

                        respectivos resultados, métricas y comparativas. Esta es una práctica común en ciencia de datos para asegurar que se elija el modelo más adecuado para la tarea en cuestión.

                        Estos resultados se pueden usar aplicamente en contexto de negocio para predecir la deserción de clientes y tomar medidas proactivas para retener a los clientes valiosos, apoyándose

                        de la teoría económica (teoría del productor y del consumidor), para la asignación eficiente de recursos en estrategias de retención del cliente.
        """)
        
        st.markdown("---")
        st.subheader("Métricas Comparativas")
        
        metrics_comparison = {
            "Regresión Logística": log_reg_res['metrics'],
            "Random Forest": rf_res['metrics']
        }
        metrics_comp_df = pd.DataFrame(metrics_comparison).T
        st.dataframe(metrics_comp_df.style.format("{:.4f}"))

