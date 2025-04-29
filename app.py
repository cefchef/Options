# --- CONFIGURACIÓN INICIAL Y DARK MODE REAL ---
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm

# Configuración de la página
st.set_page_config(page_title="Greeks & Volatility Visualizer v1.2", layout="wide")

# --- HEADER PRINCIPAL ---
st.title("📈 Greeks & Volatility Visualizer v1.2 – Interactive Pro Edition")
st.caption("_by cpetto._")
st.markdown("---")

# --- GESTIÓN DE TEMAS OSCURO/CLARO DINÁMICO ---
def set_background_color(dark_mode):
    page_bg = "#0E1117" if dark_mode else "#FFFFFF"
    text_color = "#FAFAFA" if dark_mode else "#000000"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {page_bg};
            color: {text_color};
            transition: background-color 0.5s ease, color 0.5s ease;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Sidebar de configuraciones
st.sidebar.header("Configuraciones de Visualización")
dark_mode = st.sidebar.toggle('🌞 | 🌙')
set_background_color(dark_mode)

# --- INPUTS MANUALES + SLIDERS SINCRONIZADOS ---
st.sidebar.header("Parámetros de la Opción")

def sync_input_slider(label, min_val, max_val, default, step):
    slider = st.sidebar.slider(f"{label} (slider)", min_val, max_val, default, step=step)
    number = st.sidebar.number_input(f"{label} (input manual)", min_value=min_val, max_value=max_val, value=slider, step=step)
    if number != slider:
        slider = number
    return slider

S0 = sync_input_slider("Precio Subyacente (S0)", 10, 500, 100, 1)
K0 = sync_input_slider("Strike (K)", 10, 500, 100, 1)
r0 = sync_input_slider("Tasa libre de riesgo (r)", 0.0, 0.2, 0.05, 0.001)
sigma0 = sync_input_slider("Volatilidad (σ)", 0.05, 1.0, 0.2, 0.01)
T0 = sync_input_slider("Tiempo hasta vencimiento (T)", 0.01, 2.0, 1.0, 0.01)

option_type = st.sidebar.selectbox("Tipo de Opción", ["call", "put"])
smile_view = st.sidebar.radio("Visualizar Volatility Smile:", ["2D", "3D"])

# --- FUNCIONES FINANCIERAS ---
def d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def delta_call(S, K, T, r, sigma):
    return norm.cdf(d1(S,K,T,r,sigma))

def delta_put(S, K, T, r, sigma):
    return norm.cdf(d1(S,K,T,r,sigma)) - 1

def gamma(S, K, T, r, sigma):
    return norm.pdf(d1(S,K,T,r,sigma)) / (S * sigma * np.sqrt(T))

def theta_call(S, K, T, r, sigma):
    return (-S * norm.pdf(d1(S,K,T,r,sigma)) * sigma / (2*np.sqrt(T))
            - r*K*np.exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma)))

def theta_put(S, K, T, r, sigma):
    return (-S * norm.pdf(d1(S,K,T,r,sigma)) * sigma / (2*np.sqrt(T))
            + r*K*np.exp(-r*T)*norm.cdf(-d2(S,K,T,r,sigma)))

def vega(S, K, T, r, sigma):
    return S * norm.pdf(d1(S,K,T,r,sigma)) * np.sqrt(T)

def rho_call(S, K, T, r, sigma):
    return K * T * np.exp(-r*T) * norm.cdf(d2(S,K,T,r,sigma))

def rho_put(S, K, T, r, sigma):
    return -K * T * np.exp(-r*T) * norm.cdf(-d2(S,K,T,r,sigma))

# --- CÁLCULO DE GREEKS ---
S_range = np.linspace(0.5 * S0, 1.5 * S0, 100)

if option_type == "call":
    delta_vals = delta_call(S_range, K0, T0, r0, sigma0)
    theta_vals = theta_call(S_range, K0, T0, r0, sigma0)
    rho_vals = rho_call(S_range, K0, T0, r0, sigma0)
else:
    delta_vals = delta_put(S_range, K0, T0, r0, sigma0)
    theta_vals = theta_put(S_range, K0, T0, r0, sigma0)
    rho_vals = rho_put(S_range, K0, T0, r0, sigma0)

gamma_vals = gamma(S_range, K0, T0, r0, sigma0)
vega_vals = vega(S_range, K0, T0, r0, sigma0)
index_S0 = np.argmin(np.abs(S_range - S0))

# --- FUNCIÓN PARA CARDS CON GREEKS ---
def greek_card_with_surface(title, current_val, data_series, surface_function, colormap, tooltip_text):
    st.markdown(f"<h3 style='color: {'white' if dark_mode else '#222222'};'>{title} ℹ️</h3>", unsafe_allow_html=True)
    st.caption(tooltip_text)

    spark_color = {
        "Delta": "#1f77b4",
        "Gamma": "#2ca02c",
        "Theta": "#ff7f0e",
        "Vega": "#9467bd",
        "Rho": "#d62728"
    }

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data_series, mode='lines', line=dict(color=spark_color[title], width=2)))
    fig.update_layout(height=100, margin=dict(l=10, r=10, t=10, b=10), xaxis_visible=False, yaxis_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    S_grid, T_grid = np.meshgrid(np.linspace(0.5*S0, 1.5*S0, 40), np.linspace(0.01, 2.0, 40))
    if option_type == "call" and title == "Delta":
        Z = delta_call(S_grid, K0, T_grid, r0, sigma0)
    elif option_type == "put" and title == "Delta":
        Z = delta_put(S_grid, K0, T_grid, r0, sigma0)
    elif option_type == "call" and title == "Theta":
        Z = theta_call(S_grid, K0, T_grid, r0, sigma0)
    elif option_type == "put" and title == "Theta":
        Z = theta_put(S_grid, K0, T_grid, r0, sigma0)
    elif option_type == "call" and title == "Rho":
        Z = rho_call(S_grid, K0, T_grid, r0, sigma0)
    elif option_type == "put" and title == "Rho":
        Z = rho_put(S_grid, K0, T_grid, r0, sigma0)
    else:
        Z = surface_function(S_grid, K0, T_grid, r0, sigma0)

    fig3d = plt.figure(figsize=(6, 4))
    ax = fig3d.add_subplot(111, projection='3d')
    ax.plot_surface(S_grid, T_grid, Z, cmap=colormap, edgecolor='none')
    ax.set_xlabel('S')
    ax.set_ylabel('T')
    ax.set_zlabel(title)
    ax.set_title(f"{title} Surface")
    if dark_mode:
        ax.set_facecolor('#111111')
        fig3d.patch.set_facecolor('#111111')
    st.pyplot(fig3d)

# --- MOSTRAR LAS GREEKS ---
st.markdown(f"<h2 style='color: {'white' if dark_mode else '#222222'};'>Sensibilidades (Greeks)</h2>", unsafe_allow_html=True)

tooltips = {
    "Delta": "Sensibilidad del precio de la opción al movimiento del precio del subyacente.",
    "Gamma": "Tasa de cambio de Delta respecto al precio del subyacente.",
    "Theta": "Sensibilidad del precio de la opción al paso del tiempo (decadencia temporal).",
    "Vega": "Sensibilidad del precio de la opción a cambios en la volatilidad del subyacente.",
    "Rho": "Sensibilidad del precio de la opción a cambios en la tasa libre de riesgo."
}

col1, col2, col3 = st.columns(3)
col4, col5 = st.columns(2)

with col1:
    greek_card_with_surface("Delta", delta_vals[index_S0], delta_vals, delta_call, "Blues", tooltips["Delta"])
with col2:
    greek_card_with_surface("Gamma", gamma_vals[index_S0], gamma_vals, gamma, "Greens", tooltips["Gamma"])
with col3:
    greek_card_with_surface("Theta", theta_vals[index_S0], theta_vals, theta_call, "Oranges", tooltips["Theta"])
with col4:
    greek_card_with_surface("Vega", vega_vals[index_S0], vega_vals, vega, "Purples", tooltips["Vega"])
with col5:
    greek_card_with_surface("Rho", rho_vals[index_S0], rho_vals, rho_call, "Reds", tooltips["Rho"])

# --- VOLATILITY SMILE ---
st.markdown(f"<h2 style='color: {'white' if dark_mode else '#222222'};'>Volatility Smile</h2>", unsafe_allow_html=True)

strikes = np.linspace(0.8*S0, 1.2*S0, 50)
times = np.linspace(0.01, 2.0, 50)
Strike_grid, Time_grid = np.meshgrid(strikes, times)
implied_vol_grid = sigma0 + 0.05 * ((Strike_grid - S0)/S0)**2

if smile_view == "2D":
    fig_smile, ax_smile = plt.subplots(figsize=(8,4))
    if dark_mode:
        fig_smile.patch.set_facecolor('#111111')
        ax_smile.set_facecolor('#111111')
        ax_smile.spines['bottom'].set_color('white')
        ax_smile.spines['left'].set_color('white')
        ax_smile.tick_params(axis='x', colors='white')
        ax_smile.tick_params(axis='y', colors='white')
        ax_smile.yaxis.label.set_color('white')
        ax_smile.xaxis.label.set_color('white')
        ax_smile.title.set_color('white')

    ax_smile.plot(strikes, sigma0 + 0.05 * ((strikes - S0)/S0)**2, color="#4A90E2")
    ax_smile.axvline(S0, color='grey', linestyle='--')
    ax_smile.set_xlabel("Strike")
    ax_smile.set_ylabel("Volatilidad Implícita")
    ax_smile.set_title("Volatility Smile")
    ax_smile.grid(True)
    st.pyplot(fig_smile)

else:
    fig_smile3d = plt.figure(figsize=(8,6))
    ax3d = fig_smile3d.add_subplot(111, projection='3d')
    surf = ax3d.plot_surface(Strike_grid, Time_grid, implied_vol_grid, cmap="viridis", edgecolor='none')
    ax3d.set_xlabel('Strike')
    ax3d.set_ylabel('T')
    ax3d.set_zlabel('Volatilidad Implícita')
    ax3d.set_title('Volatility Smile 3D')
    if dark_mode:
        ax3d.set_facecolor('#111111')
        fig_smile3d.patch.set_facecolor('#111111')
    st.pyplot(fig_smile3d)

# --- FOOTER ---
st.markdown("---")
