# --- CONFIGURACIÓN INICIAL ---
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Greeks & Volatility Visualizer v1.2", layout="wide")

# --- HEADER ---
st.title("📈 Greeks & Volatility Visualizer v1.2 – Interactive Pro Edition")
st.caption("_by cpetto._")
st.markdown("---")

# --- FUNCIÓN DE SYNC ENTRE SLIDER Y INPUT ---
def synced_slider_input(label, key_slider, key_input, min_val, max_val, default_val, step_val):
    if key_slider not in st.session_state:
        st.session_state[key_slider] = default_val
    if key_input not in st.session_state:
        st.session_state[key_input] = default_val

    def slider_change():
        st.session_state[key_input] = st.session_state[key_slider]

    def input_change():
        st.session_state[key_slider] = st.session_state[key_input]

    st.slider(label + " (slider)", 
              min_val, max_val, key=key_slider, step=step_val, on_change=slider_change)
    st.number_input(label + " (input manual)", 
                    min_value=min_val, max_value=max_val, key=key_input, step=step_val, on_change=input_change)

# --- SIDEBAR: PARÁMETROS ---
st.sidebar.header("Parámetros de la Opción")

option_type = st.sidebar.selectbox("Tipo de Opción", ["call", "put"])
synced_slider_input("Tiempo hasta vencimiento (T)", "T_slider", "T_input", 0.01, 2.0, 1.0, 0.01)
synced_slider_input("Precio Subyacente (S0)", "S0_slider", "S0_input", 10, 500, 100, 1)
synced_slider_input("Strike (K)", "K0_slider", "K0_input", 10, 500, 100, 1)
synced_slider_input("Tasa libre de riesgo (r)", "r0_slider", "r0_input", 0.0, 0.2, 0.05, 0.001)
synced_slider_input("Volatilidad (σ)", "sigma0_slider", "sigma0_input", 0.05, 1.0, 0.2, 0.01)

smile_view = st.sidebar.radio("Visualizar Volatility Smile:", ["2D", "3D"])

# --- CAPTURAR VALORES SINCRONIZADOS ---
T0 = st.session_state["T_input"]
S0 = st.session_state["S0_input"]
K0 = st.session_state["K0_input"]
r0 = st.session_state["r0_input"]
sigma0 = st.session_state["sigma0_input"]

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

# --- FUNCIONES PARA CARDS ---
def greek_card(title, current_val, data_series, surface_function, color):
    st.subheader(title)
    st.metric(label="Valor actual", value=f"{current_val:.4f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data_series, mode='lines', line=dict(color=color, width=2)))
    fig.update_layout(height=100, margin=dict(l=10, r=10, t=10, b=10), xaxis_visible=False, yaxis_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Superficie 3D
    S_grid, T_grid = np.meshgrid(np.linspace(0.5*S0, 1.5*S0, 40), np.linspace(0.01, 2.0, 40))
    Z = surface_function(S_grid, K0, T_grid, r0, sigma0)

    fig3d = plt.figure(figsize=(6,4))
    ax = fig3d.add_subplot(111, projection='3d')
    ax.plot_surface(S_grid, T_grid, Z, cmap=color, edgecolor='none')
    ax.set_xlabel('S')
    ax.set_ylabel('T')
    ax.set_zlabel(title)
    ax.set_title(f"{title} Surface")
    st.pyplot(fig3d)

# --- MOSTRAR CARDS DE GREEKS ---
st.header("Sensibilidades (Greeks)")

col1, col2, col3 = st.columns(3)
col4, col5 = st.columns(2)

with col1:
    greek_card("Delta", delta_vals[index_S0], delta_vals, delta_call, "Blues")
with col2:
    greek_card("Gamma", gamma_vals[index_S0], gamma_vals, gamma, "Greens")
with col3:
    greek_card("Theta", theta_vals[index_S0], theta_vals, theta_call, "Oranges")
with col4:
    greek_card("Vega", vega_vals[index_S0], vega_vals, vega, "Purples")
with col5:
    greek_card("Rho", rho_vals[index_S0], rho_vals, rho_call, "Reds")

# --- VOLATILITY SMILE ---
st.header("Volatility Smile")

strikes = np.linspace(0.8*S0, 1.2*S0, 50)
times = np.linspace(0.01, 2.0, 50)
Strike_grid, Time_grid = np.meshgrid(strikes, times)
implied_vol_grid = sigma0 + 0.05 * ((Strike_grid - S0)/S0)**2

if smile_view == "2D":
    fig_smile, ax_smile = plt.subplots(figsize=(8,4))
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
    st.pyplot(fig_smile3d)

# --- FOOTER ---
st.markdown("---")
st.caption("Built with 💙 by Derivatives Specialist AI")
