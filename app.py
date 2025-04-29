# --- CONFIGURACIÓN GENERAL ---
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Greeks & Volatility Visualizer", layout="wide")

# --- HEADER ---
st.title("📈 Greeks & Volatility Visualizer")
st.caption("_Explore Option Sensitivities and Black-Scholes Outputs with a scientific minimalistic style._")
st.markdown("---")

# --- FUNCIONES FINANCIERAS ---

def d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

# Precios
def call_price(S, K, T, r, sigma):
    return S * norm.cdf(d1(S,K,T,r,sigma)) - K * np.exp(-r*T) * norm.cdf(d2(S,K,T,r,sigma))

def put_price(S, K, T, r, sigma):
    return K * np.exp(-r*T) * norm.cdf(-d2(S,K,T,r,sigma)) - S * norm.cdf(-d1(S,K,T,r,sigma))

# Greeks
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

# --- SIDEBAR DE PARÁMETROS ---
st.sidebar.header("Parámetros de la Opción")
option_type = st.sidebar.selectbox("Tipo de Opción", ["call", "put"])
S0 = st.sidebar.number_input("Spot Price (S0)", value=100.0)
K0 = st.sidebar.number_input("Strike Price (K)", value=100.0)
r0 = st.sidebar.number_input("Tasa libre de riesgo (r)", value=0.05)
sigma0 = st.sidebar.number_input("Volatilidad (σ)", value=0.2)
T0 = st.sidebar.number_input("Tiempo hasta vencimiento (T)", value=1.0)

# --- CÁLCULO PRINCIPAL ---
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

# Cálculo en S0 puntual
index_S0 = np.argmin(np.abs(S_range - S0))

# --- LAYOUT DE CARDS ---
st.subheader("Sensibilidades (Greeks)")

col1, col2, col3 = st.columns(3)
col4, col5 = st.columns(2)

# Función para crear una tarjeta de Greek
def greek_card(title, current_val, data_series, color):
    st.markdown(f"**{title}**")
    st.metric(label="Valor actual", value=f"{current_val:.4f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data_series, mode='lines', line=dict(color=color, width=2)))
    fig.update_layout(height=100, margin=dict(l=10, r=10, t=10, b=10), xaxis_visible=False, yaxis_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with col1:
    greek_card("Delta", delta_vals[index_S0], delta_vals, "#1f77b4")

with col2:
    greek_card("Gamma", gamma_vals[index_S0], gamma_vals, "#2ca02c")

with col3:
    greek_card("Theta", theta_vals[index_S0], theta_vals, "#ff7f0e")

with col4:
    greek_card("Vega", vega_vals[index_S0], vega_vals, "#9467bd")

with col5:
    greek_card("Rho", rho_vals[index_S0], rho_vals, "#d62728")

# --- VOLATILITY SMILE ---
st.subheader("Volatility Smile (Curva)")
strikes = np.linspace(0.8*S0, 1.2*S0, 50)
implied_vol = sigma0 + 0.05 * ((strikes - S0)/S0)**2

fig_smile = plt.figure(figsize=(8,4))
plt.plot(strikes, implied_vol, color="#4A90E2")
plt.axvline(S0, color='grey', linestyle='--')
plt.xlabel("Strike")
plt.ylabel("Volatilidad Implícita")
plt.title("Volatility Smile")
plt.grid(True)
st.pyplot(fig_smile)

# --- VOLATILITY SURFACE 3D ---
st.subheader("Volatility Surface (Superficie)")
strikes_surface = np.linspace(0.8*S0, 1.2*S0, 40)
maturities_surface = np.linspace(0.1, 2.0, 40)
K_mesh, T_mesh = np.meshgrid(strikes_surface, maturities_surface)
vol_surface = sigma0 + 0.05 * ((K_mesh - S0)/S0)**2 + 0.02*(T_mesh - 1)**2

fig_surface = plt.figure(figsize=(10,6))
ax = fig_surface.add_subplot(111, projection='3d')
ax.plot_surface(K_mesh, T_mesh, vol_surface, cmap='coolwarm', edgecolor='none')
ax.set_xlabel('Strike')
ax.set_ylabel('Tiempo hasta Vencimiento (T)')
ax.set_zlabel('Volatilidad Implícita')
ax.set_title('Volatility Surface')
st.pyplot(fig_surface)

# --- FUTURO: Animación de Vega ---
st.subheader("🚧 Animación de Vega (próximamente...)")
st.caption("_Aquí integraremos animaciones científicas sobre la evolución de Vega a lo largo del tiempo._")

# --- FOOTER ---
st.markdown("---")
st.caption("Built with 💙 by Derivatives Specialist AI")

