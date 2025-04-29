# --- CONFIGURACIÓN INICIAL ---
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Greeks & Volatility Visualizer", layout="wide")

# --- HEADER ---
st.title("📈 Greeks & Volatility Visualizer")
st.caption("_by cpetto._")
st.markdown("---")

# --- FUNCIONES FINANCIERAS ---

def d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def call_price(S, K, T, r, sigma):
    return S * norm.cdf(d1(S,K,T,r,sigma)) - K * np.exp(-r*T) * norm.cdf(d2(S,K,T,r,sigma))

def put_price(S, K, T, r, sigma):
    return K * np.exp(-r*T) * norm.cdf(-d2(S,K,T,r,sigma)) - S * norm.cdf(-d1(S,K,T,r,sigma))

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

# --- SIDEBAR PARÁMETROS ---
st.sidebar.header("Parámetros de la Opción")

option_type = st.sidebar.selectbox("Tipo de Opción", ["call", "put"])
S0 = st.sidebar.slider("Precio Subyacente (S0)", 50.0, 150.0, 100.0, step=1.0)
K0 = st.sidebar.slider("Strike (K)", 50.0, 150.0, 100.0, step=1.0)
r0 = st.sidebar.slider("Tasa libre de riesgo (r)", 0.0, 0.2, 0.05, step=0.005)
sigma0 = st.sidebar.slider("Volatilidad (σ)", 0.05, 0.8, 0.2, step=0.01)
T0 = st.sidebar.slider("Tiempo hasta vencimiento (T)", 0.05, 2.0, 1.0, step=0.05)

# --- CALCULAMOS GREEKS ---
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

# Localizar valor en S0
index_S0 = np.argmin(np.abs(S_range - S0))

# --- FUNCIÓN PARA CREAR TARJETAS DE GREEKS ---
def greek_card(title, current_val, data_series, color):
    st.markdown(f"### {title}")
    st.metric(label="Valor actual", value=f"{current_val:.4f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data_series, mode='lines', line=dict(color=color, width=2)))
    fig.update_layout(height=100, margin=dict(l=10, r=10, t=10, b=10), xaxis_visible=False, yaxis_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- MOSTRAMOS TARJETAS ---
st.subheader("Sensibilidades (Greeks)")

col1, col2, col3 = st.columns(3)
col4, col5 = st.columns(2)

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
st.subheader("Volatility Smile")

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

# --- SUPERFICIES 3D INDIVIDUALES ---
st.subheader("Superficies 3D de Greeks")

def plot_surface(greek_name, calc_function, cmap):
    S_grid, T_grid = np.meshgrid(
        np.linspace(0.5*S0, 1.5*S0, 50),
        np.linspace(0.01, 2.0, 50)
    )
    if option_type == "call" and greek_name == "Delta":
        Z = delta_call(S_grid, K0, T_grid, r0, sigma0)
    elif option_type == "put" and greek_name == "Delta":
        Z = delta_put(S_grid, K0, T_grid, r0, sigma0)
    elif option_type == "call" and greek_name == "Theta":
        Z = theta_call(S_grid, K0, T_grid, r0, sigma0)
    elif option_type == "put" and greek_name == "Theta":
        Z = theta_put(S_grid, K0, T_grid, r0, sigma0)
    elif option_type == "call" and greek_name == "Rho":
        Z = rho_call(S_grid, K0, T_grid, r0, sigma0)
    elif option_type == "put" and greek_name == "Rho":
        Z = rho_put(S_grid, K0, T_grid, r0, sigma0)
    else:
        Z = calc_function(S_grid, K0, T_grid, r0, sigma0)

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S_grid, T_grid, Z, cmap=cmap, edgecolor='none')
    ax.set_xlabel('Precio Subyacente (S)')
    ax.set_ylabel('Tiempo hasta Vencimiento (T)')
    ax.set_zlabel(greek_name)
    ax.set_title(f'{greek_name} Surface')
    st.pyplot(fig)

# Graficamos todas las superficies
plot_surface("Delta", delta_call, "Blues")
plot_surface("Gamma", gamma, "Greens")
plot_surface("Theta", theta_call, "Oranges")
plot_surface("Vega", vega, "Purples")
plot_surface("Rho", rho_call, "Reds")

# --- FOOTER ---
st.markdown("---")
