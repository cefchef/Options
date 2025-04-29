+39
-44
Original file line number	Diff line number	Diff line change
@@ -1,3 +1,4 @@
# --- CONFIGURACIÓN INICIAL ---
import streamlit as st
import numpy as np
@@ -53,28 +54,25 @@ def rho_put(S, K, T, r, sigma):
    return -K * T * np.exp(-r*T) * norm.cdf(-d2(S,K,T,r,sigma))


# --- SIDEBAR: INPUTS + SLIDERS + VALIDACIONES ---
st.sidebar.header("Parámetros de la Opción")

# Manejador de errores: no permitir negativos
def safe_input(label, min_value, max_value, default, step=0.01):
    val = st.sidebar.number_input(label, min_value=min_value, max_value=max_value, value=default, step=step)
    if val < min_value:
        st.sidebar.error(f"No se permiten valores negativos en {label}. Reset a valor inicial.")
        return default
    return val

# Inputs y sliders sincronizados
option_type = st.sidebar.selectbox("Tipo de Opción", ["call", "put"])
S0 = safe_input("Precio Subyacente (S0)", 0.01, 10000.0, 100.0)
K0 = safe_input("Strike Price (K)", 0.01, 10000.0, 100.0)
r0 = safe_input("Tasa libre de riesgo (r)", 0.0, 1.0, 0.05, step=0.001)
sigma0 = safe_input("Volatilidad (σ)", 0.01, 3.0, 0.2, step=0.01)
T0 = safe_input("Tiempo hasta Vencimiento (T)", 0.01, 10.0, 1.0, step=0.05)

# Toggle Dark/Light Mode
dark_mode = st.sidebar.toggle('🌞 / 🌙 Modo Oscuro')
dark_mode = st.sidebar.toggle('🌞 | 🌙')

# --- CÁLCULO DE GREEKS ---
S_range = np.linspace(0.5 * S0, 1.5 * S0, 100)
@@ -90,30 +88,29 @@ def safe_input(label, min_value, max_value, default, step=0.01):

gamma_vals = gamma(S_range, K0, T0, r0, sigma0)
vega_vals = vega(S_range, K0, T0, r0, sigma0)
# Localizar índice del spot
index_S0 = np.argmin(np.abs(S_range - S0))

# --- FUNCIONES AUXILIARES PARA CARDS Y SUPERFICIES ---
def greek_card_with_surface(title, current_val, data_series, surface_function, color, tooltip_text):
    # Título con tooltip ℹ️
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h3 style="display: inline;">{title}</h3>
        <span title="{tooltip_text}" style="font-size: 24px; cursor: help;">ℹ️</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Valor actual
    st.metric(label="Valor actual", value=f"{current_val:.4f}")
    
    # Mini sparkline
# --- FUNCIÓN PARA MOSTRAR GREEKS ---
def greek_card_with_surface(title, current_val, data_series, surface_function, colormap, tooltip_text):
    st.markdown(f"<h2 style='color: {'white' if dark_mode else '#222222'};'>{title} ℹ️</h2>", unsafe_allow_html=True)
    st.caption(tooltip_text)
    # Sparkline
    spark_color_map = {
        "Delta": "#1f77b4",
        "Gamma": "#2ca02c",
        "Theta": "#ff7f0e",
        "Vega": "#9467bd",
        "Rho": "#d62728"
    }
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data_series, mode='lines', line=dict(color=color, width=2)))
    fig.add_trace(go.Scatter(y=data_series, mode='lines', line=dict(color=spark_color_map[title], width=2)))
    fig.update_layout(height=100, margin=dict(l=10, r=10, t=10, b=10), xaxis_visible=False, yaxis_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Gráfico 3D rotable
    # Surface 3D
    S_grid, T_grid = np.meshgrid(
        np.linspace(0.5*S0, 1.5*S0, 40),
        np.linspace(0.01, 2.0, 40)
@@ -133,23 +130,26 @@ def greek_card_with_surface(title, current_val, data_series, surface_function, c
    else:
        Z = surface_function(S_grid, K0, T_grid, r0, sigma0)

    fig3d = plt.figure(figsize=(6,4))
    fig3d = plt.figure(figsize=(6, 4))
    ax = fig3d.add_subplot(111, projection='3d')
    surf = ax.plot_surface(S_grid, T_grid, Z, cmap=color, edgecolor='none')
    ax.set_xlabel('Precio Subyacente (S)')
    ax.set_ylabel('Tiempo hasta Vencimiento (T)')
    ax.plot_surface(S_grid, T_grid, Z, cmap=colormap, edgecolor='none')
    ax.set_xlabel('S')
    ax.set_ylabel('T')
    ax.set_zlabel(title)
    ax.set_title(f'{title} Surface')
    ax.set_title(f"{title} Surface")
    if dark_mode:
        ax.set_facecolor('#111111')
        fig3d.patch.set_facecolor('#111111')
    st.pyplot(fig3d)
        ax.w_xaxis.line.set_color("white")
        ax.w_yaxis.line.set_color("white")
        ax.w_zaxis.line.set_color("white")

    st.pyplot(fig3d)

# --- MOSTRAR TODAS LAS GREEKS ---
st.subheader("Sensibilidades (Greeks)")
# --- MOSTRAR LAS GREEKS ---
st.markdown(f"<h2 style='color: {'white' if dark_mode else '#222222'};'>Sensibilidades (Greeks)</h2>", unsafe_allow_html=True)

# Definiciones de tooltips
tooltips = {
    "Delta": "Sensibilidad del precio de la opción al movimiento del precio del subyacente.",
    "Gamma": "Tasa de cambio de Delta respecto al precio del subyacente.",
@@ -158,27 +158,22 @@ def greek_card_with_surface(title, current_val, data_series, surface_function, c
    "Rho": "Sensibilidad del precio de la opción a cambios en la tasa libre de riesgo."
}

# Distribuir las cards en columnas
col1, col2, col3 = st.columns(3)
col4, col5 = st.columns(2)

with col1:
    greek_card_with_surface("Delta", delta_vals[index_S0], delta_vals, delta_call, "#1f77b4", tooltips["Delta"])
    greek_card_with_surface("Delta", delta_vals[index_S0], delta_vals, delta_call, "Blues", tooltips["Delta"])
with col2:
    greek_card_with_surface("Gamma", gamma_vals[index_S0], gamma_vals, gamma, "#2ca02c", tooltips["Gamma"])
    greek_card_with_surface("Gamma", gamma_vals[index_S0], gamma_vals, gamma, "Greens", tooltips["Gamma"])
with col3:
    greek_card_with_surface("Theta", theta_vals[index_S0], theta_vals, theta_call, "#ff7f0e", tooltips["Theta"])
    greek_card_with_surface("Theta", theta_vals[index_S0], theta_vals, theta_call, "Oranges", tooltips["Theta"])
with col4:
    greek_card_with_surface("Vega", vega_vals[index_S0], vega_vals, vega, "#9467bd", tooltips["Vega"])
    greek_card_with_surface("Vega", vega_vals[index_S0], vega_vals, vega, "Purples", tooltips["Vega"])
with col5:
    greek_card_with_surface("Rho", rho_vals[index_S0], rho_vals, rho_call, "#d62728", tooltips["Rho"])
    
    greek_card_with_surface("Rho", rho_vals[index_S0], rho_vals, rho_call, "Reds", tooltips["Rho"])
# --- VOLATILITY SMILE ---
st.subheader("Volatility Smile")
st.markdown(f"<h2 style='color: {'white' if dark_mode else '#222222'};'>Volatility Smile</h2>", unsafe_allow_html=True)

strikes = np.linspace(0.8*S0, 1.2*S0, 50)
implied_vol = sigma0 + 0.05 * ((strikes - S0)/S0)**2
