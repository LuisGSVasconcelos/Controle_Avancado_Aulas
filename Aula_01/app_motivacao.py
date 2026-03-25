import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from control import tf, pade, feedback, step_response

# Configuração da página
st.set_page_config(page_title="Simulação PID com Atraso", layout="wide")
st.title("📊 Simulação de Controle PID em Processo com Atraso (FOPDT)")
st.markdown("""
Esta aplicação demonstra as **limitações do controlador PID** quando aplicado a processos com atraso de transporte (*dead time*).  
Utiliza um modelo FOPDT (primeira ordem com atraso) representando, por exemplo, um reator CSTR ou trocador de calor.  
Você pode alterar os parâmetros do processo e do PID, ou usar a **sintonia automática de Ziegler‑Nichols**.
Autor: Luis Vasconcelos em 24/03/2026                
""")

# --- Sidebar para parâmetros do processo ---
st.sidebar.header("🏭 Parâmetros do Processo (FOPDT)")
Kp = st.sidebar.number_input("Ganho estático (Kp)", value=2.0, step=0.5)
tau = st.sidebar.number_input("Constante de tempo (τ) [min]", value=5.0, step=1.0)
theta = st.sidebar.number_input("Atraso de transporte (θ) [min]", value=2.0, step=0.5)

# --- Sidebar para parâmetros do controlador ---
st.sidebar.header("🎛️ Parâmetros do PID")
tune_method = st.sidebar.radio("Método de sintonia", ["Manual", "Ziegler-Nichols (malha aberta)"])

if tune_method == "Manual":
    Kc = st.sidebar.number_input("Ganho proporcional (Kc)", value=1.2, step=0.1)
    Ti = st.sidebar.number_input("Tempo integral (Ti) [min]", value=4.0, step=0.5)
    Td = st.sidebar.number_input("Tempo derivativo (Td) [min]", value=1.0, step=0.2)
else:
    # Calcula automaticamente com base nos parâmetros do processo
    if theta > 0:
        Kc_zn = (1.2 / Kp) * (tau / theta)
        Ti_zn = 2.0 * theta
        Td_zn = 0.5 * theta
    else:
        Kc_zn = 1.0
        Ti_zn = 1.0
        Td_zn = 0.0
    st.sidebar.write(f"**Kc calculado:** {Kc_zn:.3f}")
    st.sidebar.write(f"**Ti calculado:** {Ti_zn:.3f} min")
    st.sidebar.write(f"**Td calculado:** {Td_zn:.3f} min")
    Kc, Ti, Td = Kc_zn, Ti_zn, Td_zn

# Botão para simular
simulate = st.sidebar.button("▶️ Simular")

# --- Área principal ---
if simulate or "first_run" not in st.session_state:
    st.session_state.first_run = True
    # Construção do modelo
    num = [Kp]
    den = [tau, 1]
    G_s = tf(num, den)  # parte sem atraso

    # Aproximação de Padé de 1ª ordem para o atraso
    num_pade, den_pade = pade(theta, n=1)
    G_delay = tf(num_pade, den_pade)

    # Processo completo
    G = G_s * G_delay

    # Controlador PID (forma paralela ideal)
    # C(s) = Kc * (1 + 1/(Ti*s) + Td*s)
    C = tf([Kc * Td, Kc, Kc / Ti], [1, 0])  # numerador [Kc*Td, Kc, Kc/Ti], denominador [1, 0] -> s

    # Malha fechada
    G_cl = feedback(C * G, 1)

    # Simulação da resposta ao degrau
    t_sim = np.linspace(0, 50, 1000)
    t_out, y_out = step_response(G_cl, t_sim)

    # Cria figura matplotlib com tamanho reduzido
    fig, ax = plt.subplots(figsize=(8, 4))  # ANTES era (10,5) → agora menor
    ax.plot(t_out, y_out, linewidth=2, label="Resposta ao degrau")
    ax.axhline(1.0, color='gray', linestyle='--', label="Setpoint (1)")
    ax.set_xlabel("Tempo (min)")
    ax.set_ylabel("Variável controlada (°C)")
    ax.set_title(f"Resposta ao degrau do sistema com PID (θ = {theta} min)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_xlim([0, 50])
    ax.set_ylim([0, max(1.5, max(y_out) * 1.1)])
    plt.tight_layout()  # Ajusta margens automaticamente

    # Exibir no Streamlit
    st.pyplot(fig)
    plt.close(fig)  # Libera memória

    # Estatísticas simples
    st.subheader("📈 Análise da resposta")
    overshoot = (max(y_out) - 1) * 100 if max(y_out) > 1 else 0
    # Encontrar tempo de acomodação (dentro de ±5%)
    idx_settling = np.where(np.abs(y_out - 1) <= 0.05)[0]
    settling_time = t_out[idx_settling[0]] if len(idx_settling) > 0 else np.nan

    col1, col2 = st.columns(2)
    col1.metric("Overshoot", f"{overshoot:.1f}%")
    col2.metric("Tempo de acomodação (5%)", f"{settling_time:.2f} min" if not np.isnan(settling_time) else "Não estabilizou")

    # Exibir parâmetros utilizados
    with st.expander("🔧 Parâmetros utilizados"):
        st.write(f"**Processo:** Kp = {Kp}, τ = {tau} min, θ = {theta} min")
        st.write(f"**Controlador:** Kc = {Kc:.3f}, Ti = {Ti:.3f} min, Td = {Td:.3f} min")
        if tune_method == "Ziegler-Nichols (malha aberta)":
            st.info("Sintonia Ziegler‑Nichols (malha aberta) aplicada. Esta sintonia costuma gerar overshoot elevado, especialmente com atraso grande.")

    # Advertência se atraso for grande
    if theta / tau > 1:
        st.warning("⚠️ O atraso de transporte é maior que a constante de tempo. O PID terá dificuldade em manter a estabilidade e o desempenho.")

else:
    st.info("Clique em 'Simular' para visualizar a resposta do sistema.")