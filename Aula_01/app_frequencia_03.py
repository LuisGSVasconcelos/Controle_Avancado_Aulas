import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from control import tf, pade, feedback, step_response, bode_plot, nyquist_plot, root_locus, margin

# Configuração da página
st.set_page_config(page_title="Simulação PID com Atraso", layout="wide")
st.title("📊 Simulação de Controle PID em Processo com Atraso (FOPDT)")
st.markdown("""
Esta aplicação demonstra as **limitações do controlador PID** quando aplicado a processos com atraso de transporte (*dead time*).  
Utiliza um modelo FOPDT (primeira ordem com atraso) representando, por exemplo, um reator CSTR ou trocador de calor.  
Você pode alterar os parâmetros do processo e do PID, ou usar a **sintonia automática de Ziegler‑Nichols**.
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
    C = tf([Kc * Td, Kc, Kc / Ti], [1, 0])

    # Malha aberta e malha fechada
    L = C * G
    G_cl = feedback(L, 1)

    # ========================= RESPOSTA AO DEGRAU =========================
    t_sim = np.linspace(0, 50, 1000)
    t_out, y_out = step_response(G_cl, t_sim)

    fig_step, ax_step = plt.subplots(figsize=(8, 4))
    ax_step.plot(t_out, y_out, linewidth=2, label="Resposta ao degrau")
    ax_step.axhline(1.0, color='gray', linestyle='--', label="Setpoint (1)")
    ax_step.set_xlabel("Tempo (min)")
    ax_step.set_ylabel("Variável controlada (°C)")
    ax_step.set_title(f"Resposta ao degrau do sistema com PID (θ = {theta} min)")
    ax_step.grid(True, alpha=0.3)
    ax_step.legend(loc='upper right')
    ax_step.set_xlim([0, 50])
    ax_step.set_ylim([min(-1.0, min(y_out) * 0.9), max(1.5, max(y_out) * 1.1)])
    plt.tight_layout()

    # ========================= DIAGRAMA DE BODE =========================
    fig_bode, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6))
    bode_plot(L, dB=True, ax=[ax_mag, ax_phase])
    ax_mag.set_title("Diagrama de Bode (Malha Aberta)")
    ax_mag.grid(True, alpha=0.3)
    ax_phase.grid(True, alpha=0.3)
    plt.tight_layout()

    # ========================= DIAGRAMA DE NYQUIST =========================
    fig_nyquist, ax_nyquist = plt.subplots(figsize=(6, 6))
    nyquist_plot(L, ax=ax_nyquist)
    ax_nyquist.scatter([-1], [0], color='red', marker='o', s=50, label='Ponto crítico (-1, 0)')
    ax_nyquist.axhline(0, color='gray', linewidth=0.5)
    ax_nyquist.axvline(0, color='gray', linewidth=0.5)
    ax_nyquist.set_title("Diagrama de Nyquist (Malha Aberta)")
    ax_nyquist.grid(True, alpha=0.3)
    ax_nyquist.legend()
    plt.tight_layout()

    # ========================= LUGAR DAS RAÍZES =========================
    fig_rlocus, ax_rlocus = plt.subplots(figsize=(6, 6))
    try:
        root_locus(L, ax=ax_rlocus)
        ax_rlocus.set_title("Lugar das Raízes (Malha Fechada)")
    except Exception as e:
        st.warning(f"O lugar das raízes não pôde ser calculado: {e}")
        ax_rlocus.text(0.5, 0.5, "Lugar das raízes não disponível\n(devido à aproximação de Padé)", 
                       ha='center', va='center', transform=ax_rlocus.transAxes)
    ax_rlocus.grid(True, alpha=0.3)
    plt.tight_layout()

    # ========================= MARGENS DE GANHO E FASE =========================
    try:
        gm, pm, wgm, wpm = margin(L)
        # Converte margem de ganho para dB (se finita e positiva)
        if np.isfinite(gm) and gm > 0:
            gm_db = 20 * np.log10(gm)
        else:
            gm_db = np.inf if gm == np.inf else -np.inf
        
        # Exibe os valores
        st.subheader("📊 Análise de Estabilidade (Malha Aberta)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Margem de Ganho", f"{gm_db:.2f} dB" if np.isfinite(gm_db) else ("∞" if gm_db == np.inf else "Instável"))
        col2.metric("Margem de Fase", f"{pm:.2f}°" if np.isfinite(pm) else "∞")
        col3.metric("Frequência c/ Ganho 0 dB", f"{wpm:.3f} rad/min" if np.isfinite(wpm) else "-")
        col4.metric("Frequência c/ Fase -180°", f"{wgm:.3f} rad/min" if np.isfinite(wgm) else "-")
        
        # Aviso se as margens são baixas
        if np.isfinite(gm_db) and gm_db < 6:
            st.warning("⚠️ Margem de ganho baixa (< 6 dB) – sistema pode oscilar ou ser sensível a incertezas.")
        if np.isfinite(pm) and pm < 30:
            st.warning("⚠️ Margem de fase baixa (< 30°) – sistema pode ter overshoot excessivo ou instabilidade.")
    except Exception as e:
        st.error(f"Não foi possível calcular as margens: {e}")

    # ========================= EXIBIÇÃO COM ABAS =========================
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Resposta ao Degrau", "📊 Diagrama de Bode", "🌀 Diagrama de Nyquist", "📍 Lugar das Raízes"])
    with tab1:
        st.pyplot(fig_step)
    with tab2:
        st.pyplot(fig_bode)
    with tab3:
        st.pyplot(fig_nyquist)
    with tab4:
        st.pyplot(fig_rlocus)

    # ========================= ESTATÍSTICAS DA RESPOSTA =========================
    st.subheader("📈 Análise da resposta ao degrau")
    overshoot = (max(y_out) - 1) * 100 if max(y_out) > 1 else 0
    idx_settling = np.where(np.abs(y_out - 1) <= 0.05)[0]
    settling_time = t_out[idx_settling[0]] if len(idx_settling) > 0 else np.nan
    col1, col2 = st.columns(2)
    col1.metric("Overshoot", f"{overshoot:.1f}%")
    col2.metric("Tempo de acomodação (5%)", f"{settling_time:.2f} min" if not np.isnan(settling_time) else "Não estabilizou")

    # ========================= PARÂMETROS UTILIZADOS =========================
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