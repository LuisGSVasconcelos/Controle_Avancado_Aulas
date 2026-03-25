import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from control import tf, pade, feedback, step_response, bode, nyquist, root_locus

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
    # C(s) = Kc * (1 + 1/(Ti*s) + Td*s)
    C = tf([Kc * Td, Kc, Kc / Ti], [1, 0])  # numerador [Kc*Td, Kc, Kc/Ti], denominador [1, 0] -> s

    # Malha aberta e malha fechada
    L = C * G  # função de transferência de malha aberta
    G_cl = feedback(L, 1)  # malha fechada

    # Simulação da resposta ao degrau
    t_sim = np.linspace(0, 50, 1000)
    t_out, y_out = step_response(G_cl, t_sim)

    # Cria figura da resposta ao degrau
    fig_step, ax_step = plt.subplots(figsize=(8, 4))
    ax_step.plot(t_out, y_out, linewidth=2, label="Resposta ao degrau")
    ax_step.axhline(1.0, color='gray', linestyle='--', label="Setpoint (1)")
    ax_step.set_xlabel("Tempo (min)")
    ax_step.set_ylabel("Variável controlada (°C)")
    ax_step.set_title(f"Resposta ao degrau do sistema com PID (θ = {theta} min)")
    ax_step.grid(True, alpha=0.3)
    ax_step.legend(loc='upper right')
    ax_step.set_xlim([0, 50])
    ax_step.set_ylim([0, max(1.5, max(y_out) * 1.1)])
    plt.tight_layout()

    # Diagrama de Bode
    fig_bode, ax_bode = plt.subplots(2, 1, figsize=(8, 6))
    mag, phase, omega = bode(L, plot=False)
    ax_bode[0].semilogx(omega, 20*np.log10(mag), linewidth=2)
    ax_bode[0].set_ylabel("Magnitude (dB)")
    ax_bode[0].grid(True, alpha=0.3)
    ax_bode[1].semilogx(omega, phase * 180/np.pi, linewidth=2)
    ax_bode[1].set_xlabel("Frequência (rad/min)")
    ax_bode[1].set_ylabel("Fase (graus)")
    ax_bode[1].grid(True, alpha=0.3)
    fig_bode.suptitle("Diagrama de Bode (Malha Aberta)")
    plt.tight_layout()

   # Diagrama de Nyquist
    fig_nyquist, ax_nyquist = plt.subplots(figsize=(6, 6))
    try:
        # Tenta obter dados manualmente
        nyquist_data = nyquist(L, plot=False)
        # Verifica se os dados são válidos
        if hasattr(nyquist_data, 'real') and hasattr(nyquist_data, 'imag'):
            real = np.array(nyquist_data.real).flatten()
            imag = np.array(nyquist_data.imag).flatten()
            if len(real) > 0:
                # Plota a curva
                ax_nyquist.plot(real, imag, linewidth=2, label='Nyquist')
                ax_nyquist.plot(real, -imag, '--', alpha=0.5, label='Conjugado')
                # Ajusta limites para incluir o ponto crítico (-1,0)
                all_x = np.concatenate([real, [-1]])
                all_y = np.concatenate([imag, [0], -imag])
                x_min, x_max = np.min(all_x), np.max(all_x)
                y_min, y_max = np.min(all_y), np.max(all_y)
                margin_x = 0.2 * (x_max - x_min) if x_max != x_min else 0.5
                margin_y = 0.2 * (y_max - y_min) if y_max != y_min else 0.5
                ax_nyquist.set_xlim(x_min - margin_x, x_max + margin_x)
                ax_nyquist.set_ylim(y_min - margin_y, y_max + margin_y)
            else:
                st.warning("Dados do Nyquist vazios. Usando plotagem automática.")
                nyquist(L, ax=ax_nyquist)
        else:
            # Se não tiver os atributos esperados, usa plotagem direta
            nyquist(L, ax=ax_nyquist)
    except Exception as e:
        st.warning(f"Erro ao plotar Nyquist: {e}. Usando método alternativo.")
        nyquist(L, ax=ax_nyquist)

    # Adiciona o ponto crítico e linhas de referência
    ax_nyquist.scatter([-1], [0], color='red', marker='o', s=50, label='Ponto crítico (-1,0)')
    ax_nyquist.axhline(0, color='gray', linewidth=0.5)
    ax_nyquist.axvline(0, color='gray', linewidth=0.5)
    ax_nyquist.set_xlabel("Parte Real")
    ax_nyquist.set_ylabel("Parte Imaginária")
    ax_nyquist.set_title("Diagrama de Nyquist (Malha Aberta)")
    ax_nyquist.grid(True, alpha=0.3)
    ax_nyquist.legend()
    plt.tight_layout()

    # Lugar das raízes (corrigido para sistemas com atraso aproximado)
    fig_rlocus, ax_rlocus = plt.subplots(figsize=(6, 6))
    # Para sistemas com atraso aproximado por Padé, o lugar das raízes ainda pode ser calculado.
    # root_locus(L, plot=True, ax=ax_rlocus) é suficiente, mas cuidado: ele plota diretamente.
    # Vamos usar plot=False e fazer um gráfico manual com os ganhos?
    # Para simplicidade, podemos usar root_locus(L, ax=ax_rlocus) que deve funcionar.
    # Se houver problemas, podemos extrair os dados e plotar manualmente.
    try:
        root_locus(L, ax=ax_rlocus)
    except Exception as e:
        st.warning(f"O lugar das raízes não pôde ser calculado para este sistema com atraso aproximado. Erro: {e}")
        ax_rlocus.text(0.5, 0.5, "Lugar das raízes não disponível\npara sistemas com atraso", 
                    horizontalalignment='center', verticalalignment='center', transform=ax_rlocus.transAxes)
    ax_rlocus.set_title("Lugar das Raízes (Malha Fechada)")
    ax_rlocus.grid(True, alpha=0.3)
    plt.tight_layout()
    # Exibindo os gráficos com abas
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Resposta ao Degrau", "📊 Diagrama de Bode", "🌀 Diagrama de Nyquist", "📍 Lugar das Raízes"])
    with tab1:
        st.pyplot(fig_step)
    with tab2:
        st.pyplot(fig_bode)
    with tab3:
        st.pyplot(fig_nyquist)
    with tab4:
        st.pyplot(fig_rlocus)

    # Estatísticas simples (mantido)
    st.subheader("📈 Análise da resposta ao degrau")
    overshoot = (max(y_out) - 1) * 100 if max(y_out) > 1 else 0
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