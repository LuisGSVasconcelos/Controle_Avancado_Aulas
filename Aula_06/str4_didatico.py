"""
================================================================================
STR — Self-Tuning Regulator (Regulador Auto-Ajustável)
================================================================================

Disciplina: Controle Avançado de Processos — Pós-graduação em Engenharia Química
Base teórica:  Åström & Wittenmark, "Adaptive Control" (2nd ed., Cap. 3-4)
               Seborg, Edgar, Mellichamp & Doyle, "Process Dynamics and Control"
               (4th ed., Cap. 12 — Adaptive Control)

--------------------------------------------------------------------------------
CONCEITO FUNDAMENTAL
--------------------------------------------------------------------------------

Um STR (Self-Tuning Regulator) combina três blocos funcionais:

    1. IDENTIFICAÇÃO ONLINE (RLS)
       Estima os parâmetros de um modelo ARX de primeira ordem:
           y[k] = a·y[k-1] + b·u[k-1]
       O algoritmo RLS ajusta recursivamente θ = [a, b]ᵀ a cada nova amostra.

    2. PROJETO DO CONTROLADOR
       A partir dos parâmetros estimados (a, b), calcula-se o ganho do processo
       e a constante de tempo:
           Kp_est = b / (1 - a)
           τ_est  = -Ts / ln(a)
       A sintonia PI é feita por cancelamento de polo com constante de tempo
       de malha fechada (Tc) desejada:
           Kc = τ_est / (Kp_est · (Tc + θ_dead))
           Ti = τ_est

    3. LEI DE CONTROLE (PI Digital)
           Δu[k] = Kc·(e[k] - e[k-1]) + (Kc/Ti)·e[k]·Ts
           u[k]  = u[k-1] + Δu[k]

    A grande vantagem do STR: quando a planta muda (ex.: degradação do ganho),
    o controlador se re-sintoniza automaticamente. O PI fixo, sem esta
    capacidade, deteriora seu desempenho.

--------------------------------------------------------------------------------
CENÁRIO DA SIMULAÇÃO
--------------------------------------------------------------------------------

    • Processo de primeira ordem com ganho variante no tempo:
        - Kp = 2,0          para t < 30 min  (condição nominal)
        - Kp cai para 0,8   entre 30 e 50 min (degradação)
        - Kp = 0,8          para t > 50 min  (condição degradada)
    • Constante de tempo: τ = 5,0 min (fixa)
    • Setpoint: 1,0 → 1,5 (t=20) → 1,25 (t=60)

    Duas malhas idênticas em paralelo:
        Malha A — Controlada por STR (adapta seus parâmetros)
        Malha B — Controlada por PI clássico com sintonia fixa
                  (sintonizado para Kp=2,0, τ=5,0)
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. PARÂMETROS DO PROCESSO REAL (PLANTA)
# ============================================================================

T_amostragem = 0.5                  # Ts — período de amostragem (min)
tau_real = 5.0                      # τ — constante de tempo da planta (min)
a_planta_base = np.exp(-T_amostragem / tau_real)  # ≈ 0,9048

print("=" * 60)
print("STR — REGULADOR AUTO-AJUSTÁVEL")
print(f"Período de amostragem: Ts = {T_amostragem} min")
print(f"Constante de tempo:   τ = {tau_real} min")
print(f"Polo discreto:        a = exp(-Ts/τ) ≈ {a_planta_base:.4f}")
print("=" * 60)

def ganho_planta_real(t):
    """
    Ganho do processo real (Kp) em função do tempo.
    Simula uma degradação gradual: o ganho cai de 2,0 para 0,8.
    """
    if t < 30:
        return 2.0
    elif t < 50:
        # Degradação linear entre 30 e 50 minutos
        fracao = (t - 30) / 20
        return 2.0 - fracao * 1.2
    else:
        return 0.8

def obter_parametros_planta(t):
    """Retorna (a, b) do modelo discreto y[k] = a·y[k-1] + b·u[k-1]."""
    Kp = ganho_planta_real(t)
    a = a_planta_base
    b = Kp * (1 - a_planta_base)       # b = Kp · (1 - a)
    return a, b

# ============================================================================
# 2. CONFIGURAÇÃO DOS CONTROLADORES
# ============================================================================

# ------------------------------------------------------------------
# 2.1 STR (Self-Tuning Regulator) — Adaptativo
# ------------------------------------------------------------------

fator_esquecimento = 0.98          # λ — pondera dados mais recentes
theta_estimado = np.zeros((2, 1))  # θ = [a, b]ᵀ  (parâmetros estimados)
matriz_covariancia = np.eye(2) * 1000.0  # P — alta incerteza inicial

# Parâmetros iniciais do PI adaptativo
Kc_adapt = 1.0
Ti_adapt = 5.0
u_adapt_prev = 0.0
erro_adapt_prev = 0.0

# ------------------------------------------------------------------
# 2.2 PI Clássico (Sintonia Fixa) — Referência
# ------------------------------------------------------------------

# Sintonizado para condição nominal (Kp = 2,0, τ = 5,0)
# Cancelamento de polo: Kc = τ/(Kp · Tc), Ti = τ
Kc_fixo = 1.0
Ti_fixo = 5.0

u_fixo_prev = 0.0
erro_fixo_prev = 0.0

print("\n[CONFIGURAÇÃO]")
print(f"  Fator de esquecimento (λ): {fator_esquecimento}")
print(f"  Covariância inicial (P₀):  1000·I")
print(f"  Kc do PI Fixo:             {Kc_fixo}")
print(f"  Ti do PI Fixo:             {Ti_fixo} min")

# ============================================================================
# 3. PREPARAÇÃO DA SIMULAÇÃO
# ============================================================================

t_final = 80                        # Duração da simulação (min)
n_passos = int(t_final / T_amostragem)
tempo = np.linspace(0, t_final, n_passos)

# -- Variáveis do STR --
y_adapt = np.zeros(n_passos)        # Saída da planta (STR)
u_adapt = np.zeros(n_passos)        # Sinal de controle (STR)
Kc_arr = np.zeros(n_passos)         # Histórico de Kc adaptativo
Ti_arr = np.zeros(n_passos)         # Histórico de Ti adaptativo
a_est_arr = np.zeros(n_passos)      # Histórico de a estimado
b_est_arr = np.zeros(n_passos)      # Histórico de b estimado
erro_pred_arr = np.zeros(n_passos)  # Erro de predição do modelo
traco_P_arr = np.zeros(n_passos)    # Traço da matriz de covariância

# -- Variáveis do PI Fixo --
y_fixo = np.zeros(n_passos)
u_fixo = np.zeros(n_passos)

# -- Setpoint (dinâmico) --
setpoint = np.ones(n_passos) * 1.0
idx_20 = int(20 / T_amostragem)
idx_40 = int(40 / T_amostragem)
idx_60 = int(60 / T_amostragem)
setpoint[idx_20:idx_40] = 1.5
setpoint[idx_60:] = 1.25

# -- Condições iniciais --
y_adapt[0] = 0.0
u_adapt[0] = 0.5
y_fixo[0] = 0.0
u_fixo[0] = 0.5

# -- Flag para período de graça (identificação sem atuação) --
periodo_graca = 5.0   # min
idx_graca = int(periodo_graca / T_amostragem)

print(f"\n[SIMULAÇÃO]")
print(f"  Duração: {t_final} min  ({n_passos} passos)")
print(f"  Período de graça do STR: {periodo_graca} min (apenas identificação)")
print(f"  Setpoint: 1,0 → 1,5 (t=20 min) → 1,25 (t=60 min)")
print(f"  Degradação da planta: t=30 a 50 min (Kp: 2,0 → 0,8)")
print(f"  Iniciando loop de simulação...")

# ============================================================================
# 4. LOOP PRINCIPAL DE CONTROLE
# ============================================================================

for k in range(1, n_passos):

    # ------------------------------------------------------------------
    # 4.1 O PROCESSO FÍSICO — Ambas as malhas compartilham a mesma planta
    # ------------------------------------------------------------------
    #
    # Modelo discreto de primeira ordem:
    #     y[k] = a · y[k-1] + b · u[k-1]
    #
    # Onde:
    #     a = exp(-Ts / τ)      (pólo discreto da planta)
    #     b = Kp · (1 - a)      (ganho estacionário em regime)
    #
    a_planta, b_planta = obter_parametros_planta(tempo[k])

    # Malha do STR
    y_adapt[k] = a_planta * y_adapt[k-1] + b_planta * u_adapt[k-1]
    erro_adapt = setpoint[k] - y_adapt[k]

    # Malha do PI Fixo (mesma planta)
    y_fixo[k] = a_planta * y_fixo[k-1] + b_planta * u_fixo[k-1]
    erro_fixo = setpoint[k] - y_fixo[k]

    # ------------------------------------------------------------------
    # 4.2 IDENTIFICAÇÃO ONLINE — RLS com fator de esquecimento
    #     (Apenas para o STR)
    # ------------------------------------------------------------------
    #
    # O RLS estima os parâmetros θ = [a, b]ᵀ do modelo ARX:
    #
    #     ŷ[k] = φᵀ[k] · θ[k-1]      → predição
    #     ε[k] = y[k] - ŷ[k]          → erro de predição (inovação)
    #
    #     K[k] = P[k-1]·φ / (λ + φᵀ·P[k-1]·φ)   → ganho de Kalman
    #
    #     θ[k] = θ[k-1] + K[k]·ε[k]             → atualização de θ
    #
    #     P[k] = (P[k-1] - K[k]·φᵀ·P[k-1]) / λ  → atualização da covariância
    #
    # O fator de esquecimento λ ∈ (0, 1] controla a memória do estimador:
    #   λ = 1   → memória infinita (dados antigos têm mesmo peso)
    #   λ < 1   → dados recentes têm mais peso (planta variante no tempo)
    #
    vetor_regressao = np.array([[y_adapt[k-1]], [u_adapt[k-1]]])

    # Predição do modelo
    y_predito = (vetor_regressao.T @ theta_estimado).item()
    erro_predicao = y_adapt[k] - y_predito

    # Ganho de atualização (ganho de Kalman)
    denominador = fator_esquecimento + (vetor_regressao.T @ matriz_covariancia @ vetor_regressao).item()
    ganho_kalman = (matriz_covariancia @ vetor_regressao) / denominador

    # Atualização dos parâmetros e da covariância
    theta_estimado = theta_estimado + ganho_kalman * erro_predicao
    matriz_covariancia = (matriz_covariancia - ganho_kalman @ vetor_regressao.T @ matriz_covariancia) / fator_esquecimento

    a_est = theta_estimado[0, 0]
    b_est = theta_estimado[1, 0]

    # Armazenar históricos
    erro_pred_arr[k] = erro_predicao
    traco_P_arr[k] = np.trace(matriz_covariancia)
    a_est_arr[k] = a_est
    b_est_arr[k] = b_est

    # ------------------------------------------------------------------
    # 4.3 PROJETO DO CONTROLADOR ADAPTATIVO (STR)
    # ------------------------------------------------------------------
    #
    # A partir dos parâmetros estimados (a_est, b_est), obtém-se:
    #
    #     Kp_est = b_est / (1 - a_est)     → ganho estimado do processo
    #     τ_est  = -Ts / ln(a_est)         → constante de tempo estimada
    #
    # A sintonia PI por cancelamento de polo:
    #
    #     Kc = τ_est / (Kp_est · (Tc + θ_d))
    #     Ti = τ_est
    #
    # Onde:
    #     Tc     → constante de tempo desejada em malha fechada (min)
    #     θ_d    → atraso de transporte (min)
    #
    if tempo[k] > periodo_graca and 0.0 < a_est < 0.99 and b_est > 0.001:
        Kp_est = b_est / (1 - a_est)
        tau_est = -T_amostragem / np.log(a_est)

        Tc_desejado = 2.0
        theta_atraso = T_amostragem

        # Cancelamento de polo
        Kc_adapt = tau_est / (Kp_est * (Tc_desejado + theta_atraso))
        Ti_adapt = tau_est

        # TRAVAS DE SEGURANÇA — evitam valores extremos que desestabilizariam
        Kc_adapt = np.clip(Kc_adapt, 0.1, 5.0)
        Ti_adapt = np.clip(Ti_adapt, 1.0, 6.0)

    Kc_arr[k] = Kc_adapt
    Ti_arr[k] = Ti_adapt

    # ------------------------------------------------------------------
    # 4.4 APLICAÇÃO DAS LEIS DE CONTROLE
    # ------------------------------------------------------------------

    # 4.4.1 STR (Adaptativo)
    du_adapt = (Kc_adapt * (erro_adapt - erro_adapt_prev) +
                (Kc_adapt / Ti_adapt) * erro_adapt * T_amostragem)
    u_adapt[k] = u_adapt[k-1] + du_adapt
    u_adapt[k] = np.clip(u_adapt[k], 0, 5.0)

    # 4.4.2 PI Fixo (sintonia constante)
    du_fixo = (Kc_fixo * (erro_fixo - erro_fixo_prev) +
               (Kc_fixo / Ti_fixo) * erro_fixo * T_amostragem)
    u_fixo[k] = u_fixo[k-1] + du_fixo
    u_fixo[k] = np.clip(u_fixo[k], 0, 5.0)

    # -- Atualizar memórias para a próxima iteração --
    erro_adapt_prev = erro_adapt
    erro_fixo_prev = erro_fixo

    # ------------------------------------------------------------------
    # 4.5 PRINTS DIDÁTICOS — mostram eventos importantes ao usuário
    # ------------------------------------------------------------------
    if abs(tempo[k] - 20.0) < T_amostragem / 2:
        print(f"\n  [t={tempo[k]:.1f} min] Setpoint alterado: 1,0 → 1,5")

    if abs(tempo[k] - 30.0) < T_amostragem / 2:
        print(f"\n  [t={tempo[k]:.1f} min] Planta começou a DEGRADAR (Kp: 2,0 → 0,8)")

    if abs(tempo[k] - periodo_graca) < T_amostragem / 2:
        print(f"\n  [t={tempo[k]:.1f} min] Fim do período de graça do STR")
        print(f"      Parâmetros estimados: a ≈ {a_est:.4f}, b ≈ {b_est:.4f}")
        print(f"      Kp estimado:          {b_est/(1-a_est):.4f}")
        print(f"      τ estimado:           {-T_amostragem/np.log(a_est):.2f} min")

    if abs(tempo[k] - 55.0) < T_amostragem / 2:
        Kp_est_atual = b_est / (1 - a_est) if (1 - a_est) > 0.001 else 0
        print(f"\n  [t={tempo[k]:.1f} min] Regime degradado estabelecido")
        print(f"      Kp real: {ganho_planta_real(tempo[k]):.3f} | Kp estimado: {Kp_est_atual:.3f}")
        print(f"      Kc do STR: {Kc_arr[k]:.3f} | Kc do PI Fixo: {Kc_fixo:.3f}")
        print(f"      Ti do STR: {Ti_arr[k]:.2f} min | Ti do PI Fixo: {Ti_fixo:.2f} min")

# ============================================================================
# 5. MÉTRICAS DE DESEMPENHO
# ============================================================================
#
# IAE — Integral of Absolute Error
# Calculado a partir do momento em que a planta já está degradada (t ≥ 50 min)
# para evidenciar a diferença entre STR e PI fixo.

idx_falha = int(50 / T_amostragem)

erro_str_falha = setpoint[idx_falha:] - y_adapt[idx_falha:]
erro_fixo_falha = setpoint[idx_falha:] - y_fixo[idx_falha:]

iae_str = np.sum(np.abs(erro_str_falha)) * T_amostragem
iae_fixo = np.sum(np.abs(erro_fixo_falha)) * T_amostragem

# IAE acumulado ao longo do tempo (para gráfico)
iae_str_acum = np.zeros(n_passos)
iae_fixo_acum = np.zeros(n_passos)
for k in range(1, n_passos):
    iae_str_acum[k] = iae_str_acum[k-1] + abs(setpoint[k] - y_adapt[k]) * T_amostragem
    iae_fixo_acum[k] = iae_fixo_acum[k-1] + abs(setpoint[k] - y_fixo[k]) * T_amostragem

print(f"\n{'='*60}")
print(f"RESUMO DE DESEMPENHO")
print(f"{'='*60}")
print(f"  IAE do STR (após t=50 min):     {iae_str:.4f}")
print(f"  IAE do PI Fixo (após t=50 min): {iae_fixo:.4f}")
print(f"  Melhoria relativa do STR:       {(1 - iae_str/iae_fixo)*100:.1f}%")
print(f"{'='*60}")

# ============================================================================
# 6. GRÁFICOS
# ============================================================================
#
# Organizados em 3 figuras para melhor visualização dos conceitos.

# ------------------------------------------------------------------
# FIGURA 1 — Comparação STR vs PI Fixo
# ------------------------------------------------------------------

plt.figure("1. STR vs PI Fixo — Variáveis de Processo", figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(tempo, setpoint, 'r--', label='Setpoint (SP)', linewidth=2)
plt.plot(tempo, y_adapt, 'b-', label=f'STR (Adaptativo) | IAE(t>50)={iae_str:.2f}', linewidth=2)
plt.plot(tempo, y_fixo, 'gray', linestyle='-.', label=f'PI Fixo | IAE(t>50)={iae_fixo:.2f}', linewidth=2)
plt.axvline(x=30, color='orange', linestyle=':', alpha=0.7, label='Início da degradação')
plt.axvline(x=50, color='red', linestyle=':', alpha=0.7, label='Fim da degradação')
plt.ylabel('Variável de Processo (PV)')
plt.title('STR vs PI Clássico — Resposta ao Setpoint com Degradação da Planta')
plt.grid(True, linestyle=':')
plt.legend(loc='lower right')
plt.xlim(0, t_final)

plt.subplot(3, 1, 2)
plt.plot(tempo, u_adapt, 'darkorange', label='Sinal de Controle — STR')
plt.plot(tempo, u_fixo, 'gray', linestyle='-.', alpha=0.7, label='Sinal de Controle — PI Fixo')
plt.axvline(x=30, color='orange', linestyle=':', alpha=0.5)
plt.axvline(x=50, color='red', linestyle=':', alpha=0.5)
plt.ylabel('Sinal do Atuador (u)')
plt.title('Esforço de Controle')
plt.grid(True, linestyle=':')
plt.legend(loc='upper right')
plt.xlim(0, t_final)

plt.subplot(3, 1, 3)
plt.plot(tempo, iae_str_acum, 'b-', label='IAE acumulado — STR', linewidth=2)
plt.plot(tempo, iae_fixo_acum, 'gray', linestyle='-.', label='IAE acumulado — PI Fixo', linewidth=2)
plt.axvline(x=30, color='orange', linestyle=':', alpha=0.5)
plt.axvline(x=50, color='red', linestyle=':', alpha=0.5)
plt.xlabel('Tempo (min)')
plt.ylabel('IAE acumulado')
plt.title('Erro Acumulado ao Longo do Tempo')
plt.grid(True, linestyle=':')
plt.legend(loc='upper left')
plt.xlim(0, t_final)

plt.tight_layout()

# ------------------------------------------------------------------
# FIGURA 2 — Evolução do STR (sintonia e identificação)
# ------------------------------------------------------------------

plt.figure("2. STR — Evolução da Sintonia e Identificação", figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(tempo, Kc_arr, 'g-', label='Kc do STR (ajustado dinamicamente)', linewidth=2)
plt.plot(tempo, Ti_arr, 'purple', label='Ti do STR', linewidth=2)
plt.axhline(Kc_fixo, color='gray', linestyle='-.', alpha=0.7, label=f'Kc do PI Fixo = {Kc_fixo}')
plt.axhline(Ti_fixo, color='gray', linestyle=':', alpha=0.5, label=f'Ti do PI Fixo = {Ti_fixo} min')
plt.axvline(x=30, color='orange', linestyle=':', alpha=0.5)
plt.axvline(x=50, color='red', linestyle=':', alpha=0.5)
plt.ylabel('Sintonia do PI')
plt.title('Evolução dos Parâmetros do Controlador STR')
plt.grid(True, linestyle=':')
plt.legend(loc='upper right')
plt.xlim(0, t_final)

plt.subplot(3, 1, 2)
Kp_real_arr = np.array([ganho_planta_real(ti) for ti in tempo])
b_arr = Kp_real_arr * (1 - a_planta_base)
plt.plot(tempo, Kp_real_arr, 'k--', label='Ganho Kp (Real da Planta)', linewidth=2)
Kp_estimado_arr = np.where(1 - a_est_arr > 0.001, b_est_arr / (1 - a_est_arr), np.nan)
plt.plot(tempo, Kp_estimado_arr, 'teal', label='Ganho Kp (Estimado pelo RLS)', linewidth=1.5)
plt.axvline(x=30, color='orange', linestyle=':', alpha=0.5)
plt.axvline(x=50, color='red', linestyle=':', alpha=0.5)
plt.ylabel('Ganho (Kp)')
plt.title('Identificação do Processo — RLS')
plt.grid(True, linestyle=':')
plt.legend(loc='upper right')
plt.xlim(0, t_final)

plt.subplot(3, 1, 3)
plt.semilogy(tempo, traco_P_arr, 'b-', label='tr(P) — Traço da Matriz de Covariância', linewidth=1.5)
plt.axvline(x=periodo_graca, color='green', linestyle='--', alpha=0.7, label=f'Fim do período de graça (t={periodo_graca} min)')
plt.axvline(x=30, color='orange', linestyle=':', alpha=0.5)
plt.axvline(x=50, color='red', linestyle=':', alpha=0.5)
plt.xlabel('Tempo (min)')
plt.ylabel('tr(P) [escala log]')
plt.title('Incerteza do Estimador — Quanto menor tr(P), maior a confiança nos parâmetros')
plt.grid(True, linestyle=':')
plt.legend(loc='upper right')
plt.xlim(0, t_final)

plt.tight_layout()

# ------------------------------------------------------------------
# FIGURA 3 — Erro de predição e detalhes do modelo
# ------------------------------------------------------------------

plt.figure("3. STR — Erro de Predição e Detalhes", figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(tempo, erro_pred_arr, 'b-', label='Erro de Predição ε = y − ŷ', linewidth=1)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.axvline(x=periodo_graca, color='green', linestyle='--', alpha=0.7)
plt.axvline(x=30, color='orange', linestyle=':', alpha=0.5)
plt.axvline(x=50, color='red', linestyle=':', alpha=0.5)
plt.ylabel('Erro de Predição')
plt.title('Qualidade do Modelo RLS — O erro indica o quanto o modelo se afasta da planta real')
plt.grid(True, linestyle=':')
plt.legend(loc='upper right')
plt.xlim(0, t_final)

plt.subplot(2, 1, 2)
plt.plot(tempo, a_est_arr, 'b-', label='a (polo estimado)', linewidth=1.5)
plt.plot(tempo, b_est_arr, 'orange', label='b (ganho estimado)', linewidth=1.5)
plt.axhline(a_planta_base, color='blue', linestyle='--', alpha=0.5, label=f'a real = {a_planta_base:.4f}')
plt.axvline(x=periodo_graca, color='green', linestyle='--', alpha=0.7)
plt.axvline(x=30, color='orange', linestyle=':', alpha=0.5)
plt.axvline(x=50, color='red', linestyle=':', alpha=0.5)
plt.xlabel('Tempo (min)')
plt.ylabel('Parâmetros θ = [a, b]ᵀ')
plt.title('Convergência dos Parâmetros Estimados pelo RLS')
plt.grid(True, linestyle=':')
plt.legend(loc='upper right')
plt.xlim(0, t_final)

plt.tight_layout()

# ============================================================================
# 7. EXIBIÇÃO DOS GRÁFICOS
# ============================================================================

plt.show()

# ============================================================================
# 8. REFERÊNCIAS
# ============================================================================
print(f"""
{'='*60}
REFERÊNCIAS BIBLIOGRÁFICAS
{'='*60}
  1. Åström, K. J. & Wittenmark, B. (2008).
     Adaptive Control (2nd ed.). Dover Publications.
     Cap. 3: "Real-Time Parameter Estimation"
     Cap. 4: "Adaptive Control of Linear Systems"

  2. Seborg, D. E., Edgar, T. F., Mellichamp, D. A., & Doyle III, F. J. (2016).
     Process Dynamics and Control (4th ed.). Wiley.
     Cap. 12: "Adaptive Control"

  3. Ljung, L. (1999).
     System Identification: Theory for the User (2nd ed.). Prentice Hall.
     Cap. 11: "Recursive Estimation Methods"
""")
