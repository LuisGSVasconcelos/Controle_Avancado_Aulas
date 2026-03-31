import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# 1. SIMULAÇÃO DO PROCESSO REAL (FOPDT)
# ==========================================
# Parâmetros do processo verdadeiro (desconhecidos pelo identificador)
Kp_true = 2.0        # ganho estático (°C / %)
tau_true = 5.0       # constante de tempo (min)
theta_true = 2.0     # atraso (min)
Ts = 0.5             # período de amostragem (min)

# Aplicar um degrau na entrada após estabilização
u0 = 20.0            # valor inicial da entrada (%)
u_step = 10.0        # amplitude do degrau (%)
u_final = u0 + u_step

# Simular resposta ao degrau com atraso
# Usamos um modelo exato para gerar os dados "reais"
t = np.arange(0, 50, Ts)   # tempo total de simulação
y = np.zeros_like(t)

# Para um sistema FOPDT com atraso, a resposta ao degrau é:
# y(t) = y0 + Kp * u_step * (1 - exp(-(t-theta)/tau))   para t >= theta
y0 = 0.0   # valor inicial da saída
for i, ti in enumerate(t):
    if ti < theta_true:
        y[i] = y0
    else:
        y[i] = y0 + Kp_true * u_step * (1 - np.exp(-(ti - theta_true) / tau_true))

# Adicionar ruído de medição (simular erro experimental)
np.random.seed(42)
noise = np.random.normal(0, 0.05 * Kp_true * u_step, size=len(t))
y_meas = y + noise

# ==========================================
# 2. IDENTIFICAÇÃO DO MODELO FOPDT
# ==========================================
# Função modelo para ajuste
def fopdt_model(t, K, tau, theta):
    """Modelo FOPDT para ajuste: y(t) = y0 + K * u_step * (1 - exp(-(t-theta)/tau))"""
    y_model = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < theta:
            y_model[i] = y0
        else:
            y_model[i] = y0 + K * u_step * (1 - np.exp(-(ti - theta) / tau))
    return y_model

# Estimativa inicial aproximada
K_est_initial = (y_meas[-1] - y0) / u_step   # ganho estático
# Encontrar índice onde a saída ultrapassa 5% do degrau para estimar theta
threshold = y0 + 0.05 * K_est_initial * u_step
idx = np.where(y_meas > threshold)[0]
if len(idx) > 0:
    theta_est_initial = t[idx[0]]
else:
    theta_est_initial = 1.0
# Estimar tau como o tempo para atingir 63.2% do degrau, menos theta
y_63 = y0 + 0.632 * K_est_initial * u_step
idx63 = np.where(y_meas >= y_63)[0]
if len(idx63) > 0:
    tau_est_initial = t[idx63[0]] - theta_est_initial
else:
    tau_est_initial = 5.0

p0 = [K_est_initial, tau_est_initial, theta_est_initial]

# Ajuste não linear
try:
    popt, pcov = curve_fit(lambda t, K, tau, theta: fopdt_model(t, K, tau, theta),
                           t, y_meas, p0=p0)
    K_est, tau_est, theta_est = popt
    print("Parâmetros identificados:")
    print(f"K  = {K_est:.3f}  (real: {Kp_true:.3f})")
    print(f"τ  = {tau_est:.3f} min  (real: {tau_true:.3f})")
    print(f"θ  = {theta_est:.3f} min  (real: {theta_true:.3f})")
except Exception as e:
    print("Falha no ajuste:", e)
    K_est, tau_est, theta_est = p0

# ==========================================
# 3. VALIDAÇÃO: COMPARAR MODELO COM DADOS
# ==========================================
y_fitted = fopdt_model(t, K_est, tau_est, theta_est)

# Calcular erro quadrático médio (RMSE)
rmse = np.sqrt(np.mean((y_meas - y_fitted)**2))
print(f"RMSE: {rmse:.4f}")

# ==========================================
# 4. PLOTAGEM
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(t, y_meas, 'bo', markersize=3, label='Dados experimentais (com ruído)')
plt.plot(t, y, 'g-', linewidth=2, label='Processo real (sem ruído)')
plt.plot(t, y_fitted, 'r--', linewidth=2, label='Modelo identificado (FOPDT)')
plt.axhline(y0 + Kp_true * u_step, color='gray', linestyle=':', label='Valor final real')
plt.axvline(theta_true, color='gray', linestyle=':', label='Atraso real')
plt.xlabel('Tempo (min)')
plt.ylabel('Variável controlada (°C)')
plt.title('Identificação de modelo FOPDT a partir de resposta ao degrau')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()