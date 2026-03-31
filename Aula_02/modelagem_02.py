import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import lti, step

# ============================================================================
# 1. DEFINIÇÃO DOS MODELOS VERDADEIROS (para gerar dados)
# ============================================================================

def step_response_fopdt(t, K, tau, theta, y0, u_step):
    """Resposta ao degrau de um sistema FOPDT: y(t) = y0 + K*u_step*(1 - exp(-(t-theta)/tau)) para t >= theta"""
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < theta:
            y[i] = y0
        else:
            y[i] = y0 + K * u_step * (1 - np.exp(-(ti - theta) / tau))
    return y

def step_response_sopdt(t, K, tau1, tau2, theta, y0, u_step):
    """Resposta ao degrau de um sistema SOPDT superamortecido (tau1 != tau2)"""
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < theta:
            y[i] = y0
        else:
            t_ = ti - theta
            y[i] = y0 + K * u_step * (1 - (tau1 * np.exp(-t_/tau1) - tau2 * np.exp(-t_/tau2)) / (tau1 - tau2))
    return y

def step_response_integrator(t, K, tau, theta, y0, u_step):
    """
    Sistema integrador com atraso: G(s) = K/(s (tau s + 1)) e^{-theta s}
    Resposta ao degrau: y = y0 + K*u_step * [ (t - theta) - tau*(1 - exp(-(t-theta)/tau)) ]
    """
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < theta:
            y[i] = y0
        else:
            t_ = ti - theta
            y[i] = y0 + K * u_step * (t_ - tau * (1 - np.exp(-t_ / tau)))
    return y

def step_response_inverse(t, K, tau, theta, zero_factor, y0, u_step):
    """
    Sistema com resposta inversa (zero no semiplano direito):
    G(s) = K (1 - zero_factor * tau s) / ( (tau s + 1)^2 ) e^{-theta s}
    zero_factor > 0 dá inversão inicial.
    """
    # Para simplificar, usamos um sistema de segunda ordem com zero RHP
    # A resposta analítica é mais complexa; usamos simulação numérica com scipy.signal
    # Para evitar dependência extra, implementamos uma aproximação numérica
    # Aqui, vamos gerar a resposta através de uma simulação numérica do modelo em espaço de estados.
    # Mas para manter a independência, faremos um loop com um modelo de diferenças finitas.
    # Vamos discretizar a função de transferência.
    
    # Modelo contínuo: G(s) = K (1 - z * tau s) / (tau s + 1)^2
    # Multiplicar por 1/(s) para resposta ao degrau. Fazer simulação.
    # Utilizaremos scipy.signal.lti e step para maior precisão.
    from scipy.signal import lti, step
    # Construir numerador e denominador
    # Denominador: (tau s + 1)^2 = tau^2 s^2 + 2 tau s + 1
    # Numerador: K (1 - z*tau s) = -K*z*tau s + K
    num = [-K * zero_factor * tau, K]
    den = [tau**2, 2*tau, 1]
    sys = lti(num, den)
    # Adicionar atraso: a função step não lida com atraso diretamente; faremos uma aproximação
    # Vamos simular sem atraso e depois deslocar no tempo
    t_sim = np.linspace(0, max(t) + theta, 1000)
    _, y_sim = step(sys, T=t_sim)
    y_sim = y_sim * u_step + y0
    # Interpolar no tempo desejado com deslocamento
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < theta:
            y[i] = y0
        else:
            idx = np.argmin(np.abs(t_sim - (ti - theta)))
            y[i] = y_sim[idx]
    return y

# ============================================================================
# 2. DEFINIÇÃO DOS MODELOS CANDIDATOS (para identificação)
# ============================================================================

def fopdt_model(t, K, tau, theta):
    """Modelo FOPDT para ajuste (y0 e u_step fixos)"""
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < theta:
            y[i] = y0
        else:
            y[i] = y0 + K * u_step * (1 - np.exp(-(ti - theta) / tau))
    return y

def sopdt_model(t, K, tau1, tau2, theta):
    """Modelo SOPDT superamortecido para ajuste"""
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < theta:
            y[i] = y0
        else:
            t_ = ti - theta
            y[i] = y0 + K * u_step * (1 - (tau1 * np.exp(-t_/tau1) - tau2 * np.exp(-t_/tau2)) / (tau1 - tau2))
    return y

def integrator_model(t, K, tau, theta):
    """Modelo integrador com atraso para ajuste"""
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < theta:
            y[i] = y0
        else:
            t_ = ti - theta
            y[i] = y0 + K * u_step * (t_ - tau * (1 - np.exp(-t_ / tau)))
    return y

def inverse_model(t, K, tau, theta, zero_factor):
    """Modelo com resposta inversa (RHP zero) para ajuste (aproximação numérica)"""
    # Para o ajuste, usamos a mesma estrutura da função de geração, mas agora com parâmetros a estimar
    from scipy.signal import lti, step
    num = [-K * zero_factor * tau, K]
    den = [tau**2, 2*tau, 1]
    sys = lti(num, den)
    t_sim = np.linspace(0, max(t) + theta, 1000)
    _, y_sim = step(sys, T=t_sim)
    y_sim = y_sim * u_step + y0
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < theta:
            y[i] = y0
        else:
            idx = np.argmin(np.abs(t_sim - (ti - theta)))
            y[i] = y_sim[idx]
    return y

# ============================================================================
# 3. CONFIGURAÇÃO DO EXPERIMENTO (EDITAR CONFORME DESEJADO)
# ============================================================================

# Escolha o modelo verdadeiro ('fopdt', 'sopdt', 'integrator', 'inverse')
true_model = 'inverse'   # Altere aqui

# Parâmetros do modelo verdadeiro
if true_model == 'fopdt':
    true_params = {'K': 2.0, 'tau': 5.0, 'theta': 2.0}
elif true_model == 'sopdt':
    true_params = {'K': 2.0, 'tau1': 3.0, 'tau2': 7.0, 'theta': 1.5}
elif true_model == 'integrator':
    true_params = {'K': 0.5, 'tau': 4.0, 'theta': 1.0}
elif true_model == 'inverse':
    true_params = {'K': 2.0, 'tau': 5.0, 'theta': 1.0, 'zero_factor': 0.8}
else:
    raise ValueError("Modelo verdadeiro não reconhecido")

# Parâmetros da simulação
y0 = 0.0
u_step = 10.0
Ts = 0.5
t_final = 50
t = np.arange(0, t_final, Ts)

# Ruído (desvio padrão relativo)
noise_std = 0.05   # 5% do degrau

# ============================================================================
# 4. GERAR DADOS DO PROCESSO VERDADEIRO
# ============================================================================

if true_model == 'fopdt':
    y_true = step_response_fopdt(t, **true_params, y0=y0, u_step=u_step)
elif true_model == 'sopdt':
    y_true = step_response_sopdt(t, **true_params, y0=y0, u_step=u_step)
elif true_model == 'integrator':
    y_true = step_response_integrator(t, **true_params, y0=y0, u_step=u_step)
elif true_model == 'inverse':
    y_true = step_response_inverse(t, **true_params, y0=y0, u_step=u_step)

# Adicionar ruído
np.random.seed(42)
y_meas = y_true + np.random.normal(0, noise_std * np.abs(y_true[-1] - y0), size=len(t))

# ============================================================================
# 5. IDENTIFICAR MODELOS CANDIDATOS
# ============================================================================

# Lista de modelos a testar (pode editar)
candidate_models = [
    {'name': 'FOPDT', 'func': fopdt_model, 'p0': [1.0, 5.0, 1.0], 'bounds': ([0, 0, 0], [np.inf, np.inf, np.inf])},
    {'name': 'SOPDT', 'func': sopdt_model, 'p0': [1.0, 3.0, 7.0, 1.0], 'bounds': ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])},
    {'name': 'Integrator', 'func': integrator_model, 'p0': [0.5, 4.0, 1.0], 'bounds': ([0, 0, 0], [np.inf, np.inf, np.inf])},
    {'name': 'Inverse', 'func': inverse_model, 'p0': [1.0, 5.0, 1.0, 0.5], 'bounds': ([0, 0, 0, 0], [np.inf, np.inf, np.inf, 1.0])},
]

results = []

for cand in candidate_models:
    try:
        popt, pcov = curve_fit(cand['func'], t, y_meas, p0=cand['p0'], bounds=cand['bounds'], maxfev=5000)
        y_fit = cand['func'](t, *popt)
        rmse = np.sqrt(np.mean((y_meas - y_fit)**2))
        results.append({
            'name': cand['name'],
            'params': popt,
            'rmse': rmse,
            'y_fit': y_fit
        })
        print(f"{cand['name']}: RMSE = {rmse:.4f}, Params = {popt}")
    except Exception as e:
        print(f"Falha no ajuste do modelo {cand['name']}: {e}")

# ============================================================================
# 6. PLOTAGEM DOS RESULTADOS
# ============================================================================

plt.figure(figsize=(12, 8))
plt.plot(t, y_meas, 'bo', markersize=3, label='Dados experimentais (com ruído)')
plt.plot(t, y_true, 'k-', linewidth=2, label='Processo real (sem ruído)')

colors = ['r', 'g', 'c', 'm', 'orange']
for i, res in enumerate(results):
    plt.plot(t, res['y_fit'], '--', color=colors[i % len(colors)], 
             label=f"{res['name']} (RMSE={res['rmse']:.3f})")

plt.xlabel('Tempo (min)')
plt.ylabel('Variável controlada')
plt.title(f'Identificação de modelos - Processo verdadeiro: {true_model}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()