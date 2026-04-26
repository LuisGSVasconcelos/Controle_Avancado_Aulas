import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Parâmetros do modelo (exemplo)
A = np.array([[0.8, 0.1], [0.2, 0.7]])
B = np.array([[0.5, 0], [0.1, 0.6]])
C = np.eye(2)
D = np.zeros((2,2))

# Parâmetros do MPC
Ts = 0.5
Np = 10
Nc = 4
nx = 2
nu = 2
ny = 2

# Pesos
Q = np.diag([1.0, 1.0])      # peso erro
R = np.diag([0.1, 0.1])      # peso movimento

# Restrições
u_min = np.array([0, 0])
u_max = np.array([2, 2])
du_min = np.array([-0.5, -0.5])
du_max = np.array([0.5, 0.5])
y_min = np.array([0, 0])
y_max = np.array([3, 3])      # soft

# =====================================================================
# CONSTRUÇÃO DAS MATRIZES DE PREDIÇÃO (Psi, Upsilon e Theta)
# =====================================================================
def build_prediction_matrices(A, B, C, Np, Nc, nx, nu, ny):
    # Matrizes temporárias para Y = Px * x0 + Pu * U
    Px = np.zeros((Np * ny, nx))
    Pu = np.zeros((Np * ny, Np * nu))
    
    # Preenchendo Px e Pu recursivamente
    for i in range(Np):
        A_pwr = np.linalg.matrix_power(A, i + 1)
        Px[i*ny:(i+1)*ny, :] = C @ A_pwr
        
        for j in range(i + 1):
            A_pwr_B = np.linalg.matrix_power(A, i - j) @ B
            Pu[i*ny:(i+1)*ny, j*nu:(j+1)*nu] = C @ A_pwr_B

    # Matriz M converte U absoluto para Delta U (considerando horizonte Nc)
    # Matriz Gamma propaga o u_prev ao longo do horizonte
    Gamma = np.tile(np.eye(nu), (Np, 1))
    M = np.zeros((Np * nu, Nc * nu))
    
    for i in range(Np):
        for j in range(min(i + 1, Nc)):
            M[i*nu:(i+1)*nu, j*nu:(j+1)*nu] = np.eye(nu)

    # Matrizes finais da equação Y = Psi*x_k + Upsilon*u_prev + Theta*delta_u
    Psi = Px
    Upsilon = Pu @ Gamma
    Theta = Pu @ M
    
    return Psi, Upsilon, Theta

# Gerando as matrizes antes do loop
Psi, Upsilon, Theta = build_prediction_matrices(A, B, C, Np, Nc, nx, nu, ny)

# Função para resolver o QP
def mpc_control(x_current, r_ref, u_prev):
    # Definir variáveis de decisão: Delta U (sequência de Nc movimentos) e epsilon (folga para saídas)
    delta_u = cp.Variable((Nc*nu, 1))
    epsilon = cp.Variable((Np*ny, 1))   # variáveis de folga para saídas

    # Construir trajetória de referência (assumindo constante)
    Rk = np.kron(np.ones((Np, 1)), r_ref).reshape(-1, 1)

    # Predição das saídas (incluindo a matriz Upsilon e o u_prev)
    x_curr_col = x_current.reshape(-1, 1)
    u_prev_col = u_prev.reshape(-1, 1)
    
    # Y = Psi * x_k + Upsilon * u_{k-1} + Theta * \Delta U
    Y = Psi @ x_curr_col + Upsilon @ u_prev_col + Theta @ delta_u

    # Função custo: erro + movimento + penalização das folgas
    Q_bar = np.kron(np.eye(Np), Q)
    R_bar = np.kron(np.eye(Nc), R)
    rho = 1000  # peso para folga
    cost = cp.quad_form(Y - Rk, Q_bar) + cp.quad_form(delta_u, R_bar) + rho * cp.sum_squares(epsilon)

    # ==========================================================
    # INICIALIZAÇÃO DA LISTA DE RESTRIÇÕES (Causa do erro)
    # ==========================================================
    constraints = []

    # Restrições hard: limites nos movimentos (Delta U)
    for i in range(Nc):
        constraints += [delta_u[i*nu:(i+1)*nu] >= du_min.reshape(-1, 1)]
        constraints += [delta_u[i*nu:(i+1)*nu] <= du_max.reshape(-1, 1)]

    # Restrições hard: limites na variável manipulada (U absoluto)
    u_atual = u_prev_col
    for i in range(Nc):
        # Acumula o incremento de controle atual (u_k = u_{k-1} + delta_u_k)
        u_atual = u_atual + delta_u[i*nu:(i+1)*nu]
        
        # Aplica as restrições de máximo e mínimo ao u absoluto
        constraints += [u_atual >= u_min.reshape(-1, 1)]
        constraints += [u_atual <= u_max.reshape(-1, 1)]

    # Restrições soft: limites nas saídas
    constraints += [Y >= np.kron(np.ones((Np, 1)), y_min).reshape(-1, 1) - epsilon]
    constraints += [Y <= np.kron(np.ones((Np, 1)), y_max).reshape(-1, 1) + epsilon]
    constraints += [epsilon >= 0]

    # Resolver
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    # Retornar primeiro movimento
    delta_u_opt = delta_u.value
    return delta_u_opt[:nu].flatten()

# Loop de simulação (com distúrbio)
Nsim = 100
x = np.zeros((nx, Nsim+1))
y = np.zeros((ny, Nsim+1))
u = np.zeros((nu, Nsim))
u_prev = np.array([1.0, 1.0])   # condição inicial
x[:, 0] = np.array([1.0, 1.0])  # estado inicial
r_ref = np.array([1.2, 1.5])    # setpoint constante

# Inicializar y0 baseado no estado inicial
y[:, 0] = C @ x[:, 0] 

for k in range(Nsim):
    du = mpc_control(x[:, k], r_ref, u_prev)
    u[:, k] = u_prev + du
    u_prev = u[:, k].copy()
    
    # --- INSERÇÃO DO DISTÚRBIO ---
    # Inicializa o distúrbio zerado
    disturbio = np.array([0.0, 0.0])
    
    # A partir do tempo 40, insere um distúrbio constante nos estados
    if k >= 40:
        disturbio = np.array([0.15, -0.10]) 
        #disturbio = np.array([0.08, -0.05]) 
        
    # Simular processo real (modelo da planta + distúrbio)
    x[:, k+1] = A @ x[:, k] + B @ u[:, k] + disturbio
    
    # Calcular a saída
    y[:, k+1] = C @ x[:, k+1] + D @ u[:, k]

# ==========================================================
# 5. CÁLCULO DAS MÉTRICAS DE DESEMPENHO
# ==========================================================
def calcular_metricas(y_array, setpoint, y_inicial, Ts):
    ise = np.sum((y_array - setpoint)**2) * Ts
    degrau = setpoint - y_inicial
    max_y = np.max(y_array)
    overshoot = max(0, (max_y - setpoint) / degrau * 100)
    
    banda = 0.02 * abs(degrau)
    fora_da_banda = np.where(np.abs(y_array - setpoint) > banda)[0]
    ts = fora_da_banda[-1] * Ts if len(fora_da_banda) > 0 else 0.0
        
    return ise, overshoot, ts

# As métricas são calculadas sobre o y_meas (a planta real)
ise_h1, os_h1, ts_h1 = calcular_metricas(y[0,:], r_ref[0], 1.0, Ts)
ise_h2, os_h2, ts_h2 = calcular_metricas(y[1,:], r_ref[1], 1.0, Ts)


# ==========================================================
# 6. PLOTAGEM DOS RESULTADOS COM CAIXAS DE TEXTO E EIXOS DUPLOS
# ==========================================================
tempo = np.arange(Nsim+1) * Ts
tempo_u = np.arange(Nsim) * Ts

plt.figure(figsize=(12, 8))
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')

# Plot Tanque 1
ax1 = plt.subplot(2, 2, 1)
ax1.plot(tempo, y[0,:], label='h1 real (MPC)', color='blue')
ax1.axhline(r_ref[0], color='r', linestyle='--', label='Setpoint h1')
ax1.set_title("Nível do Tanque 1 - MPC Offset-Free")
ax1.set_ylabel("Nível")
ax1.set_ylim(0.8, 2.2) 
ax1.legend(loc='lower right')
texto_h1 = f"ISE: {ise_h1:.4f}\nOS: {os_h1:.1f}%\nTs: {ts_h1:.1f} min"
ax1.text(0.05, 0.95, texto_h1, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)

# Plot Tanque 2
ax2 = plt.subplot(2, 2, 2)
ax2.plot(tempo, y[1,:], label='h2 real (MPC)', color='green')
ax2.axhline(r_ref[1], color='r', linestyle='--', label='Setpoint h2')
ax2.set_title("Nível do Tanque 2 - MPC Offset-Free")
ax2.set_ylim(0.8, 2.2) 
ax2.legend(loc='lower right')
texto_h2 = f"ISE: {ise_h2:.4f}\nOS: {os_h2:.1f}%\nTs: {ts_h2:.1f} min"
ax2.text(0.05, 0.95, texto_h2, transform=ax2.transAxes, fontsize=11, verticalalignment='top', bbox=props)

# Plot Ações de Controle
ax_u = plt.subplot(2, 1, 2)

# Válvula 1 (Eixo principal Esquerdo)
linha1 = ax_u.plot(tempo_u, u[0,:], label='u1 (Válvula 1)', color='blue')
ax_u.set_ylabel('Abertura u1', color='blue')
ax_u.tick_params(axis='y', labelcolor='blue')
ax_u.set_xlabel('Tempo (min)')
ax_u.set_title("Ações de Controle MPC MIMO")

# Válvula 2 (Eixo secundário Direito)
ax_u2 = ax_u.twinx()
linha2 = ax_u2.plot(tempo_u, u[1,:], label='u2 (Válvula 2)', color='green')
ax_u2.set_ylabel('Abertura u2', color='green')
ax_u2.tick_params(axis='y', labelcolor='green')

# Unificando as legendas
linhas = linha1 + linha2
labels = [l.get_label() for l in linhas]
ax_u.legend(linhas, labels, loc='right')

'''
# Plotar resultados (mantém o mesmo código de plotagem)
plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.plot(y[0,:], label='h1')
plt.axhline(r_ref[0], color='r', linestyle='--', label='Setpoint h1')
plt.legend()
plt.subplot(2,2,2)
plt.plot(y[1,:], label='h2')
plt.axhline(r_ref[1], color='r', linestyle='--', label='Setpoint h2')
plt.legend()
plt.subplot(2,1,2)
plt.plot(u[0,:], label='u1')
plt.plot(u[1,:], label='u2')
plt.legend()
plt.xlabel('Tempo (amostras)')
'''


plt.tight_layout()
plt.show()
