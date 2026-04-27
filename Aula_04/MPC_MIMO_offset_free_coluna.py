import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ==========================================================
# 1. PARÂMETROS DO MODELO DA COLUNA DE DESTILAÇÃO
# ==========================================================
# Matrizes em Variáveis de Desvio
A = np.array([[0.88, 0.04], 
              [0.06, 0.82]])

# u1 (Refluxo L) afeta positivamente ambas as composições.
# u2 (Vapor V) afeta negativamente ambas as composições.
B = np.array([[ 0.12, -0.08], 
              [ 0.04, -0.15]])

C = np.eye(2)
D = np.zeros((2,2))

Ts = 1.0 # Tempo de amostragem (minutos)
Np = 15  # Horizonte de Predição um pouco maior para destilação
Nc = 5   # Horizonte de Controle
nx, nu, ny = 2, 2, 2

# Valores Nominais (Ponto de Operação Base)
y_nom = np.array([0.90, 0.10]) # Topo: 90% pureza | Fundo: 10% impureza
u_nom = np.array([2.5, 3.0])   # Refluxo: 2.5 kmol/min | Vapor: 3.0 kmol/min

# Limites Físicos (Absolutos)
u_min_abs = np.array([0.5, 0.5])
u_max_abs = np.array([5.0, 6.0])
y_min_abs = np.array([0.0, 0.0])
y_max_abs = np.array([1.0, 1.0])

# Conversão dos Limites para Variáveis de Desvio (usadas no otimizador)
u_min = u_min_abs - u_nom
u_max = u_max_abs - u_nom
du_min = np.array([-0.3, -0.3]) # Restrição de movimentação da válvula
du_max = np.array([ 0.3,  0.3])
y_min = y_min_abs - y_nom
y_max = y_max_abs - y_nom

# Pesos do MPC
Q_x = np.diag([2.0, 2.0])     # Peso severo para erro de composição
R_du = np.diag([0.5, 0.5])    # Peso para evitar variação brusca de L e V
Q_u = np.diag([0.01, 0.01])

# ==========================================================
# 2. MODELO AUMENTADO E FILTRO DE KALMAN
# ==========================================================
A_a = np.block([[A, np.zeros((nx, ny))], [np.zeros((ny, nx)), np.eye(ny)]])
B_a = np.block([[B], [np.zeros((ny, nu))]])
C_a = np.block([[C, np.eye(ny)]])
nx_a = nx + ny 

Q_kf = np.diag([0.001, 0.001, 0.05, 0.05]) # Confia no modelo, ajusta no distúrbio
R_kf = np.diag([0.01, 0.01])               # Ruído do cromatógrafo/sensor

X_est = np.zeros(nx_a)  
P_kf = np.eye(nx_a)     

# ==========================================================
# 3. FUNÇÃO DO MPC (CORRIGIDA COM D_HAT NAS RESTRIÇÕES)
# ==========================================================
def mpc_kalman_distil(x_atual, x_s, u_s, u_prev, d_hat): # Adicionado d_hat aqui
    u = cp.Variable((nu, Np))
    x = cp.Variable((nx, Np+1))
    epsilon = cp.Variable((ny, Np)) 

    cost = 0
    constraints = [x[:, 0] == x_atual] 

    for i in range(Np):
        constraints += [x[:, i+1] == A @ x[:, i] + B @ u[:, i]]
        
        # CORREÇÃO CRÍTICA: A saída prevista deve incluir o distúrbio estimado!
        y_pred = C @ x[:, i+1] + d_hat 
        
        # Soft constraints para composição
        constraints += [y_pred >= y_min - epsilon[:, i]]
        constraints += [y_pred <= y_max + epsilon[:, i]]
        constraints += [epsilon[:, i] >= 0]

        if i == 0:
            du = u[:, 0] - u_prev
        else:
            du = u[:, i] - u[:, i-1]

        if i < Nc:
            constraints += [du >= du_min, du <= du_max]
            cost += cp.quad_form(du, R_du) 
        else:
            constraints += [u[:, i] == u[:, i-1]] 

        constraints += [u[:, i] >= u_min, u[:, i] <= u_max]

        cost += cp.quad_form(x[:, i+1] - x_s, Q_x)
        cost += cp.quad_form(u[:, i] - u_s, Q_u)
        cost += 10000 * cp.sum_squares(epsilon[:, i]) 

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    
    return u[:, 0].value

# ==========================================================
# 4. LOOP DE SIMULAÇÃO
# ==========================================================
Nsim = 120
x_real = np.zeros((nx, Nsim+1))  
y_meas = np.zeros((ny, Nsim+1))  
u = np.zeros((nu, Nsim))

u_prev = np.array([0.0, 0.0]) # Inicia no ponto nominal (desvio 0)
x_real[:, 0] = np.array([0.0, 0.0])  
y_meas[:, 0] = C @ x_real[:, 0]
X_est[:nx] = x_real[:, 0] 

# Setpoint desejado (Absoluto) -> Topo mais puro (0.95), Fundo mais puro (0.05)
r_ref_abs = np.array([0.96, 0.04]) 
r_ref = r_ref_abs - y_nom # Setpoint em variável de desvio

M_target = np.block([[np.eye(nx) - A, -B], [C, np.zeros((ny, nu))]])

for k in range(Nsim):
    y_k = y_meas[:, k]
    
    S = C_a @ P_kf @ C_a.T + R_kf
    K = P_kf @ C_a.T @ np.linalg.inv(S)
    
    y_hat = C_a @ X_est
    X_est = X_est + K @ (y_k - y_hat)
    P_kf = (np.eye(nx_a) - K @ C_a) @ P_kf
    
    x_hat = X_est[:nx]
    d_hat = X_est[nx:] 
    
    rhs = np.concatenate([np.zeros(nx), r_ref - d_hat])
    alvos = np.linalg.solve(M_target, rhs)
    
    x_s = alvos[:nx]
    u_s = alvos[nx:]
    
    # Agora passamos o d_hat para que as restrições considerem o viés
    u_k = mpc_kalman_distil(x_hat, x_s, u_s, u_prev, d_hat)
    u[:, k] = u_k
    u_prev = u_k.copy()
    
    X_est = A_a @ X_est + B_a @ u_k
    P_kf = A_a @ P_kf @ A_a.T + Q_kf
    
    # --- DISTÚRBIO (Mudança na carga ou composição da alimentação) ---
    disturbio_real = np.array([0.0, 0.0])
    if k >= 60:
        # A alimentação ficou mais pesada: 
        # A pureza de topo cai, e a impureza de fundo sobe.
        disturbio_real = np.array([-0.015, 0.01])
        
    x_real[:, k+1] = A @ x_real[:, k] + B @ u_k + disturbio_real
    ruido_sensor = np.random.normal(0, 0.002, ny) 
    y_meas[:, k+1] = C @ x_real[:, k+1] + D @ u_k + ruido_sensor

# ==========================================================
# 5. CONVERSÃO PARA VALORES ABSOLUTOS E PLOTAGEM
# ==========================================================
# Revertemos as variáveis de desvio para valores absolutos reais para o gráfico
y_abs = y_meas + y_nom[:, np.newaxis]
u_abs = u + u_nom[:, np.newaxis]

tempo = np.arange(Nsim+1) * Ts
tempo_u = np.arange(Nsim) * Ts

plt.figure(figsize=(12, 8))

# Plot Topo (y1)
ax1 = plt.subplot(2, 2, 1)
ax1.plot(tempo, y_abs[0,:], label='Fração Molar (yD)', color='blue')
ax1.axhline(r_ref_abs[0], color='r', linestyle='--', label='Setpoint yD (0.96)')
ax1.set_title("Composição de Topo (Destilado)")
ax1.set_ylabel("Fração Molar")
ax1.set_ylim(0.5, 1.0) 
ax1.legend(loc='lower right')

# Plot Fundo (y2)
ax2 = plt.subplot(2, 2, 2)
ax2.plot(tempo, y_abs[1,:], label='Fração Molar (xB)', color='green')
ax2.axhline(r_ref_abs[1], color='r', linestyle='--', label='Setpoint xB (0.04)')
ax2.set_title("Composição de Fundo (Produto de Fundo)")
ax2.set_ylabel("Fração Molar")
ax2.set_ylim(0.0, 0.15) 
ax2.legend(loc='upper right')

# Plot Ações de Controle (L e V)
ax_u = plt.subplot(2, 1, 2)
linha1 = ax_u.plot(tempo_u, u_abs[0,:], label='Refluxo - L (u1)', color='blue')
ax_u.set_ylabel('Vazão de Refluxo (kmol/min)', color='blue')
ax_u.tick_params(axis='y', labelcolor='blue')
ax_u.set_xlabel('Tempo (min)')
ax_u.set_title("Ações de Controle da Coluna (MPC + Kalman)")

ax_u2 = ax_u.twinx()
linha2 = ax_u2.plot(tempo_u, u_abs[1,:], label='Vapor do Refervedor - V (u2)', color='green')
ax_u2.set_ylabel('Vazão de Vapor (kmol/min)', color='green')
ax_u2.tick_params(axis='y', labelcolor='green')

ax_u.legend(linha1 + linha2, [l.get_label() for l in linha1 + linha2], loc='right')

plt.tight_layout()
plt.show()
