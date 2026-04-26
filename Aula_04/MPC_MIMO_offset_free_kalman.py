import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ==========================================================
# 1. PARÂMETROS DO MODELO DO PROCESSO
# ==========================================================
A = np.array([[0.8, 0.1], [0.2, 0.7]])
B = np.array([[0.5, 0], [0.1, 0.6]])
C = np.eye(2)
D = np.zeros((2,2))

Ts = 0.5
Np = 10
Nc = 4
nx, nu, ny = 2, 2, 2

u_min = np.array([0, 0])
u_max = np.array([2, 2])
du_min = np.array([-0.5, -0.5])
du_max = np.array([0.5, 0.5])
y_min = np.array([0, 0])
y_max = np.array([3, 3])      

# Pesos do MPC
Q_x = np.diag([1.0, 1.0])     # Peso para rastreamento do estado alvo (xs)
R_du = np.diag([1.1, 1.1])    # Peso para supressão de movimento (Delta u)
Q_u = np.diag([0.01, 0.01])   # Peso leve para rastreamento da entrada alvo (us)

# ==========================================================
# 2. MODELO AUMENTADO E FILTRO DE KALMAN
# ==========================================================
# Construção das matrizes aumentadas: X_a = [x, d]^T
A_a = np.block([[A, np.zeros((nx, ny))], 
                [np.zeros((ny, nx)), np.eye(ny)]])
B_a = np.block([[B], 
                [np.zeros((ny, nu))]])
C_a = np.block([[C, np.eye(ny)]])

nx_a = nx + ny # 4 estados (2 reais + 2 distúrbios)

# Matrizes de Covariância do Filtro de Kalman
Q_kf = np.diag([0.01, 0.01, 0.1, 0.1]) # Confia nos estados, ruído maior no distúrbio
R_kf = np.diag([0.05, 0.05])           # Ruído dos sensores

# Inicialização do Filtro
X_est = np.zeros(nx_a)  # Estado estimado inicial [x1, x2, d1, d2]
P_kf = np.eye(nx_a)     # Matriz de covariância do erro

# ==========================================================
# 3. FUNÇÃO DO MPC COM TARGET CALCULATION
# ==========================================================
def mpc_kalman(x_atual, x_s, u_s, u_prev):
    """
    Formulação Simultânea: O otimizador enxerga a planta passo a passo no horizonte
    """
    # Variáveis de decisão ao longo do horizonte
    u = cp.Variable((nu, Np))
    x = cp.Variable((nx, Np+1))
    epsilon = cp.Variable((ny, Np)) # Variável de folga

    cost = 0
    constraints = [x[:, 0] == x_atual] # Condição inicial real

    for i in range(Np):
        # 1. Equação de estado do processo (Dinâmica)
        constraints += [x[:, i+1] == A @ x[:, i] + B @ u[:, i]]
        
        # 2. Restrições na Saída (Soft Constraints)
        y_pred = C @ x[:, i+1]
        constraints += [y_pred >= y_min - epsilon[:, i]]
        constraints += [y_pred <= y_max + epsilon[:, i]]
        constraints += [epsilon[:, i] >= 0]

        # 3. Lógica do Delta U e Horizonte de Controle (Nc)
        if i == 0:
            du = u[:, 0] - u_prev
        else:
            du = u[:, i] - u[:, i-1]

        if i < Nc:
            constraints += [du >= du_min, du <= du_max]
            cost += cp.quad_form(du, R_du) # Penaliza movimentação
        else:
            constraints += [u[:, i] == u[:, i-1]] # Trava a válvula após Nc

        # 4. Restrições Físicas da Válvula
        constraints += [u[:, i] >= u_min, u[:, i] <= u_max]

        # 5. Função Custo (Rastreamento dos Alvos xs e us)
        cost += cp.quad_form(x[:, i+1] - x_s, Q_x)
        cost += cp.quad_form(u[:, i] - u_s, Q_u)
        cost += 1000 * cp.sum_squares(epsilon[:, i]) # Penaliza forte a quebra de limites

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    
    return u[:, 0].value # Retorna apenas o movimento atual

# ==========================================================
# 4. LOOP DE SIMULAÇÃO (Planta vs. Filtro de Kalman)
# ==========================================================
Nsim = 100
x_real = np.zeros((nx, Nsim+1))  
y_meas = np.zeros((ny, Nsim+1))  
u = np.zeros((nu, Nsim))

u_prev = np.array([1.0, 1.0])   
x_real[:, 0] = np.array([1.0, 1.0])  
y_meas[:, 0] = C @ x_real[:, 0]
X_est[:nx] = x_real[:, 0] # Inicializa KF com o estado real

r_ref = np.array([1.2, 1.5])    

# Matriz M para o Target Calculation
M_target = np.block([[np.eye(nx) - A, -B],
                     [C, np.zeros((ny, nu))]])

for k in range(Nsim):
    # -----------------------------------------------------
    # PASSO 1: Leitura do Sensor e Atualização do Kalman (Update)
    # -----------------------------------------------------
    y_k = y_meas[:, k]
    
    # Ganho de Kalman
    S = C_a @ P_kf @ C_a.T + R_kf
    K = P_kf @ C_a.T @ np.linalg.inv(S)
    
    # Correção da estimativa com a medição real
    y_hat = C_a @ X_est
    X_est = X_est + K @ (y_k - y_hat)
    P_kf = (np.eye(nx_a) - K @ C_a) @ P_kf
    
    x_hat = X_est[:nx]
    d_hat = X_est[nx:] # Distúrbio estimado
    
    # -----------------------------------------------------
    # PASSO 2: Target Calculation (Encontrar xs e us ideais)
    # -----------------------------------------------------
    # Resolve M * [xs, us]^T = [0, r - d_hat]^T
    rhs = np.concatenate([np.zeros(nx), r_ref - d_hat])
    alvos = np.linalg.solve(M_target, rhs)
    
    x_s = alvos[:nx]
    u_s = alvos[nx:]
    
    # -----------------------------------------------------
    # PASSO 3: Otimização MPC
    # -----------------------------------------------------
    u_k = mpc_kalman(x_hat, x_s, u_s, u_prev)
    u[:, k] = u_k
    u_prev = u_k.copy()
    
    # -----------------------------------------------------
    # PASSO 4: Predição do Filtro de Kalman para k+1
    # -----------------------------------------------------
    X_est = A_a @ X_est + B_a @ u_k
    P_kf = A_a @ P_kf @ A_a.T + Q_kf
    
    # -----------------------------------------------------
    # PASSO 5: Simulação da Planta Real (com Distúrbio Oculto)
    # -----------------------------------------------------
    disturbio_real = np.array([0.0, 0.0])
    if k >= 40:
        disturbio_real = np.array([0.08, -0.05])
        
    x_real[:, k+1] = A @ x_real[:, k] + B @ u_k + disturbio_real
    # Adicionamos ruído branco na medição para simular sensor real
    ruido_sensor = np.random.normal(0, 0.003, ny) 
    y_meas[:, k+1] = C @ x_real[:, k+1] + D @ u_k + ruido_sensor

# ==========================================================
# 5. MÉTRICAS E PLOTAGEM (Mesmo formato anterior)
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

ise_h1, os_h1, ts_h1 = calcular_metricas(y_meas[0,:], r_ref[0], 1.0, Ts)
ise_h2, os_h2, ts_h2 = calcular_metricas(y_meas[1,:], r_ref[1], 1.0, Ts)

tempo = np.arange(Nsim+1) * Ts
tempo_u = np.arange(Nsim) * Ts

plt.figure(figsize=(12, 8))
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')

ax1 = plt.subplot(2, 2, 1)
ax1.plot(tempo, y_meas[0,:], label='h1 real (MPC+KF)', color='blue')
ax1.axhline(r_ref[0], color='r', linestyle='--', label='Setpoint h1')
ax1.set_title("Nível do Tanque 1 - MPC Offset-Free (Kalman)")
ax1.set_ylabel("Nível")
ax1.set_ylim(0.8, 2.2) 
ax1.legend(loc='lower right')
ax1.text(0.05, 0.95, f"ISE: {ise_h1:.4f}\nOS: {os_h1:.1f}%\nTs: {ts_h1:.1f} min", transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)

ax2 = plt.subplot(2, 2, 2)
ax2.plot(tempo, y_meas[1,:], label='h2 real (MPC+KF)', color='green')
ax2.axhline(r_ref[1], color='r', linestyle='--', label='Setpoint h2')
ax2.set_title("Nível do Tanque 2 - MPC Offset-Free (Kalman)")
ax2.set_ylim(0.8, 2.2) 
ax2.legend(loc='lower right')
ax2.text(0.05, 0.95, f"ISE: {ise_h2:.4f}\nOS: {os_h2:.1f}%\nTs: {ts_h2:.1f} min", transform=ax2.transAxes, fontsize=11, verticalalignment='top', bbox=props)

ax_u = plt.subplot(2, 1, 2)
linha1 = ax_u.plot(tempo_u, u[0,:], label='u1 (Válvula 1)', color='blue')
ax_u.set_ylabel('Abertura u1', color='blue')
ax_u.tick_params(axis='y', labelcolor='blue')
ax_u.set_xlabel('Tempo (min)')
ax_u.set_title("Ações de Controle Otimizadas")

ax_u2 = ax_u.twinx()
linha2 = ax_u2.plot(tempo_u, u[1,:], label='u2 (Válvula 2)', color='green')
ax_u2.set_ylabel('Abertura u2', color='green')
ax_u2.tick_params(axis='y', labelcolor='green')

ax_u.legend(linha1 + linha2, [l.get_label() for l in linha1 + linha2], loc='right')

plt.tight_layout()
plt.show()
