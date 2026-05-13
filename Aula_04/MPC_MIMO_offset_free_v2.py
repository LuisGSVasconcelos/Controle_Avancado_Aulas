import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ==========================================================
# 1. PARÂMETROS DO MODELO E DO MPC
# ==========================================================
A = np.array([[0.8, 0.1], [0.2, 0.7]])
B = np.array([[0.5, 0], [0.1, 0.6]])
C = np.eye(2)
D = np.zeros((2,2))

Ts = 0.5
Np = 10
Nc = 4
nx, nu, ny = 2, 2, 2

Q = np.diag([1.0, 1.0])      # peso erro
R = np.diag([0.1, 0.1])      # peso movimento

u_min = np.array([0, 0])
u_max = np.array([2, 2])
du_min = np.array([-0.5, -0.5])
du_max = np.array([0.5, 0.5])
y_min = np.array([0, 0])
y_max = np.array([3, 3])      

# ==========================================================
# 2. CONSTRUÇÃO DAS MATRIZES DE PREDIÇÃO
# ==========================================================
def build_prediction_matrices(A, B, C, Np, Nc, nx, nu, ny):
    Px = np.zeros((Np * ny, nx))
    Pu = np.zeros((Np * ny, Np * nu))
    
    for i in range(Np):
        A_pwr = np.linalg.matrix_power(A, i + 1)
        Px[i*ny:(i+1)*ny, :] = C @ A_pwr
        for j in range(i + 1):
            A_pwr_B = np.linalg.matrix_power(A, i - j) @ B
            Pu[i*ny:(i+1)*ny, j*nu:(j+1)*nu] = C @ A_pwr_B

    Gamma = np.tile(np.eye(nu), (Np, 1))
    M = np.zeros((Np * nu, Nc * nu))
    for i in range(Np):
        for j in range(min(i + 1, Nc)):
            M[i*nu:(i+1)*nu, j*nu:(j+1)*nu] = np.eye(nu)

    Psi = Px
    Upsilon = Pu @ Gamma
    Theta = Pu @ M
    return Psi, Upsilon, Theta

Psi, Upsilon, Theta = build_prediction_matrices(A, B, C, Np, Nc, nx, nu, ny)

# ==========================================================
# 3. FUNÇÃO DO MPC COM CORREÇÃO DE VIÉS (OFFSET-FREE)
# ==========================================================
def mpc_control_offset_free(x_current, r_ref, u_prev, bias):
    delta_u = cp.Variable((Nc*nu, 1))
    epsilon = cp.Variable((Np*ny, 1))   

    Rk = np.kron(np.ones((Np, 1)), r_ref).reshape(-1, 1)
    Bias_pred = np.kron(np.ones((Np, 1)), bias).reshape(-1, 1)

    x_curr_col = x_current.reshape(-1, 1)
    u_prev_col = u_prev.reshape(-1, 1)
    
    Y = Psi @ x_curr_col + Upsilon @ u_prev_col + Theta @ delta_u + Bias_pred

    Q_bar = np.kron(np.eye(Np), Q)
    R_bar = np.kron(np.eye(Nc), R)
    rho = 1000  
    cost = cp.quad_form(Y - Rk, Q_bar) + cp.quad_form(delta_u, R_bar) + rho * cp.sum_squares(epsilon)

    constraints = []
    
    for i in range(Nc):
        constraints += [delta_u[i*nu:(i+1)*nu] >= du_min.reshape(-1, 1)]
        constraints += [delta_u[i*nu:(i+1)*nu] <= du_max.reshape(-1, 1)]

    u_atual = u_prev_col
    for i in range(Nc):
        u_atual = u_atual + delta_u[i*nu:(i+1)*nu]
        constraints += [u_atual >= u_min.reshape(-1, 1)]
        constraints += [u_atual <= u_max.reshape(-1, 1)]

    constraints += [Y >= np.kron(np.ones((Np, 1)), y_min).reshape(-1, 1) - epsilon]
    constraints += [Y <= np.kron(np.ones((Np, 1)), y_max).reshape(-1, 1) + epsilon]
    constraints += [epsilon >= 0]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    return delta_u.value[:nu].flatten()

# ==========================================================
# 4. LOOP DE SIMULAÇÃO
# ==========================================================
Nsim = 100

x_real = np.zeros((nx, Nsim+1))  
x_hat = np.zeros((nx, Nsim+1))   
y_meas = np.zeros((ny, Nsim+1))  
u = np.zeros((nu, Nsim))

u_prev = np.array([1.0, 1.0])   
x_real[:, 0] = np.array([1.0, 1.0])  
x_hat[:, 0] = np.array([1.0, 1.0])
y_meas[:, 0] = C @ x_real[:, 0]
r_ref = np.array([1.2, 1.5])    

for k in range(Nsim):
    y_k = y_meas[:, k]
    y_hat_k = C @ x_hat[:, k]
    bias_k = y_k - y_hat_k
    
    du = mpc_control_offset_free(x_hat[:, k], r_ref, u_prev, bias_k)
    u[:, k] = u_prev + du
    u_prev = u[:, k].copy()
    
    x_hat[:, k+1] = A @ x_hat[:, k] + B @ u[:, k]
    
    # Distúrbio em t = 20 min (amostra 40)
    disturbio = np.array([0.0, 0.0])
    if k >= 40:
        disturbio = np.array([0.08, -0.05])
        
    x_real[:, k+1] = A @ x_real[:, k] + B @ u[:, k] + disturbio
    y_meas[:, k+1] = C @ x_real[:, k+1] + D @ u[:, k]

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
ise_h1, os_h1, ts_h1 = calcular_metricas(y_meas[0,:], r_ref[0], 1.0, Ts)
ise_h2, os_h2, ts_h2 = calcular_metricas(y_meas[1,:], r_ref[1], 1.0, Ts)

# ==========================================================
# 6. PLOTAGEM DOS RESULTADOS COM CAIXAS DE TEXTO E EIXOS DUPLOS
# ==========================================================
tempo = np.arange(Nsim+1) * Ts
tempo_u = np.arange(Nsim) * Ts

plt.figure(figsize=(12, 8))
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')

# Plot Tanque 1
ax1 = plt.subplot(2, 2, 1)
ax1.plot(tempo, y_meas[0,:], label='h1 real (MPC)', color='blue')
ax1.axhline(r_ref[0], color='r', linestyle='--', label='Setpoint h1')
ax1.set_title("Nível do Tanque 1 - MPC Offset-Free")
ax1.set_ylabel("Nível")
ax1.set_ylim(0.8, 2.2) 
ax1.legend(loc='lower right')
texto_h1 = f"ISE: {ise_h1:.4f}\nOS: {os_h1:.1f}%\nTs: {ts_h1:.1f} min"
ax1.text(0.05, 0.95, texto_h1, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)

# Plot Tanque 2
ax2 = plt.subplot(2, 2, 2)
ax2.plot(tempo, y_meas[1,:], label='h2 real (MPC)', color='green')
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

plt.tight_layout()
plt.show()