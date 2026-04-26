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
# Nova entrada na função: 'bias' (erro entre planta e modelo)
def mpc_control_offset_free(x_current, r_ref, u_prev, bias):
    delta_u = cp.Variable((Nc*nu, 1))
    epsilon = cp.Variable((Np*ny, 1))   

    Rk = np.kron(np.ones((Np, 1)), r_ref).reshape(-1, 1)
    
    # Propaga o viés ao longo de todo o horizonte de predição
    Bias_pred = np.kron(np.ones((Np, 1)), bias).reshape(-1, 1)

    x_curr_col = x_current.reshape(-1, 1)
    u_prev_col = u_prev.reshape(-1, 1)
    
    # Predição corrigida: Y = Psi*x + Upsilon*u_prev + Theta*dU + Bias
    Y = Psi @ x_curr_col + Upsilon @ u_prev_col + Theta @ delta_u + Bias_pred

    Q_bar = np.kron(np.eye(Np), Q)
    R_bar = np.kron(np.eye(Nc), R)
    rho = 1000  
    cost = cp.quad_form(Y - Rk, Q_bar) + cp.quad_form(delta_u, R_bar) + rho * cp.sum_squares(epsilon)

    constraints = []
    
    # Restrições de movimentação
    for i in range(Nc):
        constraints += [delta_u[i*nu:(i+1)*nu] >= du_min.reshape(-1, 1)]
        constraints += [delta_u[i*nu:(i+1)*nu] <= du_max.reshape(-1, 1)]

    # Restrições da válvula
    u_atual = u_prev_col
    for i in range(Nc):
        u_atual = u_atual + delta_u[i*nu:(i+1)*nu]
        constraints += [u_atual >= u_min.reshape(-1, 1)]
        constraints += [u_atual <= u_max.reshape(-1, 1)]

    # Restrições de saída (soft)
    constraints += [Y >= np.kron(np.ones((Np, 1)), y_min).reshape(-1, 1) - epsilon]
    constraints += [Y <= np.kron(np.ones((Np, 1)), y_max).reshape(-1, 1) + epsilon]
    constraints += [epsilon >= 0]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    return delta_u.value[:nu].flatten()

# ==========================================================
# 4. LOOP DE SIMULAÇÃO (PLANTA REAL VS MODELO INTERNO)
# ==========================================================
Nsim = 100

# Agora precisamos separar a Planta Real do Modelo Interno do MPC
x_real = np.zeros((nx, Nsim+1))  # Estado verdadeiro da planta
x_hat = np.zeros((nx, Nsim+1))   # Estado nominal estimado pelo MPC
y_meas = np.zeros((ny, Nsim+1))  # Medição dos sensores
u = np.zeros((nu, Nsim))

u_prev = np.array([1.0, 1.0])   
x_real[:, 0] = np.array([1.0, 1.0])  
x_hat[:, 0] = np.array([1.0, 1.0])
y_meas[:, 0] = C @ x_real[:, 0]
r_ref = np.array([1.2, 1.5])    

for k in range(Nsim):
    # 1. Lê os sensores da planta real
    y_k = y_meas[:, k]
    
    # 2. Calcula o que o modelo "achava" que a saída seria
    y_hat_k = C @ x_hat[:, k]
    
    # 3. Estima o Distúrbio (Bias)
    bias_k = y_k - y_hat_k
    
    # 4. MPC calcula ação de controle considerando o viés
    du = mpc_control_offset_free(x_hat[:, k], r_ref, u_prev, bias_k)
    u[:, k] = u_prev + du
    u_prev = u[:, k].copy()
    
    # 5. Atualiza o modelo interno (sem enxergar o distúrbio físico)
    x_hat[:, k+1] = A @ x_hat[:, k] + B @ u[:, k]
    
    # --- INSERÇÃO DO DISTÚRBIO NA PLANTA REAL ---
    disturbio = np.array([0.0, 0.0])
    if k >= 40:
        disturbio = np.array([0.08, -0.05]) # Magnitudes ajustadas
        
    # 6. Simula a Planta Real (sofrendo o impacto do distúrbio)
    x_real[:, k+1] = A @ x_real[:, k] + B @ u[:, k] + disturbio
    y_meas[:, k+1] = C @ x_real[:, k+1] + D @ u[:, k]

# ==========================================================
# 5. PLOTAGEM DOS RESULTADOS
# ==========================================================
plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.plot(y_meas[0,:], label='h1 real', color='blue')
plt.axhline(r_ref[0], color='r', linestyle='--', label='Setpoint h1')
plt.title("Nível do Tanque 1")
plt.legend()

plt.subplot(2,2,2)
plt.plot(y_meas[1,:], label='h2 real', color='green')
plt.axhline(r_ref[1], color='r', linestyle='--', label='Setpoint h2')
plt.title("Nível do Tanque 2")
plt.legend()

plt.subplot(2,1,2)
plt.plot(u[0,:], label='u1 (Válvula 1)')
plt.plot(u[1,:], label='u2 (Válvula 2)')
plt.title("Ações de Controle")
plt.legend()
plt.xlabel('Tempo (amostras)')
plt.tight_layout()
plt.show()
