"""
comparacao_mpc_pid.py
Comparação entre MPC e PID para controle de nível de um tanque com restrições.
Ilustra a capacidade do MPC de respeitar limites de atuador e taxa de variação.
"""

import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from scipy.integrate import cumulative_trapezoid  # para simulação contínua opcional

# =============================================================================
# 1. MODELO DO PROCESSO (primeira ordem, discretizado)
# =============================================================================
Ts = 0.2          # período de amostragem [min]
tau = 2.0         # constante de tempo [min]
K = 0.5           # ganho estático [m/(m³/min)]

# Modelo discreto
A = np.exp(-Ts/tau)                 # 0.9048
B = K * (1.0 - np.exp(-Ts/tau))     # 0.0476
C = 1.0

print(f"Modelo discreto: x(k+1) = {A:.4f} x(k) + {B:.4f} u(k)")
print(f"Ganho estático: {K} m/(m³/min)\n")

# =============================================================================
# 2. CLASSE MPC (variável absoluta U, com penalidade na taxa)
# =============================================================================
class MPC:
    def __init__(self, A, B, C, Np, Nc, Q, R, Ru, u_min, u_max, du_max):
        self.A = A
        self.B = B
        self.C = C
        self.Np = Np
        self.Nc = min(Nc, Np)
        self.Q = Q
        self.R = R
        self.Ru = Ru
        self.u_min = u_min
        self.u_max = u_max
        self.du_max = du_max
        self._build_prediction_matrices()

    def _build_prediction_matrices(self):
        # Matriz Psi: (Np x 1) - predição livre
        Psi = np.zeros((self.Np, 1))
        for i in range(self.Np):
            Psi[i, 0] = self.C * (self.A ** (i+1))

        # Matriz Theta: (Np x Nc) - relação entre entradas e saídas
        Theta = np.zeros((self.Np, self.Nc))
        for i in range(self.Np):
            for j in range(min(i+1, self.Nc)):
                Theta[i, j] = self.C * (self.A ** (i-j)) * self.B
            # Para i >= Nc, a entrada mantém-se constante = u_{k+Nc-1}
            if i >= self.Nc:
                for j in range(self.Nc, i+1):
                    Theta[i, self.Nc-1] += self.C * (self.A ** (i-j)) * self.B
        self.Psi = Psi
        self.Theta = Theta

    def compute_control(self, xk, uk_prev, r_ref):
        xk_2d = np.array([[xk]])
        r_ref = np.asarray(r_ref).reshape(-1, 1)

        Q_bar = self.Q * np.eye(self.Np)
        R_bar = self.R * np.eye(self.Nc)

        Y_free = self.Psi @ xk_2d
        H = self.Theta.T @ Q_bar @ self.Theta + R_bar
        f = 2 * (self.Theta.T @ Q_bar @ (Y_free - r_ref)).flatten()

        # Penalidade na taxa de variação (suavidade)
        if self.Ru > 0:
            D = np.eye(self.Nc) - np.eye(self.Nc, k=-1)  # matriz de diferença
            H += self.Ru * (D.T @ D)

        n = self.Nc
        # Construção das restrições lineares A_ub @ U <= b_ub
        A_ub = []
        b_ub = []

        # Limites em U
        A_ub.append(np.eye(n))
        b_ub.append(self.u_max * np.ones(n))
        A_ub.append(-np.eye(n))
        b_ub.append(-self.u_min * np.ones(n))

        # Restrição de taxa no primeiro passo: |U0 - uk_prev| <= du_max
        A_ub.append(np.eye(1, n, 0))
        b_ub.append(self.du_max + uk_prev)
        A_ub.append(-np.eye(1, n, 0))
        b_ub.append(self.du_max - uk_prev)

        # Taxa nos passos seguintes: |U_i - U_{i-1}| <= du_max
        for i in range(1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[i-1] = -1.0
            A_ub.append(row.reshape(1, -1))
            b_ub.append(self.du_max)
            A_ub.append(-row.reshape(1, -1))
            b_ub.append(self.du_max)

        A_ub = np.vstack(A_ub)
        b_ub = np.hstack(b_ub)

        # Resolver QP com cvxopt
        solvers.options['show_progress'] = False
        P = matrix(H, tc='d')
        q = matrix(f, tc='d')
        G = matrix(A_ub, tc='d')
        h = matrix(b_ub, tc='d')

        sol = solvers.qp(P, q, G, h)
        if sol['status'] != 'optimal':
            # Fallback: manter a última ação
            uk = uk_prev
        else:
            U_opt = np.array(sol['x']).flatten()
            uk = U_opt[0]
        return np.clip(uk, self.u_min, self.u_max)


# =============================================================================
# 3. CLASSE PID COM ANTI-WINDUP (back-calculation)
# =============================================================================
class PID:
    def __init__(self, Kp, Ki, Kd, Ts, u_min, u_max, du_max=None, tau_antiwindup=0.1):
        """
        Kp, Ki, Kd: ganhos proporcional, integral e derivativo.
        Ts: período de amostragem.
        u_min, u_max: limites da saída.
        du_max: taxa máxima de variação (opcional, se None, sem limite de taxa).
        tau_antiwindup: constante de tempo para back-calculation.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ts = Ts
        self.u_min = u_min
        self.u_max = u_max
        self.du_max = du_max
        self.tau_antiwindup = tau_antiwindup

        # Estados internos
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

    def compute(self, setpoint, y, uk_prev=None):
        """
        setpoint: referência atual (escalar)
        y: saída medida atual
        uk_prev: valor anterior de u (necessário para limitar taxa)
        Retorna: ação de controle u_k
        """
        error = setpoint - y

        # Proporcional
        P = self.Kp * error

        # Integral com back-calculation (anti-windup)
        # Calcula saída sem saturação
        u_sat = np.clip(P + self.integral, self.u_min, self.u_max)
        # Back-calculation: reduz o integral se a saída saturar
        if self.Ki > 0:
            antiwindup = (u_sat - (P + self.integral)) / self.tau_antiwindup
            self.integral += self.Ki * error * self.Ts - antiwindup * self.Ts
        else:
            self.integral = 0.0

        # Derivativo (filtrado opcionalmente, aqui usamos diferença simples)
        D = self.Kd * (error - self.prev_error) / self.Ts
        self.prev_error = error

        # Saída do PID (ainda sem limitar taxa)
        u_raw = P + self.integral + D
        u = np.clip(u_raw, self.u_min, self.u_max)

        # Limitação de taxa de variação (se especificada)
        if self.du_max is not None and uk_prev is not None:
            du = u - uk_prev
            if du > self.du_max:
                u = uk_prev + self.du_max
            elif du < -self.du_max:
                u = uk_prev - self.du_max

        # Atualiza integral apenas com o valor após back-calculation
        # (já foi atualizado acima)
        self.prev_output = u
        return u


# =============================================================================
# 4. PARÂMETROS DE SIMULAÇÃO
# =============================================================================
# Parâmetros do MPC (ajustados para resposta suave)
Np = 30
Nc = 10
Q = 1.0
R = 0.05
Ru = 0.05
u_min, u_max = 0.0, 3.0
du_max = 0.5

# Parâmetros do PID (sintonizados para mesma dinâmica, sem overshoot)
# Ganhos obtidos por método de alocação de polos para primeira ordem:
# Tempo de subida desejado ~ 2 min (10 amostras). Usamos:
Kp = 1.5 / K   # aproximadamente 3.0
Ki = 0.5 / K   # 1.0
Kd = 0.2 * Kp * tau / 10  # pequeno
# Ajuste fino para evitar overshoot e respeitar limites
Kp = 2.0
Ki = 0.8
Kd = 0.2

pid = PID(Kp, Ki, Kd, Ts, u_min, u_max, du_max=du_max, tau_antiwindup=0.2)

# Inicialização do MPC
mpc = MPC(A, B, C, Np, Nc, Q, R, Ru, u_min, u_max, du_max)

# Tempo de simulação
Tsim = 12.0
Nsteps = int(Tsim / Ts) + 1
time = np.linspace(0, Tsim, Nsteps)

# Setpoint (degraus)
setpoint = np.zeros(Nsteps)
for i, t in enumerate(time):
    if t < 1.0:
        setpoint[i] = 0.0
    elif t < 5.0:
        setpoint[i] = 0.8
    else:
        setpoint[i] = 1.0

# Simulação para MPC
x_mpc = 0.0
u_prev_mpc = 0.0
y_mpc_hist = np.zeros(Nsteps)
u_mpc_hist = np.zeros(Nsteps)
y_mpc_hist[0] = 0.0
u_mpc_hist[0] = 0.0

# Simulação para PID
x_pid = 0.0
u_pid = 0.0
y_pid_hist = np.zeros(Nsteps)
u_pid_hist = np.zeros(Nsteps)
y_pid_hist[0] = 0.0
u_pid_hist[0] = 0.0

# =============================================================================
# 5. LAÇO DE SIMULAÇÃO (MPC e PID)
# =============================================================================
for k in range(Nsteps - 1):
    # ----- MPC -----
    r_future = np.zeros(Np)
    for i in range(Np):
        idx = min(k + i + 1, Nsteps - 1)
        r_future[i] = setpoint[idx]
    uk_mpc = mpc.compute_control(x_mpc, u_prev_mpc, r_future)
    x_mpc_next = A * x_mpc + B * uk_mpc
    y_mpc_next = C * x_mpc_next

    # ----- PID -----
    uk_pid = pid.compute(setpoint[k+1], y_pid_hist[k], u_pid_hist[k])
    x_pid_next = A * x_pid + B * uk_pid
    y_pid_next = C * x_pid_next

    # Armazenamento
    y_mpc_hist[k+1] = y_mpc_next
    u_mpc_hist[k+1] = uk_mpc
    y_pid_hist[k+1] = y_pid_next
    u_pid_hist[k+1] = uk_pid

    # Atualização dos estados
    x_mpc, u_prev_mpc = x_mpc_next, uk_mpc
    x_pid = x_pid_next

# =============================================================================
# 6. GRÁFICOS COMPARATIVOS
# =============================================================================
plt.figure(figsize=(12, 10))

# Saída (nível)
plt.subplot(2, 1, 1)
plt.plot(time, y_mpc_hist, 'b-', linewidth=2, label='MPC')
plt.plot(time, y_pid_hist, 'g-', linewidth=2, label='PID')
plt.plot(time, setpoint, 'r--', linewidth=1.5, label='Setpoint')
plt.ylabel('Nível (m)')
plt.legend()
plt.grid(True)
plt.title('Comparação: MPC vs PID (com anti-windup e limites de taxa)')

# Ação de controle (vazão)
plt.subplot(2, 1, 2)
plt.plot(time, u_mpc_hist, 'b-', linewidth=2, label='MPC - Vazão')
plt.plot(time, u_pid_hist, 'g-', linewidth=2, label='PID - Vazão')
plt.axhline(y=u_min, color='k', linestyle=':', label='Limite inferior (0)')
plt.axhline(y=u_max, color='k', linestyle='--', label='Limite superior (2)')
plt.xlabel('Tempo (min)')
plt.ylabel('Vazão (m³/min)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# =============================================================================
# 7. ANÁLISE DE DESEMPENHO
# =============================================================================
# Índices de desempenho
ISE_mpc = np.sum((y_mpc_hist - setpoint)**2) * Ts
ISE_pid = np.sum((y_pid_hist - setpoint)**2) * Ts
print("\n--- Comparação de desempenho ---")
print(f"ISE (MPC): {ISE_mpc:.4f}")
print(f"ISE (PID): {ISE_pid:.4f}")

# Verificação de restrições
print("\n--- Respeito às restrições ---")
print(f"MPC: u_max atingido? {np.max(u_mpc_hist):.2f} (limite {u_max})")
print(f"PID: u_max atingido? {np.max(u_pid_hist):.2f} (limite {u_max})")
du_pid = np.diff(u_pid_hist)
print(f"PID: |du| máximo = {np.max(np.abs(du_pid)):.3f} (limite {du_max})")
print(f"MPC: |du| máximo = {np.max(np.abs(np.diff(u_mpc_hist))):.3f} (limite {du_max})")

if np.max(u_pid_hist) > u_max + 0.01:
    print("PID violou o limite de vazão (anti-windup insuficiente?)")
else:
    print("PID respeitou limites de vazão graças ao anti-windup.")

if np.max(np.abs(du_pid)) > du_max + 0.01:
    print("PID violou a taxa máxima de variação!")
else:
    print("PID respeitou a taxa máxima de variação (limitador aplicado).")