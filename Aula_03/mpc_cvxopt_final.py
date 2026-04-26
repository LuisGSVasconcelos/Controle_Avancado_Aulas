"""
mpc_cvxopt_final.py
Implementação do Controlador Preditivo Baseado em Modelo (MPC)
Versão Final com variável absoluta U, restrições e cvxopt.
"""

import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# =============================================================================
# MODELO DO PROCESSO
# =============================================================================
Ts = 0.2          # Período de amostragem (min)
tau = 2.0         # Constante de tempo (min)
K = 0.5           # Ganho estático (m/(m³/min))

# Modelo de espaço de estados discretizado (ZOH)
A = np.exp(-Ts/tau)                 # ~0.9048
B = K * (1.0 - np.exp(-Ts/tau))     # ~0.0476
C = 1.0

print(f"Modelo discreto: x(k+1) = {A:.4f} x(k) + {B:.4f} u(k)")

# =============================================================================
# CLASSE MPC
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
        """ 2. Constrói as matrizes de predição Psi e Theta """
        # Matriz Psi: resposta livre do sistema
        Psi = np.zeros((self.Np, 1))
        for i in range(self.Np):
            Psi[i, 0] = self.C * (self.A ** (i+1))

        # Matriz Theta: resposta forçada (considerando u constante após Nc)
        Theta = np.zeros((self.Np, self.Nc))
        for i in range(self.Np):
            for j in range(min(i+1, self.Nc)):
                Theta[i, j] = self.C * (self.A ** (i-j)) * self.B
            if i >= self.Nc:
                for j in range(self.Nc, i+1):
                    Theta[i, self.Nc-1] += self.C * (self.A ** (i-j)) * self.B
        
        self.Psi = Psi
        self.Theta = Theta

        # 1. Variável Absoluta U: matrizes auxiliares para cálculo de Delta U
        # Delta U = D * U - E * u_prev
        self.D = np.eye(self.Nc) - np.eye(self.Nc, k=-1)
        self.E = np.zeros((self.Nc, 1))
        self.E[0, 0] = 1.0

    def compute_control(self, xk, uk_prev, r_ref):
        xk_2d = np.array([[xk]])
        r_ref = np.asarray(r_ref).reshape(-1, 1)

        # Matrizes de peso
        Q_bar = self.Q * np.eye(self.Np)
        R_bar = self.R * np.eye(self.Nc)
        Ru_bar = self.Ru * np.eye(self.Nc)

        # Predição da saída livre
        Y_free = self.Psi @ xk_2d

        # 3. Função custo quadrática:
        # J = U^T (Theta^T Q Theta + R + D^T Ru D) U + 2 (Theta^T Q (Y_free - r_ref) - D^T Ru E u_prev)^T U
        H_quad = self.Theta.T @ Q_bar @ self.Theta + R_bar + self.D.T @ Ru_bar @ self.D
        F_lin = 2 * (self.Theta.T @ Q_bar @ (Y_free - r_ref) - self.D.T @ Ru_bar @ self.E * uk_prev)

        # 5. Solver QP (cvxopt minimiza 0.5 * x^T P x + q^T x, por isso P = 2 * H_quad)
        P = matrix(2 * H_quad, tc='d')
        q = matrix(F_lin.flatten(), tc='d')

        # 4. Restrições lineares (A_ub * U <= b_ub)
        n = self.Nc
        A_ub = []
        b_ub = []

        # Limites de amplitude: u_min <= U <= u_max
        A_ub.append(np.eye(n))
        b_ub.append(self.u_max * np.ones(n))
        A_ub.append(-np.eye(n))
        b_ub.append(-self.u_min * np.ones(n))

        # Limites de taxa de variação: -du_max <= D * U - E * u_prev <= du_max
        A_ub.append(self.D)
        b_ub.append(self.du_max * np.ones(n) + (self.E * uk_prev).flatten())
        A_ub.append(-self.D)
        b_ub.append(self.du_max * np.ones(n) - (self.E * uk_prev).flatten())

        # Concatena restrições
        A_ub = np.vstack(A_ub)
        b_ub = np.hstack(b_ub)
        G = matrix(A_ub, tc='d')
        h = matrix(b_ub, tc='d')

        # Resolve o problema de otimização
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h)

        if sol['status'] != 'optimal':
            print(f"Alerta: QP status '{sol['status']}'. Mantendo ação de controle anterior.")
            uk = uk_prev
        else:
            U_opt = np.array(sol['x']).flatten()
            # 7. Horizonte deslizante: aplica apenas o primeiro elemento
            uk = U_opt[0] 

        # Grampeia por segurança numérica
        return np.clip(uk, self.u_min, self.u_max)


# =============================================================================
# PARÂMETROS DE SIMULAÇÃO (Observações didáticas)
# =============================================================================
Np = 30          # Horizonte de predição
Nc = 10          # Horizonte de controle
Q = 1.0          # Peso do erro
R = 0.05          # Peso da variável manipulada
Ru = 0.05        # Peso da taxa de variação (suavidade)
u_min, u_max = 0.0, 3.0
du_max = 0.8     # Taxa máxima de variação (m³/min por amostra)

mpc = MPC(A, B, C, Np, Nc, Q, R, Ru, u_min, u_max, du_max)

Tsim = 12.0      # Tempo total de simulação (min)
Nsteps = int(Tsim / Ts) + 1
time = np.linspace(0, Tsim, Nsteps)

x = 0.0
uk_prev = 0.0
y_hist = np.zeros(Nsteps)
u_hist = np.zeros(Nsteps)
r_hist = np.zeros(Nsteps)

# Setpoint: Degrau duplo (0 -> 0.8 -> 1.0)
setpoint = np.zeros(Nsteps)
for i, t in enumerate(time):
    if t < 1.0:
        setpoint[i] = 0.0
    elif t < 5.0:
        setpoint[i] = 0.8
    else:
        setpoint[i] = 1.0

# =============================================================================
# LAÇO DE SIMULAÇÃO
# =============================================================================
for k in range(Nsteps - 1):
    # 6. Antecipação da referência
    r_future = np.zeros(Np)
    for i in range(Np):
        idx = min(k + i + 1, Nsteps - 1)
        r_future[i] = setpoint[idx]

    # Calcula ação de controle
    uk = mpc.compute_control(x, uk_prev, r_future)

    # Simula a dinâmica da planta (Tanque)
    x_next = A * x + B * uk
    y_next = C * x_next

    # Armazena histórico
    u_hist[k+1] = uk
    y_hist[k+1] = y_next
    r_hist[k+1] = setpoint[k+1]

    # Atualiza variáveis para o próximo passo
    x = x_next
    uk_prev = uk

# =============================================================================
# VISUALIZAÇÃO E VERIFICAÇÃO
# =============================================================================
plt.figure(figsize=(10, 8))

# Gráfico de Nível
plt.subplot(2, 1, 1)
plt.plot(time, y_hist, 'b-', linewidth=2, label='Nível do Tanque (y)')
plt.plot(time, r_hist, 'r--', linewidth=1.5, label='Setpoint (r)')
plt.ylabel('Nível (m)')
plt.legend(loc='lower right')
plt.grid(True)
plt.title('Controle MPC com Antecipação e Restrições')

# Gráfico de Vazão
plt.subplot(2, 1, 2)
plt.step(time, u_hist, 'g-', linewidth=2, where='post', label='Vazão da Válvula (u)')
plt.axhline(y=u_min, color='k', linestyle=':', label=f'Limite u_min = {u_min}')
plt.axhline(y=u_max, color='k', linestyle=':', label=f'Limite u_max = {u_max}')
plt.xlabel('Tempo (min)')
plt.ylabel('Vazão (m³/min)')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()

# Cálculo do ISE
ise = np.sum((y_hist - r_hist)**2) * Ts

# Estatísticas finais
print("\n--- Verificação e Estatísticas ---")
print(f"Nível final alcançado: {y_hist[-1]:.3f} m (Setpoint: {setpoint[-1]:.3f} m)")
print(f"Vazão Máxima utilizada: {np.max(u_hist):.3f} m³/min (Limite: {u_max})")
print(f"Vazão Mínima utilizada: {np.min(u_hist):.3f} m³/min (Limite: {u_min})")
du_calc = np.diff(u_hist)
print(f"Maior salto de vazão (|Δu|): {np.max(np.abs(du_calc)):.3f} m³/min (Limite: {du_max})")
print(f"Índice de Erro Quadrático Integral (ISE): {ise:.4f}")