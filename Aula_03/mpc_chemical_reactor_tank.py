"""
mpc_cvxopt_absoluto.py
MPC com variável de controle absoluta U – rastreamento correto do setpoint.
"""

import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# =============================================================================
# 1. MODELO DO PROCESSO
# =============================================================================
Ts = 0.2          # min
tau = 2.0         # min
K = 0.5           # m/(m³/min)

A = np.exp(-Ts/tau)                 # 0.9048
B = K * (1.0 - np.exp(-Ts/tau))     # 0.0476
C = 1.0

print(f"Modelo discreto: x(k+1) = {A:.4f} x(k) + {B:.4f} u(k)")
print(f"Ganho estático: {K} m/(m³/min)")

# =============================================================================
# 2. CLASSE MPC COM VARIÁVEL ABSOLUTA U
# =============================================================================
class MPC:
    def __init__(self, A, B, C, Np, Nc, Q, R, Ru, u_min, u_max, du_max):
        """
        Parâmetros:
        Np : horizonte de predição
        Nc : horizonte de controle (número de u's otimizados)
        Q  : peso do erro
        R  : peso da variável de controle (u)
        Ru : peso da taxa de variação (Δu) – opcional
        u_min, u_max : limites da variável manipulada
        du_max : limite máximo da taxa de variação (simétrico)
        """
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
        """Constrói as matrizes de predição usando U (variável absoluta)."""
        # Matriz Psi: relaciona estado inicial com saídas futuras
        Psi = np.zeros((self.Np, 1))
        for i in range(self.Np):
            Psi[i, 0] = self.C * (self.A ** (i+1))

        # Matriz Theta: relaciona as entradas U = [u_k, u_{k+1}, ..., u_{k+Nc-1}]
        # com as saídas futuras. Após Nc, assume-se u constante.
        Theta = np.zeros((self.Np, self.Nc))
        for i in range(self.Np):
            for j in range(min(i+1, self.Nc)):
                # coeficiente de u_{k+j} em y_{k+i+1}
                # É a resposta ao degrau no instante j
                Theta[i, j] = self.C * (self.A ** (i-j)) * self.B
            # Se i >= Nc, as entradas após Nc-1 são mantidas iguais a u_{k+Nc-1}
            # Isso já está embutido na matriz (os coeficientes das colunas j >= Nc são zero)
            # Mas precisamos adicionar o efeito da entrada constante após Nc:
            if i >= self.Nc:
                # A partir de i = Nc, a entrada é constante = u_{k+Nc-1}
                # Então somamos ao coeficiente da última coluna o efeito dos passos extras
                for j in range(self.Nc, i+1):
                    Theta[i, self.Nc-1] += self.C * (self.A ** (i-j)) * self.B
        self.Psi = Psi
        self.Theta = Theta

    def compute_control(self, xk, uk_prev, r_ref):
        """
        Entradas:
            xk : estado atual (escalar)
            uk_prev : valor da entrada no instante anterior (para limitar Δu)
            r_ref : vetor de referências futuras (tamanho Np)
        Retorna:
            uk : valor ótimo para o instante k
        """
        xk_2d = np.array([[xk]])
        r_ref = np.asarray(r_ref).reshape(-1, 1)

        # Matrizes de peso
        Q_bar = self.Q * np.eye(self.Np)
        R_bar = self.R * np.eye(self.Nc)

        # Predição da saída: Y = Psi*xk + Theta*U
        Y_free = self.Psi @ xk_2d

        # Função custo: (Y - R)' Q_bar (Y - R) + U' R_bar U + Ru * (Δu)' (Δu)
        # Vamos incluir também penalização na taxa de variação (opcional)
        H = self.Theta.T @ Q_bar @ self.Theta + R_bar
        # Se Ru > 0, adicionamos penalidade em Δu = U - U_prev (mas U_prev é conhecido)
        # Para simplificar, vamos tratar Δu como restrição dura (já incluso) e não penalizar.
        # Mas se quiser suavidade, adicione um termo extra.
        # Por enquanto, H já inclui R_bar.

        f = 2 * (self.Theta.T @ Q_bar @ (Y_free - r_ref)).flatten()

        # Restrições:
        # 1. Limites em U: u_min <= U_i <= u_max
        # 2. Limites em Δu: -du_max <= U_i - U_{i-1} <= du_max, com U_{-1} = uk_prev
        n = self.Nc
        A_ub = []
        b_ub = []

        # Limites inferiores e superiores de U
        A_ub.append(np.eye(n))
        b_ub.append(self.u_max * np.ones(n))
        A_ub.append(-np.eye(n))
        b_ub.append(-self.u_min * np.ones(n))

        # Restrições de Δu: U_i - U_{i-1} <= du_max  e  U_{i-1} - U_i <= du_max
        # Para i=0: U_0 - uk_prev <= du_max  e  uk_prev - U_0 <= du_max
        # Para i>=1: U_i - U_{i-1} <= du_max  e  U_{i-1} - U_i <= du_max
        # Matriz de diferença: D (n x n) com D[i,i]=1, D[i,i-1]=-1 para i>=1
        D = np.eye(n) - np.eye(n, k=-1)  # diferença forward
        # Para i=0, a restrição envolve uk_prev: vamos tratar separadamente
        # Restrições para i=0:
        #   U0 <= du_max + uk_prev
        #   -U0 <= du_max - uk_prev
        A_ub0_upper = np.zeros((1, n))
        A_ub0_upper[0, 0] = 1.0
        b_ub0_upper = du_max + uk_prev
        A_ub0_lower = np.zeros((1, n))
        A_ub0_lower[0, 0] = -1.0
        b_ub0_lower = du_max - uk_prev
        A_ub.append(A_ub0_upper)
        b_ub.append(b_ub0_upper)
        A_ub.append(A_ub0_lower)
        b_ub.append(b_ub0_lower)

        # Restrições para i=1..n-1: D[i,:] @ U <= du_max  e -D[i,:] @ U <= du_max
        for i in range(1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[i-1] = -1.0
            A_ub.append(row.reshape(1, -1))
            b_ub.append(du_max)
            A_ub.append(-row.reshape(1, -1))
            b_ub.append(du_max)

        # Concatena todas as restrições
        A_ub = np.vstack(A_ub)
        b_ub = np.hstack(b_ub)

        # Converte para cvxopt (formato Gx <= h)
        P = matrix(H, tc='d')
        q = matrix(f, tc='d')
        G = matrix(A_ub, tc='d')
        h = matrix(b_ub, tc='d')

        # Resolve QP
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h)
        if sol['status'] != 'optimal':
            print(f"QP status: {sol['status']} - usando uk_prev")
            uk = uk_prev
        else:
            U_opt = np.array(sol['x']).flatten()
            uk = U_opt[0]

        return np.clip(uk, self.u_min, self.u_max)


# =============================================================================
# 3. PARÂMETROS DE SIMULAÇÃO
# =============================================================================
Np = 30          # horizonte de predição maior
Nc = 15          # horizonte de controle maior (permite atingir setpoint)
Q = 1.0          # peso do erro
R = 0.1         # peso pequeno na ação de controle (permite usar u alto)
Ru = 0.0         # sem penalidade adicional na taxa (apenas restrições duras)
u_min, u_max = 0.0, 4.0
du_max = 0.5     # taxa máxima de variação por passo

mpc = MPC(A, B, C, Np, Nc, Q, R, Ru, u_min, u_max, du_max)

Tsim = 10.0      # tempo total de simulação (min)
Nsteps = int(Tsim / Ts) + 1
time = np.linspace(0, Tsim, Nsteps)

x = 0.0
uk_prev = 0.0
y_hist = np.zeros(Nsteps)
u_hist = np.zeros(Nsteps)
r_hist = np.zeros(Nsteps)
y_hist[0] = 0.0
u_hist[0] = 0.0

# Setpoint: 0 -> 0.8 m (t=1min) -> 1.0 m (t=5min)
setpoint = np.zeros(Nsteps)
for i, t in enumerate(time):
    if t < 1.0:
        setpoint[i] = 0.0
    elif t < 5.0:
        setpoint[i] = 0.8
    else:
        setpoint[i] = 1.0

# =============================================================================
# 4. SIMULAÇÃO MPC
# =============================================================================
for k in range(Nsteps - 1):
    # Referência futura
    r_future = np.zeros(Np)
    for i in range(Np):
        idx = min(k + i + 1, Nsteps - 1)
        r_future[i] = setpoint[idx]

    uk = mpc.compute_control(x, uk_prev, r_future)

    # Simula processo
    x_next = A * x + B * uk
    y_next = C * x_next

    u_hist[k+1] = uk
    y_hist[k+1] = y_next
    r_hist[k+1] = setpoint[k+1]

    x = x_next
    uk_prev = uk

# =============================================================================
# 5. GRÁFICOS
# =============================================================================
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(time, y_hist, 'b-', linewidth=2, label='Nível (MPC)')
plt.plot(time, r_hist, 'r--', linewidth=1.5, label='Setpoint')
plt.ylabel('Nível (m)')
plt.legend()
plt.grid(True)
plt.title('MPC com Variável Absoluta U – Rastreamento Correto')

plt.subplot(2, 1, 2)
plt.plot(time, u_hist, 'g-', linewidth=2, label='Vazão de entrada')
plt.axhline(y=u_min, color='k', linestyle=':', label=f'Limite inferior = {u_min}')
plt.axhline(y=u_max, color='k', linestyle=':', label=f'Limite superior = {u_max}')
plt.xlabel('Tempo (min)')
plt.ylabel('Vazão (m³/min)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Verificações
print("\n--- Resultados da simulação ---")
print(f"Nível final: {y_hist[-1]:.3f} m (setpoint: {setpoint[-1]:.3f} m)")
print(f"Máximo de u: {np.max(u_hist):.3f} m³/min (limite superior: {u_max})")
print(f"Mínimo de u: {np.min(u_hist):.3f} m³/min")
du_calc = np.diff(u_hist)
print(f"Máximo |Δu|: {np.max(np.abs(du_calc)):.3f} (limite: {du_max})")

if np.max(u_hist) > 1.5:
    print("\n✓ Controlador utilizou vazão suficiente para seguir o setpoint.")
else:
    print("\n✗ Ainda há problema – vazão máxima muito baixa.")