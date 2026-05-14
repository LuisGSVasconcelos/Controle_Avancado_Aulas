import numpy as np  
import matplotlib.pyplot as plt  
from scipy.integrate import solve_ivp

# =====================================================  
#  DEFINIÇÃO DO SISTEMA  
# =====================================================

# Planta (parâmetros verdadeiros — desconhecidos pelo MRAC)  
A_p = np.array([[0, 1],  
                [-2, -1]])  
B_p = np.array([[0],  
                [1]])

# Modelo de referência  
A_m = np.array([[0, 1],  
                [-9, -6]])  
B_m = np.array([[0],  
                [9]])

# Parâmetros ideais (desconhecidos — servem apenas para verificação)  
theta_star = np.array([-7, -5, 9])

# Matriz P (solução de A_m^T P + P A_m = -I)  
P = np.array([[7/6,    1/18],  
              [1/18,   5/54]])

PBp = (P @ B_p).flatten()  # Pré-computar (shape 2,)

# Taxas de adaptação  
Gamma = np.diag([300.0, 300.0, 300.0])

# Condições iniciais  
x_p0 = np.array([0.0, 0.0])   # planta  
x_m0 = np.array([0.0, 0.0])   # modelo  
theta0 = np.array([0.0, 0.0, 0.0])  # chute inicial (longe do ideal)

# =====================================================  
#  SINAL DE REFERÊNCIA  
# =====================================================

def r(t):  
    """Degrau unitário com perturbação senoidal adicionada em t > 15s"""  
    ref = 1.0  
    if t > 15:  
        ref += 0.3 * np.sin(2 * np.pi * 0.5 * t)  
    return ref

# =====================================================  
#  DINÂMICA DO SISTEMA COMPLETO  
# =====================================================

def sistema(t, y):  
    # Desempacotar estados  
    x_p = y[0:2]  
    x_m = y[2:4]  
    theta = y[4:7]

    # Sinal de referência  
    rt = r(t)

    # Regressor  
    phi = np.array([x_p[0], x_p[1], rt])

    # Erro de rastreamento  
    e = x_p - x_m

    # Escalar e^T P B_p  
    sigma_e = float(e @ PBp)

    # Norma do erro  
    norm_e = np.linalg.norm(e)

    # Lei de adaptação (e-modification com σ reduzido)  
    sigma_emod = 0.1  
    dtheta = -Gamma @ phi * sigma_e - sigma_emod * norm_e * theta

    # Controle  
    u = float(theta @ phi)

    # Dinâmica da planta  
    dx_p = A_p @ x_p.flatten() + B_p.flatten() * u

    # Dinâmica do modelo de referência  
    dx_m = A_m @ x_m.flatten() + B_m.flatten() * float(rt)

    return np.concatenate([dx_p, dx_m, dtheta])

# =====================================================  
#  SIMULAÇÃO  
# =====================================================

t_span = (0, 30)  
y0 = np.concatenate([x_p0, x_m0, theta0])

sol = solve_ivp(sistema, t_span, y0, max_step=0.01,  
                method='RK45', rtol=1e-8, atol=1e-10)

t = sol.t  
x_p = sol.y[0:2]  
x_m = sol.y[2:4]  
theta = sol.y[4:7]  
e = x_p - x_m

# =====================================================  
#  VISUALIZAÇÃO  
# =====================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Saída vs Referência ---  
ax = axes[0, 0]  
ax.plot(t, x_p[0], 'b-', linewidth=2, label='Planta $x_1$ (MRAC)')  
ax.plot(t, x_m[0], 'r--', linewidth=2, label='Modelo ref. $x_{m1}$')  
ax.set_xlabel('Tempo [s]')  
ax.set_ylabel('Saída $x_1$')  
ax.set_title('Rastreamento da Saída')  
ax.legend()  
ax.grid(True, alpha=0.3)

# --- Erro de rastreamento ---  
ax = axes[0, 1]  
ax.plot(t, e[0], 'b-', linewidth=1.5, label='$e_1 = x_1 - x_{m1}$')  
ax.plot(t, e[1], 'r-', linewidth=1.5, label='$e_2 = x_2 - x_{m2}$')  
ax.axhline(0, color='k', linewidth=0.5)  
ax.set_xlabel('Tempo [s]')  
ax.set_ylabel('Erro')  
ax.set_title('Erro de Rastreamento')  
ax.legend()  
ax.grid(True, alpha=0.3)

# --- Parâmetros adaptativos ---  
ax = axes[1, 0]  
labels = [r'$\theta_1$ (ideal $= -7$)',  
          r'$\theta_2$ (ideal $= -5$)',  
          r'$\theta_3$ (ideal $= 9$)']  
colors = ['blue', 'red', 'green']  
for i in range(3):  
    ax.plot(t, theta[i], color=colors[i], linewidth=2, label=labels[i])  
    ax.axhline(theta_star[i], color=colors[i], linestyle='--', alpha=0.4)  
ax.set_xlabel('Tempo [s]')  
ax.set_ylabel('Valor do parâmetro')  
ax.set_title('Evolução dos Parâmetros do Controlador')  
ax.legend()  
ax.grid(True, alpha=0.3)

# --- Sinal de controle ---  
ax = axes[1, 1]  
u = np.array([theta[0, i] * x_p[0, i] +  
              theta[1, i] * x_p[1, i] +  
              theta[2, i] * r(t[i]) for i in range(len(t))])  
ax.plot(t, u, 'purple', linewidth=1.5)  
ax.set_xlabel('Tempo [s]')  
ax.set_ylabel('$u(t)$')  
ax.set_title('Sinal de Controle')  
ax.grid(True, alpha=0.3)

plt.tight_layout()  
plt.savefig('mrac_simulacao.png', dpi=150, bbox_inches='tight')  
plt.show()

# =====================================================  
#  MÉTRICAS NUMÉRICAS  
# =====================================================

print("=" * 50)  
print("RESULTADOS DA SIMULAÇÃO MRAC")  
print("=" * 50)  
print(f"\nParametros ideais:  theta* = {theta_star}")  
print(f"Parametros finais:  theta  = [{theta[0,-1]:.4f}, {theta[1,-1]:.4f}, {theta[2,-1]:.4f}]")  
print(f"Erro parametrico:   Dtheta = [{theta[0,-1]-theta_star[0]:.4f}, "  
      f"{theta[1,-1]-theta_star[1]:.4f}, {theta[2,-1]-theta_star[2]:.4f}]")  
print(f"\nErro de saida em regime: |e1(t=30)| = {abs(e[0,-1]):.6f}")  
print(f"Norma do erro em regime:  ||e(t=30)||  = {np.linalg.norm(e[:,-1]):.6f}")  