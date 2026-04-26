import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. PARÂMETROS DO MODELO
# ==========================================================
A = np.array([[0.8, 0.1], [0.2, 0.7]])
B = np.array([[0.5, 0], [0.1, 0.6]])
C = np.eye(2)
D = np.zeros((2,2))

Ts = 0.5 # minutos
nx, nu, ny = 2, 2, 2

u_min = 0.0
u_max = 2.0

# ==========================================================
# 2. CLASSE DO CONTROLADOR PI (Com Anti-Windup)
# ==========================================================
class PI_Controller:
    def __init__(self, Kp, Ki, u_min, u_max, Ts):
        self.Kp = Kp
        self.Ki = Ki
        self.u_min = u_min
        self.u_max = u_max
        self.Ts = Ts
        self.integral = 0.0
        
    def compute(self, setpoint, pv):
        erro = setpoint - pv
        P = self.Kp * erro
        u_calc = P + self.integral + self.Ki * erro * self.Ts
        
        if u_calc > self.u_max:
            u_out = self.u_max
        elif u_calc < self.u_min:
            u_out = self.u_min
        else:
            u_out = u_calc
            self.integral += self.Ki * erro * self.Ts
            
        return u_out

# ==========================================================
# 3. SINTONIA DOS PIDs DESACOPLADOS
# ==========================================================
pi_1 = PI_Controller(Kp=1.2, Ki=0.4, u_min=u_min, u_max=u_max, Ts=Ts)
pi_2 = PI_Controller(Kp=1.2, Ki=0.4, u_min=u_min, u_max=u_max, Ts=Ts)

# ==========================================================
# 4. LOOP DE SIMULAÇÃO
# ==========================================================
Nsim = 100
x = np.zeros((nx, Nsim+1))
y = np.zeros((ny, Nsim+1))
u = np.zeros((nu, Nsim))

x[:, 0] = np.array([1.0, 1.0])
y[:, 0] = C @ x[:, 0]
r_ref = np.array([1.2, 1.5])

for k in range(Nsim):
    h1 = y[0, k]
    h2 = y[1, k]
    
    u1 = pi_1.compute(r_ref[0], h1)
    u2 = pi_2.compute(r_ref[1], h2)
    u[:, k] = [u1, u2]
    
    # Distúrbio em t = 20 min (amostra 40)
    disturbio = np.array([0.0, 0.0])
    if k >= 40:
        disturbio = np.array([0.08, -0.05])
        
    x[:, k+1] = A @ x[:, k] + B @ u[:, k] + disturbio
    y[:, k+1] = C @ x[:, k+1] + D @ u[:, k]

# ==========================================================
# 5. CÁLCULO DAS MÉTRICAS DE DESEMPENHO
# ==========================================================
def calcular_metricas(y_array, setpoint, y_inicial, Ts):
    # 1. ISE (Integral Square Error) em todo o horizonte
    ise = np.sum((y_array - setpoint)**2) * Ts
    
    # 2. Overshoot (%)
    degrau = setpoint - y_inicial
    max_y = np.max(y_array)
    overshoot = max(0, (max_y - setpoint) / degrau * 100)
    
    # 3. Tempo de Acomodação (critério de 2% do degrau)
    banda = 0.02 * abs(degrau)
    fora_da_banda = np.where(np.abs(y_array - setpoint) > banda)[0]
    
    if len(fora_da_banda) > 0:
        ts = fora_da_banda[-1] * Ts
    else:
        ts = 0.0
        
    return ise, overshoot, ts

ise_h1, os_h1, ts_h1 = calcular_metricas(y[0,:], r_ref[0], 1.0, Ts)
ise_h2, os_h2, ts_h2 = calcular_metricas(y[1,:], r_ref[1], 1.0, Ts)

# ==========================================================
# 6. PLOTAGEM DOS RESULTADOS COM CAIXAS DE TEXTO
# ==========================================================
tempo = np.arange(Nsim+1) * Ts
tempo_u = np.arange(Nsim) * Ts

# 1. Cria a tela em branco (ATENÇÃO: Não usar plt.subplots aqui)
plt.figure(figsize=(12, 8))
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')

# 2. Plot Tanque 1 (Posição 1 de um grid 2x2)
ax1 = plt.subplot(2, 2, 1)
ax1.plot(tempo, y[0,:], label='h1 (PI)', color='blue')
ax1.axhline(r_ref[0], color='r', linestyle='--', label='Setpoint h1')
ax1.set_title("Nível do Tanque 1 - Controle PI")
ax1.set_ylabel("Nível")
ax1.set_ylim(0.8, 1.8) 
ax1.legend(loc='lower right')
texto_h1 = f"ISE: {ise_h1:.4f}\nOS: {os_h1:.1f}%\nTs: {ts_h1:.1f} min"
ax1.text(0.05, 0.95, texto_h1, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)

# 3. Plot Tanque 2 (Posição 2 de um grid 2x2)
ax2 = plt.subplot(2, 2, 2)
ax2.plot(tempo, y[1,:], label='h2 (PI)', color='green')
ax2.axhline(r_ref[1], color='r', linestyle='--', label='Setpoint h2')
ax2.set_title("Nível do Tanque 2 - Controle PI")
ax2.set_ylim(0.8, 1.8) 
ax2.legend(loc='lower right')
texto_h2 = f"ISE: {ise_h2:.4f}\nOS: {os_h2:.1f}%\nTs: {ts_h2:.1f} min"
ax2.text(0.05, 0.95, texto_h2, transform=ax2.transAxes, fontsize=11, verticalalignment='top', bbox=props)

# 4. Plot Ações de Controle (Posição 2 de um grid 2x1 - ocupa toda a parte inferior)
ax_u = plt.subplot(2, 1, 2)

# Válvula 1 (Eixo principal Esquerdo)
linha1 = ax_u.plot(tempo_u, u[0,:], label='u1 (Válvula 1)', color='blue')
ax_u.set_ylabel('Abertura u1', color='blue')
ax_u.tick_params(axis='y', labelcolor='blue')
ax_u.set_xlabel('Tempo (min)')
ax_u.set_title("Ações de Controle PI Desacopladas")

# Válvula 2 (Eixo secundário Direito)
ax_u2 = ax_u.twinx()
linha2 = ax_u2.plot(tempo_u, u[1,:], label='u2 (Válvula 2)', color='green')
ax_u2.set_ylabel('Abertura u2', color='green')
ax_u2.tick_params(axis='y', labelcolor='green')

# Unificando as legendas
linhas = linha1 + linha2
labels = [l.get_label() for l in linhas]
ax_u.legend(linhas, labels, loc='right') # loc='right' para não encobrir as curvas que caem no início

plt.tight_layout()
plt.show()