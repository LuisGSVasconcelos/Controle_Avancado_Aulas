import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, step, lsim
from control import tf, pade, feedback, step_response

#  Simulação de um sistema de controle de temperatura com atraso usando um controlador PID

# Modelo FOPDT com atraso
Kp = 2.0
tau = 5.0
theta = 2.0
num = [Kp]
den = [tau, 1]
G_s = tf(num, den)

# Aproximação de Padé de 1ª ordem para o atraso
num_pade, den_pade = pade(theta, n=1)
G_delay = tf(num_pade, den_pade)

# Sistema com atraso
G = G_s * G_delay

# Sintonia PID por Ziegler-Nichols (malha aberta)
# Primeiro, obter resposta ao degrau do sistema sem controlador
t, y = step_response(G)
# Encontrar ponto de inflexão e ajustar (simplificado, usaremos parâmetros típicos)
Kc = 1.2 / (Kp * (theta/tau))  # aproximação para FOPDT
Ti = 2.0 * theta
Td = 0.5 * theta

# Controlador PID em forma paralela
C = tf([Kc * Td, Kc, Kc / Ti], [1, 0])  # PID ideal

# Malha fechada
G_cl = feedback(C * G, 1)

# Simulação
t_sim = np.linspace(0, 50, 1000)
t_out, y_out = step_response(G_cl, t_sim)

plt.plot(t_out, y_out)
plt.xlabel('Tempo (min)')
plt.ylabel('Temperatura (°C)')
plt.title('Resposta ao degrau no setpoint com PID + atraso')
plt.grid()
plt.show()