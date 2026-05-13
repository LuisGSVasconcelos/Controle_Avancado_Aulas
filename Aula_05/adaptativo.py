import numpy as np
import matplotlib.pyplot as plt

def K_real(t: float) -> float:
    """
    Função que define o ganho real K(t) da planta em função do tempo.
    Simula uma variação de ganho ao longo do tempo.
    """
    if t < 20: return 2.0
    elif t < 30: return 2.0 - (t - 20) / 10 * 1.5
    else: return 0.5

def run_simulation():
    """
    Executa a simulação de Controle Adaptativo de Modelo de Referência (MRAC).
    """
    
    # =========================================================================
    # 1. PARÂMETROS DO SISTEMA
    # =========================================================================
    
    # Parâmetros da Planta (Processo real - desconhecidos pelo controlador)
    tau = 5.0         # Constante de tempo (min)
    
    # Parâmetros do Modelo de Referência (Ideal e Estável)
    am = 1/3.0        # Constante do modelo (-am)
    bm = am           # Ganho unitário (Manter a relação simples para estabilidade)

    # Parâmetros do Controlador
    gamma = 0.5       # Ganho de adaptação (learning rate)

    # Variáveis de Inicialização
    theta1 = 1.0      # Parâmetro adaptativo 1
    theta2 = 0.0      # Parâmetro adaptativo 2
    y = 0.0            # Saída da planta (variável de estado)
    ym = 0.0           # Saída do modelo de referência (variável de estado)
    rf = 0.0           # Sinal de referência filtrado
    yf = 0.0           # Saída filtrada da planta
    
    # =========================================================================
    # 2. CONFIGURAÇÃO DA SIMULAÇÃO
    # =========================================================================
    dt = 0.05          # Passo de tempo (discretização)
    t_final = 80       # Tempo total de simulação
    t = np.arange(0, t_final, dt)
    n = len(t)

    # Arrays para armazenar resultados
    y_arr = np.zeros(n)
    ym_arr = np.zeros(n)
    u_arr = np.zeros(n)
    theta1_arr = np.zeros(n)
    theta2_arr = np.zeros(n)

    r = 1.0            # Setpoint (Degrau unitário, 1.0)

    # =========================================================================
    # 3. LOOP DE SIMULAÇÃO (MRAC)
    # =========================================================================
    for i, ti in enumerate(t):
        
        # 3.1. Ganho e Sinal de Controle
        K = K_real(ti)
        
        # Equação do Controlador (Hipótese de controle: u = theta1*r - theta2*y)
        u = theta1 * r - theta2 * y
        u_arr[i] = u
        
        # 3.2. Atualização da Planta (ODE: dy/dt = (-y + K*u)/tau)
        # Usando integração de Euler: y(t+dt) = y(t) + f(y, u) * dt
        y += (-y + K * u) / tau * dt
        y_arr[i] = y
        
        # 3.3. Modelo de Referência (ODE: dy_m/dt = -am*ym + bm*r)
        # Este modelo define o comportamento desejado (referência estável)
        ym += (-am * ym + bm * r) * dt
        ym_arr[i] = ym
        
        # 3.4. Cálculo do Erro
        e = y - ym
        
        # 3.5. Filtros (Primeiro-ordem IIR)
        # rf: Filtra a entrada r
        rf += (-am * rf + r) * dt
        # yf: Filtra a saída y
        yf += (-am * yf + y) * dt
        
        # 3.6. Adaptação dos Parâmetros (Regra MIT - Model-Ideal-Theory)
        # theta1 e theta2 são ajustados de acordo com o erro e os filtros,
        # buscando minimizar o erro (e -> 0).
        theta1 += -gamma * e * rf * dt
        theta2 += gamma * e * yf * dt
        
        # Armazenamento dos parâmetros adaptados
        theta1_arr[i] = theta1
        theta2_arr[i] = theta2

    # =========================================================================
    # 4. APRESENTAÇÃO DOS GRÁFICOS
    # =========================================================================
    
    fig = plt.figure(figsize=(12, 10))
    plt.suptitle('Simulação de Controlador Adaptativo (MRAC)', fontsize=16)

    # GRÁFICO 1: Saídas (Performance)
    plt.subplot(3, 1, 1)
    plt.plot(t, y_arr, label='Saída da Planta (Adaptada)')
    plt.plot(t, ym_arr, '--', label='Modelo de Referência (Desejado)')
    plt.plot(t, [r]*n, ':', label='Setpoint (r)')
    plt.title('Saída do Sistema (y(t)) vs. Referência (y_m(t))')
    plt.ylabel('Saída (Temperatura)')
    plt.grid(True, linestyle='--')
    plt.legend()

    # GRÁFICO 2: Sinal de Controle
    plt.subplot(3, 1, 2)
    plt.plot(t, u_arr, label='Sinal de Controle (u(t))')
    plt.title('Sinal de Controle Adaptativo')
    plt.ylabel('u')
    plt.grid(True, linestyle='--')
    plt.legend()

    # GRÁFICO 3: Parâmetros Adaptativos e Ganho Real
    plt.subplot(3, 1, 3)
    plt.plot(t, theta1_arr, label='Parâmetro Adaptativo $\\theta_1$')
    plt.plot(t, theta2_arr, label='Parâmetro Adaptativo $\\theta_2$')
    # Plotando o ganho real para comparação
    plt.plot(t, [K_real(ti) for ti in t], '--', label='Ganho Real $K(t)$')
    plt.title('Evolução dos Parâmetros Adaptativos e Ganho Real')
    plt.xlabel('Tempo (min)')
    plt.ylabel('Valores de Parâmetros')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajusta o layout para dar espaço ao super title
    plt.show()

if __name__ == "__main__":
    run_simulation()
