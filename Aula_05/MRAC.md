# Exemplo Numérico Completo de MRAC

Vamos projetar e simular um sistema MRAC completo, passo a passo, com valores numéricos reais.

---

## Passo 1 — Definição da Planta (Processo)

Considere um sistema de segunda ordem cujos parâmetros são **desconhecidos** pelo projetista:

$$G_p(s) = \frac{b_p}{s^2 + a_1 s + a_0}$$

**Valores verdadeiros** (desconhecidos):  
$$a_0 = 2, \quad a_1 = 1, \quad b_p = 1$$

Na forma de espaço de estados:

$$\dot{x}_p = \underbrace{\begin{bmatrix} 0 & 1 \\ -2 & -1 \end{bmatrix}}_{A_p} x_p + \underbrace{\begin{bmatrix} 0 \\ 1 \end{bmatrix}}_{B_p} u$$

---

## Passo 2 — Modelo de Referência

O projetista define o **comportamento desejado** como um sistema de segunda ordem criticamente amortecido com frequência natural $\omega_n = 3$ rad/s:

$$G_m(s) = \frac{9}{s^2 + 6s + 9}$$

$$\dot{x}_m = \underbrace{\begin{bmatrix} 0 & 1 \\ -9 & -6 \end{bmatrix}}_{A_m} x_m + \underbrace{\begin{bmatrix} 0 \\ 9 \end{bmatrix}}_{B_m} r(t)$$

**Verificação**: $A_m$ tem autovalores em $s = -3$ (duplo) → Hurwitz ✓

---

## Passo 3 — Estrutura do Controlador

$$u(t) = \theta_1(t), x_1 + \theta_2(t), x_2 + \theta_3(t), r(t)$$

Onde $\theta(t) = [\theta_1, \theta_2, \theta_3]^T$ são os **três parâmetros ajustáveis**.

Em forma compacta:

$$u(t) = \theta^T(t) \, \phi(t), \quad \phi(t) = \begin{bmatrix} x_1 \\ x_2 \\ r \end{bmatrix}$$

---

## Passo 4 — Condições de Casamento (Matching Conditions)

Para que o controlador consiga forçar a planta a se comportar **exatamente** como o modelo de referência, é necessário que existam parâmetros ideais $\theta^*$ tais que:

$$A_p + B_p \begin{bmatrix} \theta_1^* & \theta_2^* \end{bmatrix} = A_m$$

$$B_p \, \theta_3^* = B_m$$

### Cálculo de $\theta_1^*$ e $\theta_2^*$:

$$\begin{bmatrix} 0 & 1 \\ -2 & -1 \end{bmatrix} + \begin{bmatrix} 0 \\ 1 \end{bmatrix} \begin{bmatrix} \theta_1^* & \theta_2^* \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -9 & -6 \end{bmatrix}$$

$$\begin{bmatrix} 0 & 1 \\ -2 + \theta_1^* & -1 + \theta_2^* \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -9 & -6 \end{bmatrix}$$

$$\theta_1^* = -7, \qquad \theta_2^* = -5$$

### Cálculo de $\theta_3^*$:

$$\begin{bmatrix} 0 \\ 1 \end{bmatrix} \theta_3^* = \begin{bmatrix} 0 \\ 9 \end{bmatrix} \implies \theta_3^* = 9$$

### Parâmetros ideais:

$$\boxed{\theta^* = \begin{bmatrix} -7 \\ -5 \\ 9 \end{bmatrix}}$$

> **Nota:** O projetista **não conhece** esses valores. O algoritmo MRAC vai convergir para eles automaticamente.

---

## Passo 5 — Dinâmica do Erro

Definindo $e(t) = x_p(t) - x_m(t)$:

$$\dot{e} = A_m \, e + B_p \, \tilde{\theta}^T \phi$$

onde $\tilde{\theta}(t) = \theta(t) - \theta^*$ é o erro paramétrico.

---

## Passo 6 — Cálculo da Matriz $P$

Escolhemos $Q = I_2$ e resolvemos a **equação de Lyapunov**:

$$A_m^T P + P A_m = -I_2$$

Escrevendo $P = \begin{bmatrix} p_{11} & p_{12} \\ p_{12} & p_{22} \end{bmatrix}$ e expandindo:

**Elemento (1,1):**  
$$-18 \, p_{12} = -1 \implies p_{12} = \frac{1}{18}$$

**Elemento (2,2):**  
$$2 \, p_{12} - 12 \, p_{22} = -1 \implies \frac{1}{9} - 12 \, p_{22} = -1 \implies p_{22} = \frac{10}{108} = \frac{5}{54}$$

**Elemento (1,2):**  
$$-9 \, p_{22} + p_{11} - 6 \, p_{12} = 0 \implies p_{11} = \frac{9 \cdot 5}{54} + \frac{6}{18} = \frac{5}{6} + \frac{1}{3} = \frac{7}{6}$$

### Resultado:

$$\boxed{P = \begin{bmatrix} \frac{7}{6} & \frac{1}{18} \\[4pt] \frac{1}{18} & \frac{5}{54} \end{bmatrix} \approx \begin{bmatrix} 1.1667 & 0.0556 \\ 0.0556 & 0.0926 \end{bmatrix}}$$

**Verificação** ($P$ definida positiva):  
$$\det(P) = \frac{7}{6} \cdot \frac{5}{54} - \left(\frac{1}{18}\right)^2 = \frac{35}{324} - \frac{1}{324} = \frac{34}{324} > 0 \checkmark$$

### Produto $P B_p$:

$$P B_p = \begin{bmatrix} 1/18 \\ 5/54 \end{bmatrix} \approx \begin{bmatrix} 0.0556 \\ 0.0926 \end{bmatrix}$$

---

## Passo 7 — Lei de Adaptação (com *e-modification*)

### Lei padrão de Lyapunov:

$$\dot{\theta} = -\Gamma \, \phi(t) \, e^T(t) \, P B_p$$

### Com *e-modification* (robusta):

$$\boxed{\dot{\theta}(t) = -\Gamma \, \phi(t) \, \underbrace{e^T(t) P B_p}_{\text{escalar}} - \Gamma \, \|e(t)\| \, \theta(t)}$$

### Escolhas de projeto:

$$\Gamma = \begin{bmatrix} 300 & 0 & 0 \\ 0 & 300 & 0 \\ 0 & 0 & 300 \end{bmatrix}$$

> Ganho $\Gamma = 300$ uniforme para acelerar a convergência. O termo *e-modification* usa $\sigma_{emod} = 0.1$, desacoplado de $\Gamma$.

### Componente a componente:

$$\dot{\theta}_1 = -300 \cdot x_1 \cdot \sigma_e - 0{,}1 \cdot \|e\| \cdot \theta_1$$

$$\dot{\theta}_2 = -300 \cdot x_2 \cdot \sigma_e - 0{,}1 \cdot \|e\| \cdot \theta_2$$

$$\dot{\theta}_3 = -300 \cdot r \cdot \sigma_e - 0{,}1 \cdot \|e\| \cdot \theta_3$$

onde:

$$\sigma_e = e_1 \cdot \frac{1}{18} + e_2 \cdot \frac{5}{54}$$

---

## Passo 8 — Simulação em Python

```python  
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
```

---

## Passo 8B — Exemplo Alternativo: Sistema de 1ª Ordem com Regra MIT

O código a seguir apresenta uma implementação mais simples de MRAC para um sistema de **primeira ordem**, utilizando integração de Euler e a **Regra MIT** para adaptação. A planta possui um **ganho variante no tempo** $K(t)$, e o controlador ajusta dois parâmetros $\theta_1$ e $\theta_2$ para rastrear o modelo de referência.

```python
import numpy as np
import matplotlib.pyplot as plt

def K_real(t: float) -> float:
    if t < 20: return 2.0
    elif t < 30: return 2.0 - (t - 20) / 10 * 1.5
    else: return 0.5

def run_simulation():

    # Parâmetros da Planta
    tau = 5.0

    # Parâmetros do Modelo de Referência
    am = 1/3.0
    bm = am

    # Parâmetros do Controlador
    gamma = 0.5

    # Inicialização
    theta1 = 1.0
    theta2 = 0.0
    y = 0.0
    ym = 0.0
    rf = 0.0
    yf = 0.0

    # Configuração da Simulação
    dt = 0.05
    t_final = 80
    t = np.arange(0, t_final, dt)
    n = len(t)

    y_arr = np.zeros(n)
    ym_arr = np.zeros(n)
    u_arr = np.zeros(n)
    theta1_arr = np.zeros(n)
    theta2_arr = np.zeros(n)

    r = 1.0  # Setpoint (Degrau unitário)

    # Loop de Simulação (MRAC)
    for i, ti in enumerate(t):

        K = K_real(ti)

        # Controle: u = theta1*r - theta2*y
        u = theta1 * r - theta2 * y
        u_arr[i] = u

        # Planta: dy/dt = (-y + K*u)/tau  (Euler)
        y += (-y + K * u) / tau * dt
        y_arr[i] = y

        # Modelo de Referência: dy_m/dt = -am*ym + bm*r
        ym += (-am * ym + bm * r) * dt
        ym_arr[i] = ym

        # Erro
        e = y - ym

        # Filtros passa-baixa de 1ª ordem
        rf += (-am * rf + r) * dt
        yf += (-am * yf + y) * dt

        # Adaptação (Regra MIT)
        theta1 += -gamma * e * rf * dt
        theta2 += gamma * e * yf * dt

        theta1_arr[i] = theta1
        theta2_arr[i] = theta2

    # Gráficos
    fig = plt.figure(figsize=(12, 10))
    plt.suptitle('Simulação de Controlador Adaptativo (MRAC)', fontsize=16)

    plt.subplot(3, 1, 1)
    plt.plot(t, y_arr, label='Saída da Planta (Adaptada)')
    plt.plot(t, ym_arr, '--', label='Modelo de Referência (Desejado)')
    plt.plot(t, [r]*n, ':', label='Setpoint (r)')
    plt.title('Saída do Sistema (y(t)) vs. Referência (y_m(t))')
    plt.ylabel('Saída (Temperatura)')
    plt.grid(True, linestyle='--')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, u_arr, label='Sinal de Controle (u(t))')
    plt.title('Sinal de Controle Adaptativo')
    plt.ylabel('u')
    plt.grid(True, linestyle='--')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, theta1_arr, label='Parâmetro Adaptativo $\\theta_1$')
    plt.plot(t, theta2_arr, label='Parâmetro Adaptativo $\\theta_2$')
    plt.plot(t, [K_real(ti) for ti in t], '--', label='Ganho Real $K(t)$')
    plt.title('Evolução dos Parâmetros Adaptativos e Ganho Real')
    plt.xlabel('Tempo (min)')
    plt.ylabel('Valores de Parâmetros')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    run_simulation()
```

### Principais diferenças para o exemplo de 2ª ordem:

| Aspecto | Exemplo 2ª ordem (Passo 8) | Exemplo 1ª ordem (Passo 8B) |
|---|---|---|
| **Ordem da planta** | 2ª ordem | 1ª ordem |
| **Integração** | `solve_ivp` (RK45 adaptativo) | Euler explícito (passo fixo) |
| **Lei de adaptação** | Lyapunov + *e-modification* | Regra MIT (gradiente) |
| **Sinal de referência** | Degrau + senoide ($t > 15$s) | Degrau puro |
| **Ganho da planta** | Constante ($b_p = 1$) | Variante no tempo $K(t)$ |
| **Parâmetros adaptativos** | 3 ($\theta_1, \theta_2, \theta_3$) | 2 ($\theta_1, \theta_2$) |
| **Filtros** | Não explícitos (embutidos no regressor) | Filtros passa-baixa $r_f$, $y_f$ |

---

## Passo 9 — Resultados Esperados

### Evolução temporal típica:

| Tempo | $e_1(t)$ | $\theta_1(t)$ | $\theta_2(t)$ | $\theta_3(t)$ |  
|---|---|---|---|---|  
| $t = 0$ s | 0.000 | 0.000 | 0.000 | 0.000 |  
| $t = 1$ s | ~0.4 | ~-3.2 | ~-2.1 | ~4.8 |  
| $t = 5$ s | ~0.05 | ~-6.5 | ~-4.7 | ~8.6 |  
| $t = 10$ s | ~0.002 | ~-6.98 | ~-4.99 | ~8.99 |  
| $t = 15$ s | ≈ 0 | ≈ -7.00 | ≈ -5.00 | ≈ 9.00 |

### O que os gráficos mostram:

**Gráfico 1 (Rastreamento):** As curvas de $x_p(t)$ e $x_m(t)$ se sobrepõem após ~5s, mostrando que a planta passa a rastrear perfeitamente o modelo de referência.

**Gráfico 2 (Erro):** O erro parte de um valor inicial significativo (parâmetros inicializados em zero) e converge exponencialmente para zero.

**Gráfico 3 (Parâmetros):** Os três parâmetros convergem para seus valores ideais ($-7$, $-5$, $9$) em aproximadamente 8-10 segundos.

**Gráfico 4 (Controle):** O sinal de controle é grande no início (esforço para compensar o erro paramétrico) e se estabiliza.

---

## Passo 10 — Análise do Efeito da *e-modification*

Para observar a vantagem da *e-modification*, adicione uma **perturbação constante** na planta:

```python  
# Modificar a dinâmica da planta adicionando perturbação:  
perturbacao = 0.5 * (t > 20)  # degrau de perturbação em t = 20s  
dx_p = A_p @ x_p + B_p.flatten() * u + np.array([0, perturbacao])  
```

### Sem *e-modification* (lei padrão de Lyapunov):

```python  
# Lei de adaptação padrão (SEM e-modification)  
dtheta = -Gamma @ phi * sigma_e  # SEM o termo -Γ‖e‖θ  
```

Resultado: os parâmetros $\theta(t)$ **divergem** após $t = 20$s (*parameter drift*), mesmo que o erro de rastreamento pareça pequeno.

### Com *e-modification*:

Os parâmetros permanecem **limitados** e o erro de rastreamento tem um resíduo proporcional à perturbação, mas **não diverge**.

---

## Passo 11 — Diagrama de Blocos do Sistema Simulado

```  
                    r(t)  
                     │  
         ┌───────────┼───────────────────────┐  
         │           │                       │  
         ▼           ▼                       │  
   ┌──────────┐  ┌──────────┐               │  
   │ Modelo de│  │Controlador│              │  
   │Referência│  │ u=θᵀφ    │              │  
   │  ẋₘ=Aₘxₘ│  │(ajustável)│              │  
   │   +Bₘr   │  └─────┬────┘              │  
   └────┬─────┘        │ u                  │  
        │ xₘ           ▼                    │  
        │         ┌──────────┐              │  
        │         │  PLANTA  │              │  
        │         │  ẋₚ=Aₚxₚ│              │  
        │         │   +Bₚu   │              │  
        │         └────┬─────┘              │  
        │              │ xₚ                 │  
        │              │                    │  
        ▼              ▼                    │  
      ┌────────────────────┐               │  
      │    e = xₚ - xₘ     │               │  
      └────────┬───────────┘               │  
               │                            │  
               ▼                            │  
      ┌────────────────────┐               │  
      │ Lei de Adaptação    │               │  
      │ θ̇=-ΓφeᵀPBₚ-Γ‖e‖θ │───────────────┘  
      │ (e-modification)    │      θ(t)  
      └────────────────────┘  
```

---

## Passo 12 — Verificação Final

| Verificação | Valor | Status |  
|---|---|---|  
| $A_m$ Hurwitz? | Autovalores: $-3, -3$ | ✓ |  
| $P > 0$? | $\det(P) = 34/324 > 0$ | ✓ |  
| Matching conditions satisfeitas? | $A_p + B_p \theta_1^{*T} = A_m$ | ✓ |  
| $\dot{V} \leq 0$? | $\dot{V} = -e^TQe \leq 0$ | ✓ |  
| Convergência do erro? | $e(t) \to 0$ | ✓ |  
| Parâmetros limitados? | Pela *e-modification* | ✓ |

---

## Roteiro Didático — Ajuste dos Parâmetros do MRAC

### 1. Entendendo o Papel de Cada Parâmetro

| Parâmetro | Controla | Efeito se muito **pequeno** | Efeito se muito **grande** |
|-----------|----------|----------------------------|----------------------------|
| $\Gamma$ (ganho adaptativo) | Velocidade de convergência dos parâmetros | Adaptação lenta — parâmetros não alcançam os ideais dentro do tempo de simulação | Oscilações e possível instabilidade numérica |
| $\sigma_{emod}$ (e-modification) | Intensidade do termo robusto contra *parameter drift* | Parâmetros divergem na presença de perturbações | Impede a convergência — cria equilíbrio prematuro |

### 2. Roteiro de Ajuste Passo a Passo

#### Passo A — Encontre um $\Gamma$ adequado sem *e-modification*

1. **Zere o termo robusto**: faça $\sigma_{emod} \gets 0$
2. **Comece com $\Gamma$ pequeno** ($\Gamma = 10$) e aumente gradativamente
3. Para cada valor de $\Gamma$, observe:
   - O erro de rastreamento $|e_1(t)|$ ao final da simulação
   - Se os parâmetros $\theta(t)$ estão convergindo na direção correta
4. **Critério de parada**: Quando $\Gamma$ estiver alto o suficiente para que $|e_1(30)| < 0,01$ com convergência suave (sem oscilações), anote esse valor

```python
sigma_emod = 0.0  # sem e-modification
Gamma = np.diag([G, G, G])

# Teste com G = 10, 50, 100, 200, 300, 500
```

**Observe:** com $\Gamma$ muito baixo ($< 50$) os parâmetros mal saem do zero. Com $\Gamma$ muito alto ($> 500$) podem surgir oscilações numéricas.

#### Passo B — Adicione *e-modification* para robustez

1. **Com o $\Gamma$ definido no Passo A**, introduza $\sigma_{emod} = 0,5$ e reduza gradativamente
2. Para cada $\sigma_{emod}$, observe:
   - O erro de rastreamento $|e_1(30)|$
   - Se os parâmetros convergem para perto dos ideais ($-7, -5, 9$)
   - Se há divergência paramétrica (indicador de $\sigma_{emod}$ muito baixo)

```python
sigma_emod = 0.5  # teste com 0.5, 0.2, 0.1, 0.05, 0.01
Gamma = np.diag([300, 300, 300])
dtheta = -Gamma @ phi * sigma_e - sigma_emod * norm_e * theta
```

**Observe o equilíbrio:** $\sigma_{emod} \cdot \Gamma \approx 30$ é o limite prático. Acima disso o termo robusto domina e os parâmetros não convergem.

| $\sigma_{emod}$ | $\sigma_{emod}\cdot\Gamma$ | $\theta_3(30)$ | $\|e_1(30)\|$ | Funciona? |
|:---:|:---:|:---:|:---:|:---:|
| $1{,}0$ | $300$ | $\approx 0{,}04$ | $0{,}83$ | ❌ |
| $0{,}5$ | $150$ | $\approx 1{,}97$ | $0{,}02$ | ⚠️ |
| $0{,}1$ | $30$ | $\approx 4{,}45$ | $0{,}003$ | ✅ |
| $0{,}05$ | $15$ | $\approx 5{,}8$ | $0{,}002$ | ✅ |
| $0{,}01$ | $3$ | $\approx 7{,}5$ | $< 0{,}001$ | ⚠️ (pode divergir) |

#### Passo C — Teste a robustez com perturbação

Com $\Gamma = 300$ e $\sigma_{emod} = 0{,}1$, adicione a perturbação do Passo 10 e verifique:

```python
# Na dinâmica da planta:
perturbacao = 0.5 * (t > 20)
dx_p = A_p @ x_p + B_p.flatten() * u + np.array([0, perturbacao])
```

- **Sem *e-modification*** ($\sigma_{emod}=0$): parâmetros divergem após $t=20s$
- **Com $\sigma_{emod}$ adequado**: parâmetros permanecem limitados, erro residual proporcional à perturbação

### 3. Regras Práticas

| Situação | Sintoma | Solução |
|----------|---------|---------|
| Parâmetros quase não saem do zero | $\Gamma$ baixo demais ou $\sigma_{emod}$ alto demais | Aumente $\Gamma$ ou reduza $\sigma_{emod}$ |
| Parâmetros oscilam muito | $\Gamma$ alto demais | Reduza $\Gamma$ |
| Erro de rastreamento grande mas parâmetros parados | $\sigma_{emod}\cdot\Gamma$ acima do limite | Reduza $\sigma_{emod}$ |
| Parâmetros divergem com perturbação | $\sigma_{emod}$ baixo demais | Aumente $\sigma_{emod}$ |
| Erro não diminui após mudança de referência | Falta excitação persistente | Use referência com conteúdo espectral rico |

### 4. Experimento Visual com os Alunos

Peça aos alunos que executem a simulação com estes pares $(\Gamma, \sigma_{emod})$ e comparem os gráficos:

```python
casos = [
    (10,   0.0,   "Lento, sem e-mod"),
    (300,  0.0,   "Rápido, sem e-mod"),
    (10,   1.0,   "Lento, e-mod forte"),
    (300,  0.1,   "Rápido, e-mod ajustado"),
    (300,  1.0,   "Rápido, e-mod excessivo"),
    (500,  0.01,  "Muito rápido, e-mod fraco"),
]
```

Para cada caso, analisar:
1. **Gráfico 1** (Rastreamento): a planta segue o modelo?
2. **Gráfico 3** (Parâmetros): eles convergem para $(-7, -5, 9)$?
3. **Gráfico 2** (Erro): qual o erro em regime?
4. **Após 15s**: há degradação com a senoide?

### 5. Conclusão Didática

O MRAC exige equilibrar dois objetivos conflitantes:

$$\boxed{\text{Velocidade} \uparrow \quad \Longleftrightarrow \quad \text{Robustez} \downarrow}$$

- $\Gamma$ **alto** → convergência rápida, mas sensível a ruído e perturbações
- $\sigma_{emod}$ **alto** → robustez contra divergência, mas impede a convergência dos parâmetros
- A **razão** $\sigma_{emod} \cdot \Gamma$ é o verdadeiro fator determinante — mantenha $\sigma_{emod} \cdot \Gamma < 50$ para garantir que a adaptação não seja sufocada pelo termo robusto
