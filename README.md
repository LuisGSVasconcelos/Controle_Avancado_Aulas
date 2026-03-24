# Controle_Avancado_Aulas
Repositório da disciplina Controle Avançado de Processos no Programa de Pós-Graduação em Engenharia Quimica

## Atividade Prática em Python (40 min)
Cenário: Controle de Temperatura de um CSTR com Atraso

Utilizaremos um modelo linearizado de primeira ordem com atraso (FOPDT) representando a dinâmica entre a vazão de fluido refrigerante e a temperatura do reator. 

###Parâmetros típicos:

#### Ganho estático 
$$
Kp = 2.0  \frac{°C}{(m³/h)}
$$

#### Constante de tempo 
$$
τ = 5 min
$$

#### Atraso de transporte 
$$
θ = 2 min
$$

### Função de transferência:

$$
G(s)= \frac{2.0}{5s + 1}e^{-2s}
$$
 
### Passos da Simulação
Definir o modelo usando a biblioteca control (função tf e pade para aproximar o atraso).

Projetar um PID com sintonia de Ziegler-Nichols baseada na resposta ao degrau em malha aberta (ganho último ou método de malha aberta).

Simular a malha fechada para uma mudança de setpoint e para uma perturbação na carga (ex.: aumento na temperatura de alimentação).

Analisar overshoot, tempo de acomodação, erro em regime e comportamento oscilatório devido ao atraso.
