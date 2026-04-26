# 🚀 Controle Avançado
Repositório da disciplina Controle Avançado de Processos no Programa de Pós-Graduação em Engenharia Quimica

## Aula 01: Atividade Prática em Python (40 min)
Cenário: Controle de Temperatura de um CSTR com Atraso

Utilizaremos um modelo linearizado de primeira ordem com atraso (FOPDT) representando a dinâmica entre a vazão de fluido refrigerante e a temperatura do reator. 

### Parâmetros típicos:

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

✅ Definir o modelo usando a biblioteca control (função tf e pade para aproximar o atraso).

✅ Projetar um PID com sintonia de Ziegler-Nichols baseada na resposta ao degrau em malha aberta (ganho último ou método de malha aberta).

✅ Simular a malha fechada para uma mudança de setpoint e para uma perturbação na carga (ex.: aumento na temperatura de alimentação).

✅ Analisar overshoot, tempo de acomodação, erro em regime e comportamento oscilatório devido ao atraso.

### Análise dos Resultados

- Observa-se overshoot elevado e possíveis oscilações devido ao atraso.

- Se o atraso for aumentado para 4 minutos, o sistema pode se tornar instável com os mesmos ganhos.

- Discussão: o PID não consegue “enxergar” o futuro; o atraso força o controlador a reagir tardiamente, prejudicando a estabilidade.

## Aula 02: Atividade Prática em Python (40 min)
Como usar
Salve o código acima como modelagem.py na pasta Aula_02.

Certifique‑se de ter as bibliotecas instaladas:

pip install numpy matplotlib scipy

Execute:

python modelagem.py

Explicação do código
Geração de dados: simula um processo FOPDT verdadeiro com atraso, adiciona ruído gaussiano e amostra com período Ts.

Identificação: usa curve_fit para ajustar a função fopdt_model aos dados. Fornece chutes iniciais baseados na resposta visual (ganho, atraso e constante de tempo).

Validação: calcula o RMSE e plota os dados reais (com ruído), a resposta exata (sem ruído) e o modelo identificado.

Saída: exibe os parâmetros identificados e o gráfico comparativo.

O script é autossuficiente e deve rodar sem erros. Se ainda houver problemas, verifique a versão do Python e das bibliotecas.

Como usar
Escolha o modelo verdadeiro alterando a variável true_model no início do script (opções: 'fopdt', 'sopdt', 'integrator', 'inverse').

Ajuste os parâmetros do modelo verdadeiro (dentro do bloco if true_model == ...) conforme desejar.

Selecione quais modelos candidatos serão ajustados editando a lista candidate_models (comente os que não quiser testar).

Execute o script:

python modelagem.py

O script exibirá o RMSE e os parâmetros estimados para cada modelo candidato e gerará um gráfico comparativo.

Explicação do código
Geração dos dados: utiliza funções analíticas (ou simulação numérica para o modelo de resposta inversa) para gerar a resposta ao degrau do processo verdadeiro, acrescenta ruído gaussiano.

Identificação: para cada modelo candidato, chama curve_fit para ajustar os parâmetros minimizando o erro quadrático entre a saída do modelo e os dados ruidosos.

Avaliação: calcula o RMSE e armazena os resultados.

Gráfico: mostra os dados, o processo real (sem ruído) e as respostas dos modelos identificados.

Exemplos de uso
Testar um processo de segunda ordem: altere true_model = 'sopdt' e ajuste tau1, tau2.

Ver o efeito da resposta inversa: use true_model = 'inverse' e varie zero_factor.

Comparar identificação: observe qual modelo candidato (ex.: FOPDT) consegue ou não representar bem o processo verdadeiro.


## Aula 03:

---

## Explicação do código (para os alunos)

### Modelo do processo
Utilizamos um **tanque de nível linearizado de primeira ordem**, discretizado com período de amostragem `Ts = 0,2 min`. A função de transferência contínua é  
$$

/frac{H(s)}{Q(s)} = /frac{K}{\tau s + 1}

$$  

com ganho `K = 0,5 m/(m³/min)` e constante de tempo `τ = 2 min`.  
A discretização exata (ZOH) resulta no modelo de espaço de estados:

$$
x_{k+1} = A x_k + B u_k,\qquad y_k = C x_k
$$  

onde `A = exp(-Ts/τ) ≈ 0,9048`, `B = K·(1 - exp(-Ts/τ)) ≈ 0,0476` e `C = 1`.

### Classe MPC (versão final)
A implementação segue os passos essenciais do MPC, com as seguintes características **corrigidas e melhoradas** em relação às versões iniciais:

1. **Variável de controle absoluta `U`** – ao contrário da formulação com incrementos `ΔU`, a versão final otimiza diretamente a sequência de vazões futuras `[u_k, u_{k+1}, ..., u_{k+N_c-1}]`. Isso simplifica as restrições e evita problemas de convergência.

2. **Matrizes de predição `Ψ` e `Θ`** – construídas conforme o slide 8, considerando que após o horizonte de controle `N_c` a entrada permanece constante.

3. **Função custo quadrática** (slide 6):
   $$
   J = \sum_{i=1}^{N_p} \|y_{k+i} - r_{k+i}\|_Q^2 \;+\; \sum_{j=0}^{N_c-1} \|u_{k+j}\|_R^2 \;+\; \sum_{j=1}^{N_c-1} \|\Delta u_{k+j}\|_{R_u}^2
   $$  
   O termo extra `R_u` penaliza mudanças bruscas na vazão (suavidade). Na forma matricial, obtém-se `0.5·Uᵀ·H·U + fᵀ·U`.

4. **Restrições lineares** (slide 7) – tratam limites de amplitude e de taxa de variação:
   - `u_min ≤ u_k ≤ u_max` (ex.: 0 a 2 m³/min)
   - `|u_k - u_{k-1}| ≤ du_max` (ex.: 0,5 m³/min por amostra)
   - Escritas como `A_ub·U ≤ b_ub` e passadas ao solver.

5. **Solver QP** – utiliza a biblioteca `cvxopt`, especializada em programação quadrática convexa. É numericamente estável, não produz avisos de singularidade e é muito mais rápida que `scipy.optimize.minimize`. O problema é sempre convexo porque `R > 0` e `R_u ≥ 0`.

6. **Antecipação da referência** – o vetor `r_future` contém os valores do setpoint nos próximos `N_p` passos. Isso permite que o MPC **abra a válvula antes** da mudança programada do setpoint, reduzindo o atraso e o overshoot. (Se essa antecipação não for desejada, basta substituir `r_future` pelo valor atual do setpoint.)

7. **Horizonte deslizante** – a cada instante, resolve-se um novo QP e aplica-se apenas o primeiro elemento `u_k` da sequência ótima.

### Simulação
O laço principal percorre o tempo total (12 min), atualiza o estado do processo e chama o controlador a cada período `Ts`. A referência é um **degrau duplo** (0 → 0,8 m em t = 1 min; 0,8 → 1,0 m em t = 5 min), ambos alcançáveis dentro dos limites do atuador.

### Visualização
Os gráficos de saída (nível) e variável manipulada (vazão) são gerados lado a lado, mostrando:
- A saída seguindo o setpoint com pequeno ou nenhum overshoot.
- A vazão respeitando os limites `[0, 2]` e a taxa máxima `0,5` por amostra.
- O efeito da antecipação: o nível começa a subir **antes** do primeiro degrau (em t ≈ 0,8 min).

### Verificação
Ao final, o código imprime estatísticas que comprovam o respeito às restrições e calcula o Índice de Erro Quadrático Integral (ISE) para avaliar o desempenho.

## Como executar

1. Instale os pacotes necessários:
   ```bash
   pip install numpy matplotlib cvxopt
   ```
2. Execute o script `mpc_cvxopt_final.py` (ou o nome que você escolheu).
3. A figura será exibida e as estatísticas aparecerão no terminal.

## Observações didáticas

- **Horizontes** – `Np = 30` (cobre 6 min, mais que o tempo de acomodação do tanque) e `Nc = 10` (dá liberdade para planejar a subida). Valores maiores aumentam a estabilidade e o custo computacional; valores menores tornam o controle mais reativo.
- **Pesos** – `Q = 1` (enfatiza o erro), `R = 0,1` (penaliza vazões altas) e `R_u = 0,05` (penaliza mudanças bruscas). Aumentar `R` reduz o overshoot, mas pode deixar a resposta mais lenta. Aumentar `R_u` suaviza a vazão, evitando picos.
- **Restrições duras** – Tanto `u` quanto `Δu` são respeitados exatamente, evitando saturação e movimentos bruscos. Essa é a principal vantagem do MPC sobre o PID, que requer lógica adicional de anti-windup e limitadores.
- **Antecipação** – O conhecimento futuro do setpoint é uma característica única do MPC. Em processos com setpoints programados (ex.: mudanças de carga em reatores), isso traz grande benefício. Para simular um cenário realimentado puro, basta fazer `r_future = np.full(Np, setpoint[k])`.

Este código serve como base para experimentos: os alunos podem variar `Np`, `Nc`, `Q`, `R`, `R_u` e observar os efeitos na resposta, ou introduzir perturbações de carga para testar a robustez. A comparação com um PID (fornecida em outro script) evidencia como o MPC lida naturalmente com restrições.
