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
