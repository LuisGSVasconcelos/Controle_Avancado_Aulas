# STR — Self-Tuning Regulator (Regulador Auto-Ajustável)

**Disciplina:** Controle Avançado de Processos — Pós-graduação em Engenharia Química

Simulação didática de um **Regulador Auto-Ajustável (STR)** comparado a um **PI clássico com sintonia fixa** em um processo de primeira ordem com **ganho variante no tempo**.

---

## Conceito Fundamental

Um STR (Self-Tuning Regulator) combina três blocos funcionais que operam em tempo real a cada período de amostragem:

### 1. Identificação Online (RLS)

Estima os parâmetros de um modelo ARX de primeira ordem:

```
y[k] = a·y[k-1] + b·u[k-1]
```

O algoritmo de **Mínimos Quadrados Recursivos (RLS)** com **fator de esquecimento (λ)** atualiza recursivamente θ = [a, b]ᵀ a cada nova amostra:

```
ŷ[k] = φᵀ[k] · θ[k-1]              → predição
ε[k] = y[k] - ŷ[k]                  → erro de predição (inovação)
K[k] = P[k-1]·φ / (λ + φᵀ·P[k-1]·φ) → ganho de Kalman
θ[k] = θ[k-1] + K[k]·ε[k]          → atualização dos parâmetros
P[k] = (P[k-1] - K[k]·φᵀ·P[k-1]) / λ → atualização da covariância
```

O fator de esquecimento λ ∈ (0, 1] controla a memória do estimador:
- **λ = 1**: memória infinita (dados antigos têm mesmo peso)
- **λ < 1**: dados recentes têm mais peso (adequado para planta variante no tempo)

### 2. Projeto do Controlador

A partir dos parâmetros estimados (a_est, b_est), calcula-se o ganho do processo e a constante de tempo:

```
Kp_est = b_est / (1 - a_est)    → ganho estimado do processo
τ_est  = -Ts / ln(a_est)        → constante de tempo estimada
```

A sintonia PI é feita por **cancelamento de polo** com constante de tempo de malha fechada (Tc) desejada:

```
Kc = τ_est / (Kp_est · (Tc + θ_dead))
Ti = τ_est
```

Travas de segurança evitam valores extremos: Kc ∈ [0,1; 5,0] e Ti ∈ [1,0; 6,0].

### 3. Lei de Controle (PI Digital)

```
Δu[k] = Kc·(e[k] - e[k-1]) + (Kc/Ti)·e[k]·Ts
u[k]  = u[k-1] + Δu[k]
```

### Vantagem do STR

Quando a planta muda (ex.: degradação do ganho), o controlador se re-sintoniza automaticamente. O PI fixo, sem esta capacidade, deteriora seu desempenho.

---

## Cenário da Simulação

- **Processo**: primeira ordem com ganho variante no tempo
  - Kp = **2,0** para t < 30 min (condição nominal)
  - Kp cai **linearmente** de 2,0 → 0,8 entre 30 e 50 min (degradação)
  - Kp = **0,8** para t > 50 min (condição degradada)
  - Constante de tempo: τ = **5,0 min** (fixa)
  - Período de amostragem: Ts = **0,5 min**

- **Setpoint**:
  - 1,0 → 1,5 em t = 20 min
  - 1,5 → 1,25 em t = 60 min

- **Período de graça do STR**: 5 min (apenas identificação, sem atuação)

- **Duas malhas em paralelo**:
  - **Malha A** — Controlada por **STR** (adapta seus parâmetros)
  - **Malha B** — Controlada por **PI clássico com sintonia fixa** (sintonizado para Kp = 2,0, τ = 5,0)

---

## Métrica de Desempenho: IAE

O **IAE (Integral of Absolute Error)** é calculado a partir do momento em que a planta já está degradada (t ≥ 50 min) para evidenciar a diferença entre STR e PI fixo:

```
IAE = ∫ |e(t)| dt
```

O IAE acumulado ao longo do tempo também é traçado nos gráficos.

---

## Resultados Esperados

Após a degradação da planta (t ≥ 50 min):
- O STR ajusta Kc e Ti para compensar a perda de ganho
- O PI fixo (sintonizado para Kp = 2,0) apresenta erro significativo
- O STR tipicamente reduz o IAE em **60–80%** em relação ao PI fixo

---

## Figuras Geradas

### Figura 1 — STR vs PI Fixo: Variáveis de Processo
- Subplot 1: Resposta da variável de processo (PV) de ambas as malhas vs setpoint
- Subplot 2: Sinal de controle (u) de cada controlador
- Subplot 3: IAE acumulado ao longo do tempo

### Figura 2 — STR: Evolução da Sintonia e Identificação
- Subplot 1: Evolução dos parâmetros Kc e Ti do STR (vs linhas de referência do PI fixo)
- Subplot 2: Ganho Kp real da planta vs Kp estimado pelo RLS
- Subplot 3: Traço da matriz de covariância P (escala log) — quanto menor tr(P), maior a confiança nos parâmetros

### Figura 3 — STR: Erro de Predição e Detalhes do Modelo
- Subplot 1: Erro de predição ε = y − ŷ (indica o quanto o modelo se afasta da planta real)
- Subplot 2: Convergência dos parâmetros estimados a e b vs valores reais

---

## Parâmetros de Configuração

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| Ts | 0,5 min | Período de amostragem |
| τ | 5,0 min | Constante de tempo da planta |
| λ | 0,98 | Fator de esquecimento do RLS |
| P₀ | 1000·I | Covariância inicial (alta incerteza) |
| Kc (fixo) | 1,0 | Ganho proporcional do PI fixo |
| Ti (fixo) | 5,0 min | Tempo integral do PI fixo |
| Tc | 2,0 min | Constante de tempo desejada em malha fechada |

---

## Dependências

- Python 3.x
- numpy
- matplotlib

## Como Executar

```bash
python str4_didatico.py
```

---

## Base Teórica

1. **Åström, K. J. & Wittenmark, B. (2008).** *Adaptive Control* (2nd ed.). Dover Publications.
   - Cap. 3: "Real-Time Parameter Estimation"
   - Cap. 4: "Adaptive Control of Linear Systems"

2. **Seborg, D. E., Edgar, T. F., Mellichamp, D. A., & Doyle III, F. J. (2016).** *Process Dynamics and Control* (4th ed.). Wiley.
   - Cap. 12: "Adaptive Control"

3. **Ljung, L. (1999).** *System Identification: Theory for the User* (2nd ed.). Prentice Hall.
   - Cap. 11: "Recursive Estimation Methods"
