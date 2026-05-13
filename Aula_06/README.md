# STR — Self-Tuning Regulator (Regulador Auto-Ajustável)

Simulação didática de um **Regulador Auto-Ajustável (STR)** comparado a um **PI clássico com sintonia fixa** em um processo de primeira ordem com **ganho variante no tempo**.

## Conceito

O STR combina três blocos funcionais em tempo real:

1. **Identificação Online (RLS)** — Estima os parâmetros de um modelo ARX de primeira ordem via Mínimos Quadrados Recursivos com fator de esquecimento.
2. **Projeto do Controlador** — Calcula a sintonia PI por cancelamento de polo a partir dos parâmetros estimados.
3. **Lei de Controle (PI Digital)** — Aplica a ação de controle com os ganhos atualizados.

Quando a planta muda, o STR se re-sintoniza automaticamente — ao contrário do PI fixo, que deteriora seu desempenho.

## Cenário da Simulação

- **Processo**: 1ª ordem, τ = 5 min, ganho variante: Kp = 2,0 (t < 30 min) → degradação linear → Kp = 0,8 (t ≥ 50 min)
- **Setpoint**: 1,0 → 1,5 (t = 20 min) → 1,25 (t = 60 min)
- **Período de amostragem**: Ts = 0,5 min

## Resultados Esperados

Após a degradação da planta (t ≥ 50 min):
- O STR ajusta Kc e Ti para compensar a perda de ganho
- O PI fixo (sintonizado para Kp = 2,0) apresenta erro significativo
- O STR tipicamente reduz o IAE em 60–80% em relação ao PI fixo

## Figuras Geradas

| Figura | Conteúdo |
|--------|----------|
| 1 | Comparação STR vs PI Fixo (PV, controle, IAE acumulado) |
| 2 | Evolução da sintonia STR, ganho estimado vs real, traço da covariância |
| 3 | Erro de predição e convergência dos parâmetros estimados |

## Dependências

- Python 3.x
- numpy
- matplotlib

## Como Executar

```bash
python str4_didatico.py
```

## Base Teórica

- Åström & Wittenmark, *Adaptive Control* (2nd ed.), Cap. 3–4
- Seborg et al., *Process Dynamics and Control* (4th ed.), Cap. 12
- Ljung, *System Identification* (2nd ed.), Cap. 11
