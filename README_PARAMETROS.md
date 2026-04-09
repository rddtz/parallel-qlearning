# RESUMO RÁPIDO: Modificar Parâmetros do Q-Learning

## Os 9 Parâmetros Principais (Linhas 60-86)

| # | Parâmetro | Padrão | O que faz | Aumentar | Diminuir |
|---|-----------|--------|-----------|----------|----------|
| 1 | `GRID_ROWS` / `GRID_COLS` | 4 | Tamanho grid | Grid maior | Grid menor |
| 2 | `ALPHA` | 0.1 | Velocidade aprendizado | +rápido | +lento |
| 3 | `GAMMA` | 0.9 | Importância futuro | +futuro | -futuro |
| 4 | `EPSILON` | 0.1 | Exploração (%) | +exploração | -exploração |
| 5 | `NUM_EPISODES` | 1000 | Episódios treino | +convergência | teste rápido |
| 6 | `MAX_STEPS` | 100 | Limite passos | caminhos longos | caminhos curtos |
| 7 | `REWARD_GOAL` | 100.0 | Motivação objetivo | +motivação | -motivação |
| 8 | `REWARD_OBSTACLE` | -100.0 | Punição obstáculo | +punição | -punição |
| 9 | `REWARD_STEP` | -1.0 | Penalidade passos | +curto | -curto |

## 3 Formas de Mudar

### Forma 1: Editor (Mais Fácil)
```bash
nano qlearning_sequential.c   # Edita linhas 60-86
# Ctrl+X para salvar
make clean && make all && make run
```

### Forma 2: Linha de Comando (sed)
```bash
# Exemplo: ALPHA de 0.1 para 0.2
sed -i 's/#define ALPHA 0.1/#define ALPHA 0.2/' qlearning_sequential.c
make clean && make all && make run
```

### Forma 3: Criar Variante
```bash
cp qlearning_sequential.c qlearning_teste.c
nano qlearning_teste.c  # Editar a cópia
gcc -o qlearning_teste qlearning_teste.c -lm
./qlearning_teste
```

## 4 Receitas Prontas

### ⚡ Teste Rápido (1 segundo)
```c
#define GRID_ROWS 3
#define GRID_COLS 3
#define ALPHA 0.2
#define GAMMA 0.8
#define EPSILON 0.15
#define NUM_EPISODES 300
#define GOAL_STATE 8      // (2,2)
#define OBSTACLE_STATE 4  // (1,1)
```

### ✓ Padrão Robusto (5 segundos) ← RECOMENDADO
```c
#define GRID_ROWS 4
#define GRID_COLS 4
#define ALPHA 0.1
#define GAMMA 0.9
#define EPSILON 0.1
#define NUM_EPISODES 1000
#define GOAL_STATE 15     // (3,3)
#define OBSTACLE_STATE 5  // (1,1)
```

### 🎯 Grid Grande (30 segundos)
```c
#define GRID_ROWS 8
#define GRID_COLS 8
#define ALPHA 0.08
#define GAMMA 0.92
#define EPSILON 0.08
#define NUM_EPISODES 3000
#define GOAL_STATE 63     // (7,7)
#define OBSTACLE_STATE 9  // (1,1)
```

### 🚀 Aprendizado Rápido (2 segundos)
```c
#define GRID_ROWS 4
#define GRID_COLS 4
#define ALPHA 0.3         // ← RÁPIDO
#define GAMMA 0.95
#define EPSILON 0.05
#define NUM_EPISODES 500
#define GOAL_STATE 15
#define OBSTACLE_STATE 5
```

## Mudando o Tamanho do Grid

⚠️ **IMPORTANTE:** Se mudar GRID_ROWS/GRID_COLS, RECALCULE também:

```
GOAL_STATE = (últimaLinha) × GRID_COLS + (últimaColuna)
OBSTACLE_STATE = linha × GRID_COLS + coluna

Exemplos:
Grid 5×5: GOAL_STATE = 4×5 + 4 = 24
Grid 8×8: GOAL_STATE = 7×8 + 7 = 63
Grid 3×3: GOAL_STATE = 2×3 + 2 = 8
```

E aumente `NUM_EPISODES`:
- Grid 3×3 → 300-500 episódios
- Grid 4×4 → 1000 episódios (padrão)
- Grid 5×5 → 1500 episódios
- Grid 8×8 → 3000 episódios

## Efeitos Importantes

### ALPHA (Taxa de Aprendizado)
| Valor | Efeito |
|-------|--------|
| 0.01 | Muito lento, muito estável |
| 0.05 | Lento, estável |
| **0.1** | **BALANCEADO** |
| 0.2 | Rápido, menos estável |
| 0.5 | Muito rápido, instável |

### GAMMA (Fator de Desconto)
| Valor | Significa |
|-------|----------|
| 0.0 | Ignora futuro (cego) |
| 0.5 | Visão curta (2-3 passos) |
| **0.9** | **BALANCEADO** (longo prazo) |
| 0.95 | Futuro muito importante |

### EPSILON (Exploração)
| Valor | Comportamento |
|-------|--------------|
| 0.0 | Nunca aleatório (greedy puro) |
| **0.1** | **10% aleatório, 90% ganancioso** |
| 0.2 | 20% aleatório |
| 1.0 | 100% aleatório (sem aprendizado) |

### NUM_EPISODES
| Valor | Tempo | Convergência |
|-------|-------|--------------|
| 100 | <1s | Ruim |
| 300 | 2s | Parcial |
| **1000** | **5s** | **Excelente** |
| 2000 | 10s | Perfeita |
| 5000 | 25s | Overfit |

## Dicas Rápidas

✅ **Comece com a Receita 2 (Padrão)**

✅ **Para convergência mais rápida:**
- Aumente ALPHA (0.1 → 0.2)
- Diminua NUM_EPISODES (1000 → 500)
- Diminua EPSILON (0.1 → 0.05)

✅ **Para resultados mais estáveis:**
- Diminua ALPHA (0.1 → 0.05)
- Aumente NUM_EPISODES (1000 → 2000)
- Aumente GAMMA (0.9 → 0.95)

✅ **Sempre recompile após mudanças:**
```bash
make clean && make all && make run
```

✅ **Teste os testes para garantir tudo funciona:**
```bash
make test  # Deve passar em 44 testes
```

## Arquivos de Referência

- `RELATORIO.org` → Detalhes completos (32KB)
- `PARAMETROS.org` → Guia formatado em Org Mode
- `COMO_MODIFICAR.sh` → Exemplos executáveis
- `qlearning_sequential.c` → Código comentado (600+ linhas)
- `test_qlearning.c` → Testes automatizados (44 testes)

---

**Pronto!** Agora você pode modificar qualquer parâmetro facilmente! 🎉
