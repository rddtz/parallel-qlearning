#!/bin/bash
# =============================================================================
# GUIA: Como Modificar Parâmetros do Q-Learning
# =============================================================================
# 
# Este arquivo documenta todos os parâmetros que podem ser alterados
# no código qlearning_sequential.c e o impacto de cada mudança.
#
# =============================================================================

## PARÂMETROS DO GRID (Tamanho do Ambiente)
## ============================================

### 1. Tamanho do Grid
#
# PADRÃO:
#   #define GRID_ROWS 4
#   #define GRID_COLS 4
#
# PARA GRID 5x5:
#   #define GRID_ROWS 5
#   #define GRID_COLS 5
#
# PARA GRID 8x8:
#   #define GRID_ROWS 8
#   #define GRID_COLS 8
#
# PARA GRID 10x10:
#   #define GRID_ROWS 10
#   #define GRID_COLS 10
#
# ⚠️  IMPORTANTE: Depois de mudar o tamanho do grid, é NECESSÁRIO 
#     ajustar também:
#     - GOAL_STATE (novo índice do canto inferior direito)
#     - OBSTACLE_STATE (novo índice do obstáculo)
#     - NUM_EPISODES (pode precisar de mais episódios)
#
# EXEMPLO para grid 5x5:
#   #define GRID_ROWS 5
#   #define GRID_COLS 5
#   #define GOAL_STATE 24        // (4,4) = 4*5 + 4 = 24
#   #define OBSTACLE_STATE 6     // (1,1) = 1*5 + 1 = 6
#   #define NUM_EPISODES 2000    // Mais episódios para convergir
#


## HIPERPARÂMETROS DO Q-LEARNING
## ===============================

### 2. ALPHA (Taxa de Aprendizado)
#
# PADRÃO: #define ALPHA 0.1
#
# EFEITOS:
#   Alpha = 0.05    → Aprendizado LENTO mas ESTÁVEL
#   Alpha = 0.1     → BALANCEADO (recomendado)
#   Alpha = 0.2     → Aprendizado RÁPIDO mas INSTÁVEL
#   Alpha = 0.01    → Muito conservador (pode não convergir)
#   Alpha = 0.5     → Muito agressivo (pula estimativas)
#
# ALTERAR PARA:
#   #define ALPHA 0.05    // Aprendizado mais lento
#   #define ALPHA 0.2     // Aprendizado mais rápido
#   #define ALPHA 0.01    // Muito conservador
#


### 3. GAMMA (Fator de Desconto)
#
# PADRÃO: #define GAMMA 0.9
#
# EFEITOS:
#   Gamma = 0.0     → Ignora recompensas futuras (cego)
#   Gamma = 0.5     → Valora recompensas futuras moderadamente
#   Gamma = 0.9     → BALANCEADO (recomendado)
#   Gamma = 0.99    → Valoriza muito recompensas futuras
#   Gamma → 1.0     → Diverge (instável)
#
# ALTERAR PARA:
#   #define GAMMA 0.5     // Menos consideração do futuro
#   #define GAMMA 0.99    // Mais consideração do futuro
#   #define GAMMA 0.8     // Intermediário
#


### 4. EPSILON (Taxa de Exploração)
#
# PADRÃO: #define EPSILON 0.1 (10% exploração, 90% explotação)
#
# EFEITOS:
#   Epsilon = 0.0     → Nunca explora (greedy puro)
#   Epsilon = 0.05    → Pouca exploração (5%)
#   Epsilon = 0.1     → BALANCEADO (recomendado) (10%)
#   Epsilon = 0.3     → Muita exploração (30%)
#   Epsilon = 1.0     → Sempre aleatório (totalmente cego)
#
# ALTERAR PARA:
#   #define EPSILON 0.05   // Menos exploração
#   #define EPSILON 0.2    // Mais exploração
#   #define EPSILON 0.01   // Quase greedy puro
#


## PARÂMETROS DE TREINAMENTO
## ===========================

### 5. NUM_EPISODES (Número de Episódios)
#
# PADRÃO: #define NUM_EPISODES 1000
#
# EFEITOS:
#   100 episódios    → Rápido mas pode não convergir
#   500 episódios    → Convergência parcial
#   1000 episódios   → BALANCEADO (recomendado)
#   5000 episódios   → Muito bem convergido (lento)
#   10000 episódios  → Quase perfeito (muito lento)
#
# ALTERAR PARA:
#   #define NUM_EPISODES 500    // Teste rápido
#   #define NUM_EPISODES 2000   // Convergência completa
#   #define NUM_EPISODES 5000   // Overtraining
#


### 6. MAX_STEPS (Limite de Passos por Episódio)
#
# PADRÃO: #define MAX_STEPS 100
#
# EFEITOS:
#   MAX_STEPS = 10   → Episódios muito curtos (pode não chegar objetivo)
#   MAX_STEPS = 50   → Razoável
#   MAX_STEPS = 100  → BALANCEADO (recomendado)
#   MAX_STEPS = 200  → Permite caminhos mais longos
#
# ALTERAR PARA:
#   #define MAX_STEPS 50    // Limite mais curto
#   #define MAX_STEPS 200   // Limite mais longo
#


## SISTEMA DE RECOMPENSAS
## =======================

### 7. REWARD_GOAL (Recompensa ao Atingir Objetivo)
#
# PADRÃO: #define REWARD_GOAL 100.0
#
# EFEITOS:
#   REWARD_GOAL = 10.0    → Fraca motivação
#   REWARD_GOAL = 50.0    → Moderada
#   REWARD_GOAL = 100.0   → BALANCEADO (recomendado)
#   REWARD_GOAL = 1000.0  → Forte motivação
#
# ALTERAR PARA:
#   #define REWARD_GOAL 50.0    // Menos motivação
#   #define REWARD_GOAL 200.0   // Mais motivação
#


### 8. REWARD_OBSTACLE (Penalidade ao Bater em Obstáculo)
#
# PADRÃO: #define REWARD_OBSTACLE -100.0
#
# EFEITOS:
#   REWARD_OBSTACLE = -10.0   → Fraca punição
#   REWARD_OBSTACLE = -50.0   → Moderada
#   REWARD_OBSTACLE = -100.0  → BALANCEADO (recomendado)
#   REWARD_OBSTACLE = -1000.0 → Forte punição
#
# ALTERAR PARA:
#   #define REWARD_OBSTACLE -50.0   // Menos punição
#   #define REWARD_OBSTACLE -200.0  // Mais punição
#


### 9. REWARD_STEP (Penalidade por Cada Passo)
#
# PADRÃO: #define REWARD_STEP -1.0
#
# EFEITOS:
#   REWARD_STEP = 0.0      → Sem penalidade (incentiva loops)
#   REWARD_STEP = -0.5     → Fraca penalidade
#   REWARD_STEP = -1.0     → BALANCEADO (incentiva caminho curto)
#   REWARD_STEP = -5.0     → Forte penalidade (favorece 1 passo)
#   REWARD_STEP = -10.0    → Muito forte (só pensa em comprimento)
#
# ALTERAR PARA:
#   #define REWARD_STEP -0.1    // Menos incentivo de caminho curto
#   #define REWARD_STEP -5.0    // Mais incentivo de caminho curto
#


## POSIÇÕES ESPECIAIS NO GRID
## ============================

### 10. Posições dos Objetos
#
# Para calcular os índices:
#   estado = linha × GRID_COLS + coluna
#
# GRID 4x4 (PADRÃO):
#   START_STATE = 0        (posição 0,0 = 0×4 + 0 = 0)
#   GOAL_STATE = 15        (posição 3,3 = 3×4 + 3 = 15)
#   OBSTACLE_STATE = 5     (posição 1,1 = 1×4 + 1 = 5)
#
# GRID 5x5:
#   GOAL_STATE = 24        (posição 4,4 = 4×5 + 4 = 24)
#   OBSTACLE_STATE = 6     (posição 1,1 = 1×5 + 1 = 6)
#
# GRID 8x8:
#   GOAL_STATE = 63        (posição 7,7 = 7×8 + 7 = 63)
#   OBSTACLE_STATE = 9     (posição 1,1 = 1×8 + 1 = 9)
#


## EXEMPLOS DE CONFIGURAÇÕES
## ===========================

echo "=== EXEMPLO 1: Grid Pequeno e Aprendizado Rápido ==="
echo "Para testar rapidamente:"
cat << 'EOF'
#define GRID_ROWS 3
#define GRID_COLS 3
#define ALPHA 0.3        // Aprendizado rápido
#define GAMMA 0.8        // Menos futuro
#define EPSILON 0.2      // Mais exploração
#define NUM_EPISODES 300 // Menos episódios
#define MAX_STEPS 50
#define GOAL_STATE 8      // (2,2) = 2×3 + 2 = 8
#define OBSTACLE_STATE 4  // (1,1) = 1×3 + 1 = 4
EOF

echo ""
echo "=== EXEMPLO 2: Grid Grande e Aprendizado Conservador ==="
echo "Para resultados robustos:"
cat << 'EOF'
#define GRID_ROWS 6
#define GRID_COLS 6
#define ALPHA 0.05       // Aprendizado lento mas estável
#define GAMMA 0.95       // Muito futuro
#define EPSILON 0.05     // Pouca exploração
#define NUM_EPISODES 5000 // Muitos episódios
#define MAX_STEPS 200
#define GOAL_STATE 35     // (5,5) = 5×6 + 5 = 35
#define OBSTACLE_STATE 7  // (1,1) = 1×6 + 1 = 7
EOF

echo ""
echo "=== EXEMPLO 3: Teste de Convergência Rápida ==="
echo "Para ver convergência em segundos:"
cat << 'EOF'
#define GRID_ROWS 4
#define GRID_COLS 4
#define ALPHA 0.2        // Rápido
#define GAMMA 0.8        // Simples
#define EPSILON 0.15     // Balanceado
#define NUM_EPISODES 200 // Poucos episódios
#define MAX_STEPS 50
// Mesmos estados: 0, 5, 15
EOF

echo ""
echo "=== EXEMPLO 4: Desafio (Grid Grande) ==="
echo "Para grid maior:"
cat << 'EOF'
#define GRID_ROWS 10
#define GRID_COLS 10
#define ALPHA 0.1
#define GAMMA 0.9
#define EPSILON 0.1
#define NUM_EPISODES 2000  // Grid maior = mais episódios
#define MAX_STEPS 150
#define GOAL_STATE 99      // (9,9) = 9×10 + 9 = 99
#define OBSTACLE_STATE 11  // (1,1) = 1×10 + 1 = 11
// Adicionar mais obstáculos em qlearning_init()
EOF


## COMO MODIFICAR O CÓDIGO
## ========================

echo ""
echo "=== PASSO-A-PASSO PARA MUDAR PARÂMETROS ==="
cat << 'EOF'

1. ABRA O ARQUIVO:
   $ gedit qlearning_sequential.c
   ou
   $ vim qlearning_sequential.c
   ou
   $ nano qlearning_sequential.c

2. VÁ PARA A SEÇÃO DE #define (linhas 60-86)

3. ALTERE OS VALORES QUE DESEJAR:
   Exemplo: mudar GRID_ROWS de 4 para 5
   
   ANTES:  #define GRID_ROWS 4
   DEPOIS: #define GRID_ROWS 5

4. SE MUDAR TAMANHO DO GRID:
   - Recalcule GOAL_STATE
   - Recalcule OBSTACLE_STATE
   - Aumente NUM_EPISODES

5. SALVE O ARQUIVO (Ctrl+S no gedit/nano)

6. RECOMPILE:
   $ make clean
   $ make all

7. EXECUTE:
   $ make run

EOF


## EXEMPLOS COM ALTERAÇÕES CONCRETAS
## ==================================

echo ""
echo "=== EXEMPLO PRÁTICO 1: Mudar para Grid 5x5 ==="
cat << 'EOF'

ARQUIVO: qlearning_sequential.c
LINHAS A MUDAR:

60:  #define GRID_ROWS 4  →  #define GRID_ROWS 5
61:  #define GRID_COLS 4  →  #define GRID_COLS 5

80:  #define GOAL_STATE 15   →  #define GOAL_STATE 24
     Cálculo: (4,4) = 4×5 + 4 = 24

81:  #define OBSTACLE_STATE 5  →  #define OBSTACLE_STATE 6
     Cálculo: (1,1) = 1×5 + 1 = 6

75:  #define NUM_EPISODES 1000  →  #define NUM_EPISODES 1500
     (grid maior = mais episódios)

DEPOIS:
$ make clean && make all && make run

EOF

echo ""
echo "=== EXEMPLO PRÁTICO 2: Aprendizado Muito Rápido ==="
cat << 'EOF'

Para testar a convergência rapidamente:

72:  #define ALPHA 0.1   →  #define ALPHA 0.3
74:  #define EPSILON 0.1  →  #define EPSILON 0.2
75:  #define NUM_EPISODES 1000  →  #define NUM_EPISODES 200

Resultado: Treinamento em 1 segundo em vez de 5 segundos

EOF

echo ""
echo "=== EXEMPLO PRÁTICO 3: Aumentar Dificuldade ==="
cat << 'EOF'

Para tornar mais desafiador (mais episódios, menos exploração):

74:  #define EPSILON 0.1  →  #define EPSILON 0.02
72:  #define ALPHA 0.1    →  #define ALPHA 0.05
75:  #define NUM_EPISODES 1000  →  #define NUM_EPISODES 3000

Resultado: Agente precisa aprender de forma mais cuidadosa

EOF


## RESUMO RÁPIDO
## =============

echo ""
echo "=== TABELA DE REFERÊNCIA RÁPIDA ==="
cat << 'EOF'

Parâmetro          | Padrão | Aumentar        | Diminuir
-------------------|--------|-----------------|----------
GRID_ROWS/COLS     | 4      | Problema maior  | Problema menor
ALPHA              | 0.1    | Aprendizado +   | Aprendizado -
GAMMA              | 0.9    | Futuro +        | Futuro -
EPSILON            | 0.1    | Exploração +    | Exploração -
NUM_EPISODES       | 1000   | Convergência +  | Teste rápido
MAX_STEPS          | 100    | Caminho longo   | Caminho curto
REWARD_GOAL        | 100    | Motivação +     | Motivação -
REWARD_OBSTACLE    | -100   | Punição +       | Punição -
REWARD_STEP        | -1.0   | Curto +         | Curto -

EOF

echo "FIM DO GUIA"
