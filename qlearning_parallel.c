/**
 * =============================================================================
 * Q-LEARNING SEQUENCIAL EM C
 * =============================================================================
 * 
 * Este código implementa o algoritmo Q-Learning para aprendizado por reforço.
 * O ambiente usado é um Grid World (mundo em grade) onde um agente deve
 * aprender a encontrar o caminho até um objetivo.
 * 
 * CONCEITOS DO Q-LEARNING:
 * ------------------------
 * - Q-Table: Tabela que armazena o valor esperado de recompensa para cada
 *   par (estado, ação). Q(s, a) = recompensa esperada ao tomar ação 'a' no estado 's'
 * 
 * - Equação de Bellman (atualização Q):
 *   Q(s, a) = Q(s, a) + α * [R + γ * max(Q(s', a')) - Q(s, a)]
 *   
 *   Onde:
 *   - α (alpha): Taxa de aprendizado (0 < α ≤ 1)
 *   - γ (gamma): Fator de desconto (0 ≤ γ < 1) - importância de recompensas futuras
 *   - R: Recompensa recebida
 *   - s': Próximo estado
 *   - max(Q(s', a')): Melhor valor Q possível no próximo estado
 * 
 * - Política ε-greedy: Com probabilidade ε explora (ação aleatória),
 *   caso contrário explota (melhor ação conhecida)
 * 
 * AMBIENTE GRID WORLD:
 * --------------------
 * O agente começa em uma posição inicial e deve chegar ao objetivo.
 * 
 *   +---+---+---+---+
 *   | S |   |   |   |   S = Start (início)
 *   +---+---+---+---+   G = Goal (objetivo)
 *   |   | X |   |   |   X = Obstáculo
 *   +---+---+---+---+
 *   |   |   |   |   |
 *   +---+---+---+---+
 *   |   |   |   | G |
 *   +---+---+---+---+
 * 
 * Ações: 0=CIMA, 1=BAIXO, 2=ESQUERDA, 3=DIREITA
 * 
 * Autor: Trabalho de Programação Paralela
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <omp.h>

/* =============================================================================
 * DEFINIÇÕES E CONSTANTES
 * =============================================================================
 */

/* Dimensões do Grid World */
#define GRID_ROWS 4      /* Número de linhas do grid */
#define GRID_COLS 4      /* Número de colunas do grid */
#define NUM_STATES (GRID_ROWS * GRID_COLS)  /* Total de estados = 16 */
#define NUM_ACTIONS 4    /* Ações possíveis: cima, baixo, esquerda, direita */

/* Identificadores das ações */
#define ACTION_UP    0
#define ACTION_DOWN  1
#define ACTION_LEFT  2
#define ACTION_RIGHT 3

/* Hiperparâmetros do Q-Learning */
#define ALPHA 0.1        /* Taxa de aprendizado: quão rápido atualiza Q */
#define GAMMA 0.9        /* Fator de desconto: importância de recompensas futuras */
#define EPSILON 0.1      /* Taxa de exploração: probabilidade de ação aleatória */
#define NUM_EPISODES 1000 /* Número de episódios de treinamento */
#define MAX_STEPS 100    /* Máximo de passos por episódio (evita loops infinitos) */

/* Posições especiais no grid */
#define START_STATE 0    /* Estado inicial (posição 0,0) */
#define GOAL_STATE 15    /* Estado objetivo (posição 3,3) */
#define OBSTACLE_STATE 5 /* Estado com obstáculo (posição 1,1) */

/* Recompensas */
#define REWARD_GOAL 100.0     /* Recompensa ao atingir o objetivo */
#define REWARD_OBSTACLE -100.0 /* Penalidade ao bater em obstáculo */
#define REWARD_STEP -1.0      /* Pequena penalidade por passo (incentiva caminho curto) */

/* =============================================================================
 * ESTRUTURA DE DADOS
 * =============================================================================
 */

/**
 * Estrutura que representa o ambiente e o agente Q-Learning
 */
typedef struct {
    double q_table[NUM_STATES][NUM_ACTIONS];  /* Tabela Q: valores Q(s,a) */
    int grid[GRID_ROWS][GRID_COLS];           /* Representação do grid */
    int current_state;                         /* Estado atual do agente */
} QLearning;

/* =============================================================================
 * FUNÇÕES AUXILIARES
 * =============================================================================
 */

/**
 * Converte coordenadas (linha, coluna) para índice de estado único
 * 
 * O grid é "achatado" em um array 1D para facilitar indexação na Q-table.
 * Exemplo: posição (1,2) no grid 4x4 = 1*4 + 2 = 6
 */
int coords_to_state(int row, int col) {
    return row * GRID_COLS + col;
}

/**
 * Converte índice de estado para coordenadas (linha, coluna)
 * 
 * Operação inversa de coords_to_state
 */
void state_to_coords(int state, int *row, int *col) {
    *row = state / GRID_COLS;
    *col = state % GRID_COLS;
}

/**
 * Gera número aleatório entre 0 e 1
 * 
 * Usado para decisão ε-greedy (explorar vs explotar)
 */
double random_double() {
    return (double)rand() / RAND_MAX;
}

/**
 * Gera número inteiro aleatório entre 0 e max-1
 * 
 * Usado para selecionar ação aleatória durante exploração
 */
int random_int(int max) {
    return rand() % max;
}

/* =============================================================================
 * FUNÇÕES DO Q-LEARNING
 * =============================================================================
 */

/**
 * Inicializa a estrutura Q-Learning
 * 
 * - Zera toda a Q-table (valores iniciais neutros)
 * - Configura o grid com obstáculos
 * - Define o estado inicial do agente
 */
void qlearning_init(QLearning *ql) {
    int i, j;
    
    /* Inicializa Q-table com zeros */
    /* Todos os pares (estado, ação) começam com valor 0 */
    #pragma omp parallel for
    for (i = 0; i < NUM_STATES; i++) {
        for (j = 0; j < NUM_ACTIONS; j++) {
            ql->q_table[i][j] = 0.0;
        }
    }
    
    /* Inicializa o grid */
    /* 0 = célula livre, 1 = obstáculo, 2 = objetivo */
    #pragma omp parallel for
    for (i = 0; i < GRID_ROWS; i++) {
        for (j = 0; j < GRID_COLS; j++) {
            ql->grid[i][j] = 0;  /* Todas as células iniciam livres */
        }
    }
    
    /* Configura posições especiais */
    ql->grid[1][1] = 1;  /* Obstáculo na posição (1,1) */
    ql->grid[3][3] = 2;  /* Objetivo na posição (3,3) */
    
    /* Agente começa no estado inicial */
    ql->current_state = START_STATE;
}

/**
 * Verifica se um estado é válido (dentro dos limites do grid)
 * 
 * Retorna 1 se válido, 0 se inválido
 */
int is_valid_state(int row, int col) {
    return (row >= 0 && row < GRID_ROWS && col >= 0 && col < GRID_COLS);
}

/**
 * Calcula o próximo estado dado o estado atual e uma ação
 * 
 * Se a ação levar para fora do grid, o agente permanece no mesmo estado.
 * 
 * Retorna: índice do próximo estado
 */
int get_next_state(int current_state, int action) {
    int row, col, new_row, new_col;
    
    /* Converte estado atual para coordenadas */
    state_to_coords(current_state, &row, &col);
    
    /* Calcula nova posição baseada na ação */
    new_row = row;
    new_col = col;
    
    switch (action) {
        case ACTION_UP:
            new_row = row - 1;  /* Move para cima (diminui linha) */
            break;
        case ACTION_DOWN:
            new_row = row + 1;  /* Move para baixo (aumenta linha) */
            break;
        case ACTION_LEFT:
            new_col = col - 1;  /* Move para esquerda (diminui coluna) */
            break;
        case ACTION_RIGHT:
            new_col = col + 1;  /* Move para direita (aumenta coluna) */
            break;
    }
    
    /* Verifica se nova posição é válida */
    if (is_valid_state(new_row, new_col)) {
        return coords_to_state(new_row, new_col);
    }
    
    /* Se inválida, permanece no mesmo estado */
    return current_state;
}

/**
 * Calcula a recompensa para transição de estado
 * 
 * Sistema de recompensas:
 * - Atingir objetivo: +100 (recompensa máxima)
 * - Bater em obstáculo: -100 (penalidade severa)
 * - Qualquer outro passo: -1 (incentiva caminho curto)
 */
double get_reward(int next_state) {
    if (next_state == GOAL_STATE) {
        return REWARD_GOAL;       /* Chegou ao objetivo! */
    } else if (next_state == OBSTACLE_STATE) {
        return REWARD_OBSTACLE;   /* Bateu no obstáculo */
    }
    return REWARD_STEP;           /* Passo normal */
}

/**
 * Verifica se o episódio terminou
 * 
 * O episódio termina quando o agente atinge o objetivo
 */
int is_terminal_state(int state) {
    return (state == GOAL_STATE);
}

/**
 * Seleciona ação usando política ε-greedy
 * 
 * Esta é a estratégia de balanceamento exploração/explotação:
 * - Com probabilidade ε: escolhe ação ALEATÓRIA (exploração)
 *   Isso permite descobrir novos caminhos
 * - Com probabilidade (1-ε): escolhe MELHOR ação conhecida (explotação)
 *   Isso usa o conhecimento já adquirido    
 */
int select_action(QLearning *ql, int state) {
    int action, best_action;
    double best_value;
    
    /* Com probabilidade ε, explora (ação aleatória) */
    if (random_double() < EPSILON) {
        return random_int(NUM_ACTIONS);
    }
    
    /* Caso contrário, explota (melhor ação conhecida) */
    /* Encontra a ação com maior valor Q para o estado atual */
    best_action = 0;
    best_value = ql->q_table[state][0];
    
    /*
    Possui duas regiões críticas: best_value e best_action.
    Foi criado duas variáveis locais local_best_value e local_best_action onde
    cada thread terá o seu melhor valor local e melhor ação local. No final, atualiza a região crítica.
    */
    #pragma omp parallel
    {

        double local_best_value = ql->q_table[state][0];
        int local_best_action = 0;

        // Usa-se o nowait para evitar a sincronização de threads depois do loop.
        // Faz com que as threads possam atualizar de forma segura o melhor valor globa e o melhor ação global
        #pragma omp for nowait
        for (action = 1; action < NUM_ACTIONS; action++) {
            if (ql->q_table[state][action] > best_value) {
                best_value = ql->q_table[state][action];
                best_action = action;
            }
        }

        #pragma omp critical
        {
            if(local_best_value > best_value){
                    best_value = local_best_value;
                    best_action = local_best_action;
            }
        }

    }
    
    return best_action;
}

/**
 * Encontra o valor máximo Q para um dado estado
 * 
 * max Q(s', a') - usado na equação de Bellman
 * Representa a melhor recompensa futura esperada a partir do estado s'
 */
double get_max_q_value(QLearning *ql, int state) {
    int action;
    double max_value = ql->q_table[state][0];
    
    for (action = 1; action < NUM_ACTIONS; action++) {
        if (ql->q_table[state][action] > max_value) {
            max_value = ql->q_table[state][action];
        }
    }
    
    return max_value;
}

/**
 * Atualiza o valor Q usando a equação de Bellman
 * 
 * Q(s,a) = Q(s,a) + α * [R + γ * max Q(s',a') - Q(s,a)]
 * 
 * Esta é a ESSÊNCIA do Q-Learning:
 * - O termo [R + γ * max Q(s',a')] é o "target" (valor alvo)
 * - O termo [target - Q(s,a)] é o "erro temporal" (TD error)
 * - Multiplicamos pelo learning rate α para suavizar a atualização
 */
void update_q_value(QLearning *ql, int state, int action, 
                    double reward, int next_state) {
    double current_q = ql->q_table[state][action];
    double max_next_q = get_max_q_value(ql, next_state);
    
    /* Equação de Bellman para Q-Learning */
    /* target = reward + gamma * max_next_q */
    /* new_q = current_q + alpha * (target - current_q) */
    ql->q_table[state][action] = current_q + 
        ALPHA * (reward + GAMMA * max_next_q - current_q);
}

/**
 * Executa um episódio completo de treinamento
 * 
 * Um episódio consiste em:
 * 1. Começar no estado inicial
 * 2. Repetir até atingir estado terminal ou máximo de passos:
 *    a. Selecionar ação (ε-greedy)
 *    b. Executar ação, observar próximo estado e recompensa
 *    c. Atualizar Q-table
 *    d. Mover para próximo estado
 * 
 * Retorna: recompensa total acumulada no episódio
 */
double run_episode(QLearning *ql) {
    int step, state, action, next_state;
    double reward, total_reward;
    
    /* Reinicia no estado inicial */
    state = START_STATE;
    total_reward = 0.0;
    
    /* Loop principal do episódio */
    for (step = 0; step < MAX_STEPS; step++) {
        /* 1. Seleciona ação usando política ε-greedy */
        action = select_action(ql, state);
        
        /* 2. Executa ação e observa resultado */
        next_state = get_next_state(state, action);
        reward = get_reward(next_state);
        
        /* 3. Atualiza Q-table com a nova experiência */
        update_q_value(ql, state, action, reward, next_state);
        
        /* 4. Acumula recompensa */
        total_reward += reward;
        
        /* 5. Verifica se atingiu estado terminal */
        if (is_terminal_state(next_state)) {
            break;  /* Episódio termina ao atingir objetivo */
        }
        
        /* 6. Move para o próximo estado */
        state = next_state;
    }
    
    return total_reward;
}

/**
 * Treina o agente Q-Learning por múltiplos episódios
 * 
 * O treinamento consiste em executar muitos episódios para que
 * o agente aprenda a política ótima através de tentativa e erro.
 */
void train(QLearning *ql, int num_episodes, int verbose) {
    int episode;
    double total_reward;
    double avg_reward = 0.0;
    
    printf("Iniciando treinamento com %d episodios...\n", num_episodes);
    printf("Hiperparametros: alpha=%.2f, gamma=%.2f, epsilon=%.2f\n\n", 
           ALPHA, GAMMA, EPSILON);
    
    for (episode = 0; episode < num_episodes; episode++) {
        total_reward = run_episode(ql);
        avg_reward += total_reward;
        
        /* Imprime progresso a cada 100 episódios */
        if (verbose && (episode + 1) % 100 == 0) {
            printf("Episodio %4d | Recompensa media (ultimos 100): %.2f\n",
                   episode + 1, avg_reward / 100.0);
            avg_reward = 0.0;
        }
    }
    
    printf("\nTreinamento concluido!\n");
}

/* =============================================================================
 * FUNÇÕES DE VISUALIZAÇÃO
 * =============================================================================
 */

/**
 * Converte código de ação para string legível
 */
const char* action_to_string(int action) {
    switch (action) {
        case ACTION_UP:    return "CIMA";
        case ACTION_DOWN:  return "BAIXO";
        case ACTION_LEFT:  return "ESQ";
        case ACTION_RIGHT: return "DIR";
        default:           return "???";
    }
}

/**
 * Imprime a Q-table formatada
 * 
 * Mostra os valores Q(s,a) para cada estado e ação
 */
void print_q_table(QLearning *ql) {
    int state, action;
    
    printf("\n=== Q-TABLE ===\n");
    printf("Estado |   CIMA   |  BAIXO   |   ESQ    |   DIR    | Melhor\n");
    printf("-------|----------|----------|----------|----------|-------\n");
    
    for (state = 0; state < NUM_STATES; state++) {
        int row, col, best_action = 0;
        double best_value = ql->q_table[state][0];
        
        state_to_coords(state, &row, &col);
        printf("(%d,%d)  |", row, col);
        
        for (action = 0; action < NUM_ACTIONS; action++) {
            printf(" %8.2f |", ql->q_table[state][action]);
            if (ql->q_table[state][action] > best_value) {
                best_value = ql->q_table[state][action];
                best_action = action;
            }
        }
        printf(" %s\n", action_to_string(best_action));
    }
}

/**
 * Imprime a política aprendida como um grid visual
 * 
 * Mostra a melhor ação em cada estado usando setas:
 * ↑ (cima), ↓ (baixo), ← (esquerda), → (direita)
 */
void print_policy(QLearning *ql) {
    int row, col, state, action;
    double best_value;
    int best_action;
    char symbols[4] = {'^', 'v', '<', '>'};  /* Símbolos ASCII para ações */
    
    printf("\n=== POLITICA APRENDIDA ===\n");
    printf("(^ = cima, v = baixo, < = esq, > = dir)\n\n");
    
    for (row = 0; row < GRID_ROWS; row++) {
        printf("+---");
        for (col = 1; col < GRID_COLS; col++) printf("+---");
        printf("+\n");
        
        for (col = 0; col < GRID_COLS; col++) {
            state = coords_to_state(row, col);
            
            if (state == GOAL_STATE) {
                printf("| G ");  /* Goal */
            } else if (state == OBSTACLE_STATE) {
                printf("| X ");  /* Obstáculo */
            } else {
                /* Encontra melhor ação para este estado */
                best_action = 0;
                best_value = ql->q_table[state][0];
                for (action = 1; action < NUM_ACTIONS; action++) {
                    if (ql->q_table[state][action] > best_value) {
                        best_value = ql->q_table[state][action];
                        best_action = action;
                    }
                }
                printf("| %c ", symbols[best_action]);
            }
        }
        printf("|\n");
    }
    printf("+---+---+---+---+\n");
}

/**
 * Simula e mostra o caminho que o agente seguiria após treinamento
 * 
 * Usa apenas explotação (sempre a melhor ação), sem exploração
 */
void demonstrate_path(QLearning *ql) {
    int state = START_STATE;
    int step = 0;
    int action, best_action;
    double best_value;
    int row, col;
    
    printf("\n=== DEMONSTRACAO DO CAMINHO ===\n");
    printf("Caminho do agente do inicio (0,0) ate o objetivo (3,3):\n\n");
    
    while (state != GOAL_STATE && step < MAX_STEPS) {
        state_to_coords(state, &row, &col);
        
        /* Encontra melhor ação (sem exploração) */
        best_action = 0;
        best_value = ql->q_table[state][0];
        for (action = 1; action < NUM_ACTIONS; action++) {
            if (ql->q_table[state][action] > best_value) {
                best_value = ql->q_table[state][action];
                best_action = action;
            }
        }
        
        printf("Passo %2d: Estado (%d,%d) -> Acao: %s\n", 
               step + 1, row, col, action_to_string(best_action));
        
        state = get_next_state(state, best_action);
        step++;
    }
    
    if (state == GOAL_STATE) {
        state_to_coords(state, &row, &col);
        printf("Passo %2d: Estado (%d,%d) -> OBJETIVO ALCANCADO!\n", 
               step + 1, row, col);
        printf("\nAgente encontrou o caminho em %d passos.\n", step);
    } else {
        printf("\nAgente nao conseguiu encontrar o objetivo em %d passos.\n", step);
    }
}

/* =============================================================================
 * FUNÇÕES DE EXPORTAÇÃO (para testes)
 * =============================================================================
 */

/**
 * Obtém a melhor ação para um estado (usada nos testes)
 */
int get_best_action(QLearning *ql, int state) {
    int action, best_action = 0;
    double best_value = ql->q_table[state][0];
    
    for (action = 1; action < NUM_ACTIONS; action++) {
        if (ql->q_table[state][action] > best_value) {
            best_value = ql->q_table[state][action];
            best_action = action;
        }
    }
    
    return best_action;
}

/**
 * Simula o caminho e retorna o número de passos até o objetivo
 * Retorna -1 se não conseguir atingir o objetivo
 */
int simulate_path(QLearning *ql) {
    int state = START_STATE;
    int step = 0;
    int action;
    
    while (state != GOAL_STATE && step < MAX_STEPS) {
        action = get_best_action(ql, state);
        state = get_next_state(state, action);
        step++;
    }
    
    if (state == GOAL_STATE) {
        return step;
    }
    return -1;  /* Não encontrou o objetivo */
}

/* =============================================================================
 * FUNÇÃO PRINCIPAL
 * =============================================================================
 */

int main(void) {
    QLearning ql;
    
    /* Inicializa gerador de números aleatórios com seed fixa para reprodutibilidade */
    /* Use time(NULL) para resultados diferentes a cada execução */
    srand(42);
    
    printf("============================================================\n");
    printf("       Q-LEARNING SEQUENCIAL - GRID WORLD 4x4\n");
    printf("============================================================\n\n");
    
    /* Fase 1: Inicialização */
    printf("Inicializando ambiente...\n");
    qlearning_init(&ql);
    printf("Grid World criado:\n");
    printf("  - Inicio: (0,0)\n");
    printf("  - Objetivo: (3,3)\n");
    printf("  - Obstaculo: (1,1)\n\n");
    
    /* Fase 2: Treinamento */
    train(&ql, NUM_EPISODES, 1);
    
    /* Fase 3: Resultados */
    print_q_table(&ql);
    print_policy(&ql);
    demonstrate_path(&ql);
    
    printf("\n============================================================\n");
    printf("                    FIM DA EXECUCAO\n");
    printf("============================================================\n");
    
    return 0;
}
