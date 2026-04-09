/**
 * =============================================================================
 * TESTES PARA Q-LEARNING SEQUENCIAL
 * =============================================================================
 * 
 * Este arquivo contém testes unitários para validar o funcionamento correto
 * do algoritmo Q-Learning implementado.
 * 
 * Os testes verificam:
 * 1. Inicialização correta da Q-table e ambiente
 * 2. Conversões de coordenadas/estados
 * 3. Cálculo de próximo estado para cada ação
 * 4. Sistema de recompensas
 * 5. Atualização da Q-table (equação de Bellman)
 * 6. Convergência do treinamento (caminho ótimo)
 * 
 * Compilação: gcc -o test_qlearning test_qlearning.c -lm
 * Execução: ./test_qlearning
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* =============================================================================
 * DEFINIÇÕES (copiadas do arquivo principal para testes independentes)
 * =============================================================================
 */

#define GRID_ROWS 4
#define GRID_COLS 4
#define NUM_STATES (GRID_ROWS * GRID_COLS)
#define NUM_ACTIONS 4

#define ACTION_UP    0
#define ACTION_DOWN  1
#define ACTION_LEFT  2
#define ACTION_RIGHT 3

#define ALPHA 0.1
#define GAMMA 0.9
#define EPSILON 0.1
#define NUM_EPISODES 1000
#define MAX_STEPS 100

#define START_STATE 0
#define GOAL_STATE 15
#define OBSTACLE_STATE 5

#define REWARD_GOAL 100.0
#define REWARD_OBSTACLE -100.0
#define REWARD_STEP -1.0

/* Tolerância para comparação de doubles */
#define EPSILON_TOLERANCE 1e-6

/* =============================================================================
 * ESTRUTURA DE DADOS
 * =============================================================================
 */

typedef struct {
    double q_table[NUM_STATES][NUM_ACTIONS];
    int grid[GRID_ROWS][GRID_COLS];
    int current_state;
} QLearning;

/* =============================================================================
 * FUNÇÕES DO Q-LEARNING (copiadas para independência)
 * =============================================================================
 */

int coords_to_state(int row, int col) {
    return row * GRID_COLS + col;
}

void state_to_coords(int state, int *row, int *col) {
    *row = state / GRID_COLS;
    *col = state % GRID_COLS;
}

double random_double() {
    return (double)rand() / RAND_MAX;
}

int random_int(int max) {
    return rand() % max;
}

void qlearning_init(QLearning *ql) {
    int i, j;
    
    for (i = 0; i < NUM_STATES; i++) {
        for (j = 0; j < NUM_ACTIONS; j++) {
            ql->q_table[i][j] = 0.0;
        }
    }
    
    for (i = 0; i < GRID_ROWS; i++) {
        for (j = 0; j < GRID_COLS; j++) {
            ql->grid[i][j] = 0;
        }
    }
    
    ql->grid[1][1] = 1;
    ql->grid[3][3] = 2;
    ql->current_state = START_STATE;
}

int is_valid_state(int row, int col) {
    return (row >= 0 && row < GRID_ROWS && col >= 0 && col < GRID_COLS);
}

int get_next_state(int current_state, int action) {
    int row, col, new_row, new_col;
    
    state_to_coords(current_state, &row, &col);
    new_row = row;
    new_col = col;
    
    switch (action) {
        case ACTION_UP:    new_row = row - 1; break;
        case ACTION_DOWN:  new_row = row + 1; break;
        case ACTION_LEFT:  new_col = col - 1; break;
        case ACTION_RIGHT: new_col = col + 1; break;
    }
    
    if (is_valid_state(new_row, new_col)) {
        return coords_to_state(new_row, new_col);
    }
    return current_state;
}

double get_reward(int next_state) {
    if (next_state == GOAL_STATE) {
        return REWARD_GOAL;
    } else if (next_state == OBSTACLE_STATE) {
        return REWARD_OBSTACLE;
    }
    return REWARD_STEP;
}

int is_terminal_state(int state) {
    return (state == GOAL_STATE);
}

int select_action(QLearning *ql, int state) {
    int action, best_action;
    double best_value;
    
    if (random_double() < EPSILON) {
        return random_int(NUM_ACTIONS);
    }
    
    best_action = 0;
    best_value = ql->q_table[state][0];
    
    for (action = 1; action < NUM_ACTIONS; action++) {
        if (ql->q_table[state][action] > best_value) {
            best_value = ql->q_table[state][action];
            best_action = action;
        }
    }
    
    return best_action;
}

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

void update_q_value(QLearning *ql, int state, int action, 
                    double reward, int next_state) {
    double current_q = ql->q_table[state][action];
    double max_next_q = get_max_q_value(ql, next_state);
    
    ql->q_table[state][action] = current_q + 
        ALPHA * (reward + GAMMA * max_next_q - current_q);
}

double run_episode(QLearning *ql) {
    int step, state, action, next_state;
    double reward, total_reward;
    
    state = START_STATE;
    total_reward = 0.0;
    
    for (step = 0; step < MAX_STEPS; step++) {
        action = select_action(ql, state);
        next_state = get_next_state(state, action);
        reward = get_reward(next_state);
        update_q_value(ql, state, action, reward, next_state);
        total_reward += reward;
        
        if (is_terminal_state(next_state)) {
            break;
        }
        state = next_state;
    }
    
    return total_reward;
}

void train(QLearning *ql, int num_episodes) {
    int episode;
    for (episode = 0; episode < num_episodes; episode++) {
        run_episode(ql);
    }
}

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
    return -1;
}

/* =============================================================================
 * FRAMEWORK DE TESTES SIMPLES
 * =============================================================================
 */

int tests_passed = 0;
int tests_failed = 0;

/**
 * Macro para verificar condição e reportar resultado
 */
#define TEST_ASSERT(condition, message) do { \
    if (condition) { \
        printf("  [PASS] %s\n", message); \
        tests_passed++; \
    } else { \
        printf("  [FAIL] %s\n", message); \
        tests_failed++; \
    } \
} while(0)

/**
 * Compara dois doubles com tolerância
 */
int double_equals(double a, double b) {
    return fabs(a - b) < EPSILON_TOLERANCE;
}

/* =============================================================================
 * TESTES UNITÁRIOS
 * =============================================================================
 */

/**
 * Teste 1: Conversão de coordenadas para estados
 * 
 * Verifica se a função coords_to_state converte corretamente
 * posições (linha, coluna) para índices de estado.
 */
void test_coords_to_state() {
    printf("\n[TESTE] Conversao coordenadas -> estado:\n");
    
    /* Canto superior esquerdo */
    TEST_ASSERT(coords_to_state(0, 0) == 0, "(0,0) -> estado 0");
    
    /* Primeira linha */
    TEST_ASSERT(coords_to_state(0, 3) == 3, "(0,3) -> estado 3");
    
    /* Obstáculo */
    TEST_ASSERT(coords_to_state(1, 1) == 5, "(1,1) -> estado 5 (obstaculo)");
    
    /* Objetivo (canto inferior direito) */
    TEST_ASSERT(coords_to_state(3, 3) == 15, "(3,3) -> estado 15 (objetivo)");
    
    /* Posição intermediária */
    TEST_ASSERT(coords_to_state(2, 1) == 9, "(2,1) -> estado 9");
}

/**
 * Teste 2: Conversão de estados para coordenadas
 * 
 * Verifica se a função state_to_coords converte corretamente
 * índices de estado para posições (linha, coluna).
 */
void test_state_to_coords() {
    int row, col;
    
    printf("\n[TESTE] Conversao estado -> coordenadas:\n");
    
    /* Estado 0 */
    state_to_coords(0, &row, &col);
    TEST_ASSERT(row == 0 && col == 0, "estado 0 -> (0,0)");
    
    /* Estado 5 (obstáculo) */
    state_to_coords(5, &row, &col);
    TEST_ASSERT(row == 1 && col == 1, "estado 5 -> (1,1)");
    
    /* Estado 15 (objetivo) */
    state_to_coords(15, &row, &col);
    TEST_ASSERT(row == 3 && col == 3, "estado 15 -> (3,3)");
    
    /* Estado intermediário */
    state_to_coords(10, &row, &col);
    TEST_ASSERT(row == 2 && col == 2, "estado 10 -> (2,2)");
}

/**
 * Teste 3: Cálculo de próximo estado (transições)
 * 
 * Verifica se as ações movem o agente corretamente pelo grid.
 * Também testa limites (paredes do grid).
 */
void test_get_next_state() {
    printf("\n[TESTE] Calculo de proximo estado (transicoes):\n");
    
    /* A partir do estado 5 (1,1) */
    TEST_ASSERT(get_next_state(5, ACTION_UP) == 1, 
                "estado 5 + CIMA -> estado 1");
    TEST_ASSERT(get_next_state(5, ACTION_DOWN) == 9, 
                "estado 5 + BAIXO -> estado 9");
    TEST_ASSERT(get_next_state(5, ACTION_LEFT) == 4, 
                "estado 5 + ESQUERDA -> estado 4");
    TEST_ASSERT(get_next_state(5, ACTION_RIGHT) == 6, 
                "estado 5 + DIREITA -> estado 6");
    
    /* Teste de limites - bordas do grid */
    /* Estado 0 (0,0) - canto superior esquerdo */
    TEST_ASSERT(get_next_state(0, ACTION_UP) == 0, 
                "estado 0 + CIMA -> estado 0 (parede)");
    TEST_ASSERT(get_next_state(0, ACTION_LEFT) == 0, 
                "estado 0 + ESQUERDA -> estado 0 (parede)");
    
    /* Estado 15 (3,3) - canto inferior direito */
    TEST_ASSERT(get_next_state(15, ACTION_DOWN) == 15, 
                "estado 15 + BAIXO -> estado 15 (parede)");
    TEST_ASSERT(get_next_state(15, ACTION_RIGHT) == 15, 
                "estado 15 + DIREITA -> estado 15 (parede)");
}

/**
 * Teste 4: Sistema de recompensas
 * 
 * Verifica se as recompensas são atribuídas corretamente
 * para cada tipo de estado.
 */
void test_rewards() {
    printf("\n[TESTE] Sistema de recompensas:\n");
    
    /* Recompensa ao atingir objetivo */
    TEST_ASSERT(double_equals(get_reward(GOAL_STATE), REWARD_GOAL), 
                "objetivo -> +100.0");
    
    /* Penalidade ao bater em obstáculo */
    TEST_ASSERT(double_equals(get_reward(OBSTACLE_STATE), REWARD_OBSTACLE), 
                "obstaculo -> -100.0");
    
    /* Passo normal */
    TEST_ASSERT(double_equals(get_reward(0), REWARD_STEP), 
                "estado normal -> -1.0");
    TEST_ASSERT(double_equals(get_reward(10), REWARD_STEP), 
                "outro estado normal -> -1.0");
}

/**
 * Teste 5: Estados terminais
 * 
 * Verifica se a função is_terminal_state identifica
 * corretamente o estado objetivo.
 */
void test_terminal_state() {
    printf("\n[TESTE] Identificacao de estado terminal:\n");
    
    TEST_ASSERT(is_terminal_state(GOAL_STATE) == 1, 
                "estado 15 (objetivo) e terminal");
    TEST_ASSERT(is_terminal_state(0) == 0, 
                "estado 0 NAO e terminal");
    TEST_ASSERT(is_terminal_state(OBSTACLE_STATE) == 0, 
                "estado 5 (obstaculo) NAO e terminal");
}

/**
 * Teste 6: Inicialização da Q-table
 * 
 * Verifica se a Q-table é inicializada com zeros.
 */
void test_qtable_initialization() {
    QLearning ql;
    int i, j;
    int all_zeros = 1;
    
    printf("\n[TESTE] Inicializacao da Q-table:\n");
    
    qlearning_init(&ql);
    
    /* Verifica se todos os valores são zero */
    for (i = 0; i < NUM_STATES && all_zeros; i++) {
        for (j = 0; j < NUM_ACTIONS && all_zeros; j++) {
            if (!double_equals(ql.q_table[i][j], 0.0)) {
                all_zeros = 0;
            }
        }
    }
    
    TEST_ASSERT(all_zeros, "Q-table inicializada com zeros");
    TEST_ASSERT(ql.current_state == START_STATE, 
                "estado inicial correto (0)");
    TEST_ASSERT(ql.grid[1][1] == 1, "obstaculo configurado em (1,1)");
    TEST_ASSERT(ql.grid[3][3] == 2, "objetivo configurado em (3,3)");
}

/**
 * Teste 7: Atualização da Q-table (Equação de Bellman)
 * 
 * Verifica se a equação de Bellman está sendo aplicada corretamente.
 * Q(s,a) = Q(s,a) + α * [R + γ * max(Q(s',a')) - Q(s,a)]
 */
void test_q_update() {
    QLearning ql;
    double expected_value;
    
    printf("\n[TESTE] Atualizacao Q-table (equacao de Bellman):\n");
    
    qlearning_init(&ql);
    
    /* Simula uma atualização simples */
    /* Estado 0, ação DIREITA, leva ao estado 1 com reward -1 */
    /* Q(0, DIR) = 0 + 0.1 * (-1 + 0.9 * 0 - 0) = -0.1 */
    update_q_value(&ql, 0, ACTION_RIGHT, -1.0, 1);
    expected_value = 0.0 + ALPHA * (-1.0 + GAMMA * 0.0 - 0.0);
    TEST_ASSERT(double_equals(ql.q_table[0][ACTION_RIGHT], expected_value), 
                "primeira atualizacao Q(0,DIR) = -0.1");
    
    /* Segunda atualização no mesmo par (s,a) */
    /* Q(0, DIR) = -0.1 + 0.1 * (-1 + 0.9 * 0 - (-0.1)) = -0.19 */
    update_q_value(&ql, 0, ACTION_RIGHT, -1.0, 1);
    expected_value = -0.1 + ALPHA * (-1.0 + GAMMA * 0.0 - (-0.1));
    TEST_ASSERT(double_equals(ql.q_table[0][ACTION_RIGHT], expected_value), 
                "segunda atualizacao Q(0,DIR) = -0.19");
}

/**
 * Teste 8: Encontrar valor máximo Q
 * 
 * Verifica se get_max_q_value retorna corretamente o maior valor Q
 * para um estado.
 */
void test_get_max_q() {
    QLearning ql;
    
    printf("\n[TESTE] Encontrar maximo Q para estado:\n");
    
    qlearning_init(&ql);
    
    /* Configura valores de teste */
    ql.q_table[5][ACTION_UP] = -5.0;
    ql.q_table[5][ACTION_DOWN] = 10.0;
    ql.q_table[5][ACTION_LEFT] = 3.0;
    ql.q_table[5][ACTION_RIGHT] = 7.0;
    
    TEST_ASSERT(double_equals(get_max_q_value(&ql, 5), 10.0), 
                "max Q(5, *) = 10.0");
    
    /* Todos negativos */
    ql.q_table[10][ACTION_UP] = -5.0;
    ql.q_table[10][ACTION_DOWN] = -10.0;
    ql.q_table[10][ACTION_LEFT] = -3.0;
    ql.q_table[10][ACTION_RIGHT] = -7.0;
    
    TEST_ASSERT(double_equals(get_max_q_value(&ql, 10), -3.0), 
                "max Q(10, *) = -3.0 (menor negativo)");
}

/**
 * Teste 9: Seleção de melhor ação
 * 
 * Verifica se get_best_action retorna a ação com maior valor Q.
 */
void test_get_best_action() {
    QLearning ql;
    
    printf("\n[TESTE] Selecao da melhor acao:\n");
    
    qlearning_init(&ql);
    
    /* Configura valores de teste */
    ql.q_table[5][ACTION_UP] = -5.0;
    ql.q_table[5][ACTION_DOWN] = 10.0;
    ql.q_table[5][ACTION_LEFT] = 3.0;
    ql.q_table[5][ACTION_RIGHT] = 7.0;
    
    TEST_ASSERT(get_best_action(&ql, 5) == ACTION_DOWN, 
                "melhor acao em estado 5 = BAIXO (Q=10)");
    
    /* Todos iguais - deve retornar a primeira (índice 0) */
    ql.q_table[6][ACTION_UP] = 5.0;
    ql.q_table[6][ACTION_DOWN] = 5.0;
    ql.q_table[6][ACTION_LEFT] = 5.0;
    ql.q_table[6][ACTION_RIGHT] = 5.0;
    
    TEST_ASSERT(get_best_action(&ql, 6) == ACTION_UP, 
                "valores iguais -> retorna primeira acao (CIMA)");
}

/**
 * Teste 10: Convergência do treinamento
 * 
 * Teste mais importante: verifica se após o treinamento o agente
 * consegue encontrar um caminho válido até o objetivo.
 * 
 * O caminho ótimo do (0,0) ao (3,3) evitando o obstáculo em (1,1)
 * tem no mínimo 6 passos.
 */
void test_training_convergence() {
    QLearning ql;
    int path_length;
    
    printf("\n[TESTE] Convergencia do treinamento:\n");
    
    srand(42);  /* Seed fixa para reprodutibilidade */
    
    qlearning_init(&ql);
    train(&ql, NUM_EPISODES);
    
    path_length = simulate_path(&ql);
    
    /* Verifica se encontrou o objetivo */
    TEST_ASSERT(path_length > 0, 
                "agente encontrou caminho ate o objetivo");
    
    /* Verifica se o caminho é razoável (não muito longo) */
    /* O caminho ótimo é 6 passos, permitimos até 10 para variações */
    TEST_ASSERT(path_length <= 10, 
                "caminho tem comprimento razoavel (<=10 passos)");
    
    /* Verifica se as ações nos estados críticos fazem sentido */
    /* Estado 0 (início): deve ir para baixo ou direita */
    int best_action_start = get_best_action(&ql, 0);
    TEST_ASSERT(best_action_start == ACTION_DOWN || best_action_start == ACTION_RIGHT, 
                "acao no inicio (0,0): BAIXO ou DIREITA");
    
    /* Estado 14 (adjacente ao objetivo): deve ir para direita */
    TEST_ASSERT(get_best_action(&ql, 14) == ACTION_RIGHT, 
                "acao em (3,2): DIREITA (para objetivo)");
    
    /* Estado 11 (adjacente ao objetivo): deve ir para baixo */
    TEST_ASSERT(get_best_action(&ql, 11) == ACTION_DOWN, 
                "acao em (2,3): BAIXO (para objetivo)");
    
    printf("  [INFO] Caminho encontrado em %d passos\n", path_length);
}

/**
 * Teste 11: Valores Q próximos ao objetivo
 * 
 * Verifica se os estados adjacentes ao objetivo têm valores Q altos
 * para as ações que levam ao objetivo.
 */
void test_q_values_near_goal() {
    QLearning ql;
    
    printf("\n[TESTE] Valores Q proximos ao objetivo:\n");
    
    srand(42);
    qlearning_init(&ql);
    train(&ql, NUM_EPISODES);
    
    /* Estado 14 (3,2) - esquerda do objetivo */
    /* Ação DIREITA deve ter o maior valor Q */
    double q_right = ql.q_table[14][ACTION_RIGHT];
    double q_up = ql.q_table[14][ACTION_UP];
    double q_left = ql.q_table[14][ACTION_LEFT];
    
    TEST_ASSERT(q_right > q_up && q_right > q_left, 
                "Q(14, DIREITA) e o maior valor para estado 14");
    
    /* Estado 11 (2,3) - acima do objetivo */
    /* Ação BAIXO deve ter o maior valor Q */
    double q_down = ql.q_table[11][ACTION_DOWN];
    q_up = ql.q_table[11][ACTION_UP];
    q_left = ql.q_table[11][ACTION_LEFT];
    
    TEST_ASSERT(q_down > q_up && q_down > q_left, 
                "Q(11, BAIXO) e o maior valor para estado 11");
    
    /* Valores Q adjacentes ao objetivo devem ser positivos */
    TEST_ASSERT(q_right > 0, 
                "Q(14, DIREITA) > 0 (proximo ao objetivo)");
    TEST_ASSERT(q_down > 0, 
                "Q(11, BAIXO) > 0 (proximo ao objetivo)");
}

/**
 * Teste 12: Política evita obstáculo
 * 
 * Verifica se a política aprendida evita o obstáculo em (1,1).
 */
void test_policy_avoids_obstacle() {
    QLearning ql;
    int state, action, next_state;
    int visits_obstacle = 0;
    
    printf("\n[TESTE] Politica evita obstaculo:\n");
    
    srand(42);
    qlearning_init(&ql);
    train(&ql, NUM_EPISODES);
    
    /* Simula o caminho e verifica se passa pelo obstáculo */
    state = START_STATE;
    while (state != GOAL_STATE && visits_obstacle == 0) {
        action = get_best_action(&ql, state);
        next_state = get_next_state(state, action);
        
        if (next_state == OBSTACLE_STATE) {
            visits_obstacle = 1;
        }
        
        if (state == next_state) break;  /* Evita loop infinito */
        state = next_state;
    }
    
    TEST_ASSERT(visits_obstacle == 0, 
                "caminho otimo NAO passa pelo obstaculo (1,1)");
}

/* =============================================================================
 * FUNÇÃO PRINCIPAL DOS TESTES
 * =============================================================================
 */

int main(void) {
    printf("============================================================\n");
    printf("     TESTES UNITARIOS - Q-LEARNING SEQUENCIAL\n");
    printf("============================================================\n");
    
    /* Executa todos os testes */
    test_coords_to_state();
    test_state_to_coords();
    test_get_next_state();
    test_rewards();
    test_terminal_state();
    test_qtable_initialization();
    test_q_update();
    test_get_max_q();
    test_get_best_action();
    test_training_convergence();
    test_q_values_near_goal();
    test_policy_avoids_obstacle();
    
    /* Resumo final */
    printf("\n============================================================\n");
    printf("                    RESUMO DOS TESTES\n");
    printf("============================================================\n");
    printf("  Testes passaram: %d\n", tests_passed);
    printf("  Testes falharam: %d\n", tests_failed);
    printf("  Total de testes: %d\n", tests_passed + tests_failed);
    printf("============================================================\n");
    
    if (tests_failed == 0) {
        printf("  *** TODOS OS TESTES PASSARAM! ***\n");
        printf("============================================================\n");
        return 0;
    } else {
        printf("  *** ALGUNS TESTES FALHARAM ***\n");
        printf("============================================================\n");
        return 1;
    }
}
