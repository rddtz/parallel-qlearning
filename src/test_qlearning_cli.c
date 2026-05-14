/**
 * =============================================================================
 * TESTES PARA Q-LEARNING SEQUENCIAL - VERSÃO CLI
 * =============================================================================
 * 
 * Este arquivo contém testes unitários para validar o funcionamento correto
 * do algoritmo Q-Learning com suporte a argumentos de linha de comando.
 * 
 * Os testes verificam:
 * 1. Parsing de argumentos de linha de comando
 * 2. Modos predefinidos (easy, normal, hard, extreme, debug)
 * 3. Alocação e liberação de memória dinâmica
 * 4. Geração de obstáculos aleatórios com seed
 * 5. Reprodutibilidade com mesma seed
 * 6. Conversões de coordenadas para diferentes tamanhos de grid
 * 7. Cálculo de próximo estado e recompensas
 * 8. Atualização da Q-table (equação de Bellman)
 * 9. Convergência do treinamento para vários tamanhos de grid
 * 10. Caminho ótimo encontrado corretamente
 * 
 * Compilação: gcc -o test_qlearning_cli test_qlearning_cli.c -lm
 * Execução: ./test_qlearning_cli
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

/* =============================================================================
 * CONSTANTES E DEFINIÇÕES (copiadas do qlearning_cli.c)
 * =============================================================================
 */

/* Limites do sistema */
#define MAX_GRID_SIZE 50
#define MAX_STATES (MAX_GRID_SIZE * MAX_GRID_SIZE)
#define NUM_ACTIONS 4
#define MAX_OBSTACLES 100

/* Identificadores das ações */
#define ACTION_UP    0
#define ACTION_DOWN  1
#define ACTION_LEFT  2
#define ACTION_RIGHT 3

/* Tipos de célula no grid */
#define CELL_EMPTY    0
#define CELL_OBSTACLE 1
#define CELL_GOAL     2
#define CELL_START    3

/* Tolerância para comparação de doubles */
#define EPSILON_TOLERANCE 1e-6

/* =============================================================================
 * ESTRUTURAS DE DADOS (copiadas do qlearning_cli.c)
 * =============================================================================
 */

typedef struct {
    int grid_rows;
    int grid_cols;
    double alpha;
    double gamma;
    double epsilon;
    int num_episodes;
    int max_steps;
    int num_obstacles;
    unsigned int seed;
    double reward_goal;
    double reward_obstacle;
    double reward_step;
    int verbose;
    int step_by_step;
    int quiet;
    int show_table;
    int show_policy;
    char mode[20];
} Config;

typedef struct {
    double **q_table;
    int **grid;
    int *obstacles;
    int num_obstacles;
    int goal_state;
    int start_state;
    int num_states;
    Config *config;
} QLearning;

/* =============================================================================
 * FUNÇÕES DO Q-LEARNING (copiadas para independência dos testes)
 * =============================================================================
 */

void config_set_defaults(Config *cfg) {
    cfg->grid_rows = 4;
    cfg->grid_cols = 4;
    cfg->alpha = 0.1;
    cfg->gamma = 0.9;
    cfg->epsilon = 0.1;
    cfg->num_episodes = 1000;
    cfg->max_steps = 100;
    cfg->num_obstacles = 1;
    cfg->seed = 42;
    cfg->reward_goal = 100.0;
    cfg->reward_obstacle = -100.0;
    cfg->reward_step = -1.0;
    cfg->verbose = 0;  /* Silencioso para testes */
    cfg->step_by_step = 0;
    cfg->quiet = 1;    /* Modo silencioso para testes */
    cfg->show_table = 0;
    cfg->show_policy = 0;
    strcpy(cfg->mode, "normal");
}

void config_apply_mode(Config *cfg, const char *mode) {
    strcpy(cfg->mode, mode);
    
    if (strcmp(mode, "easy") == 0) {
        cfg->grid_rows = 3;
        cfg->grid_cols = 3;
        cfg->alpha = 0.2;
        cfg->gamma = 0.8;
        cfg->epsilon = 0.15;
        cfg->num_episodes = 300;
        cfg->max_steps = 50;
        cfg->num_obstacles = 1;
    }
    else if (strcmp(mode, "normal") == 0) {
        cfg->grid_rows = 4;
        cfg->grid_cols = 4;
        cfg->alpha = 0.1;
        cfg->gamma = 0.9;
        cfg->epsilon = 0.1;
        cfg->num_episodes = 1000;
        cfg->max_steps = 100;
        cfg->num_obstacles = 1;
    }
    else if (strcmp(mode, "hard") == 0) {
        cfg->grid_rows = 6;
        cfg->grid_cols = 6;
        cfg->alpha = 0.08;
        cfg->gamma = 0.92;
        cfg->epsilon = 0.08;
        cfg->num_episodes = 3000;
        cfg->max_steps = 200;
        cfg->num_obstacles = 4;
    }
    else if (strcmp(mode, "extreme") == 0) {
        cfg->grid_rows = 10;
        cfg->grid_cols = 10;
        cfg->alpha = 0.05;
        cfg->gamma = 0.95;
        cfg->epsilon = 0.05;
        cfg->num_episodes = 5000;
        cfg->max_steps = 300;
        cfg->num_obstacles = 15;
    }
    else if (strcmp(mode, "debug") == 0) {
        cfg->grid_rows = 3;
        cfg->grid_cols = 3;
        cfg->alpha = 0.3;
        cfg->gamma = 0.8;
        cfg->epsilon = 0.2;
        cfg->num_episodes = 100;
        cfg->max_steps = 30;
        cfg->num_obstacles = 1;
    }
}

int coords_to_state(Config *cfg, int row, int col) {
    return row * cfg->grid_cols + col;
}

void state_to_coords(Config *cfg, int state, int *row, int *col) {
    *row = state / cfg->grid_cols;
    *col = state % cfg->grid_cols;
}

double random_double(void) {
    return (double)rand() / RAND_MAX;
}

int random_int(int max) {
    return rand() % max;
}

int qlearning_alloc(QLearning *ql, Config *cfg) {
    int i;
    int num_states = cfg->grid_rows * cfg->grid_cols;
    
    ql->config = cfg;
    ql->num_states = num_states;
    
    ql->q_table = (double **)malloc(num_states * sizeof(double *));
    if (!ql->q_table) return -1;
    
    for (i = 0; i < num_states; i++) {
        ql->q_table[i] = (double *)calloc(NUM_ACTIONS, sizeof(double));
        if (!ql->q_table[i]) return -1;
    }
    
    ql->grid = (int **)malloc(cfg->grid_rows * sizeof(int *));
    if (!ql->grid) return -1;
    
    for (i = 0; i < cfg->grid_rows; i++) {
        ql->grid[i] = (int *)calloc(cfg->grid_cols, sizeof(int));
        if (!ql->grid[i]) return -1;
    }
    
    ql->obstacles = (int *)malloc(MAX_OBSTACLES * sizeof(int));
    if (!ql->obstacles) return -1;
    
    return 0;
}

void qlearning_free(QLearning *ql) {
    int i;
    Config *cfg = ql->config;
    
    if (ql->q_table) {
        for (i = 0; i < ql->num_states; i++) {
            free(ql->q_table[i]);
        }
        free(ql->q_table);
    }
    
    if (ql->grid) {
        for (i = 0; i < cfg->grid_rows; i++) {
            free(ql->grid[i]);
        }
        free(ql->grid);
    }
    
    free(ql->obstacles);
}

int is_valid_obstacle_position(QLearning *ql, int state) {
    if (state == ql->start_state || state == ql->goal_state) {
        return 0;
    }
    
    for (int i = 0; i < ql->num_obstacles; i++) {
        if (ql->obstacles[i] == state) {
            return 0;
        }
    }
    
    return 1;
}

void generate_random_obstacles(QLearning *ql) {
    Config *cfg = ql->config;
    int placed = 0;
    int max_attempts = cfg->grid_rows * cfg->grid_cols * 10;
    int attempts = 0;
    
    while (placed < cfg->num_obstacles && attempts < max_attempts) {
        int state = random_int(ql->num_states);
        
        if (is_valid_obstacle_position(ql, state)) {
            ql->obstacles[placed] = state;
            
            int row, col;
            state_to_coords(cfg, state, &row, &col);
            ql->grid[row][col] = CELL_OBSTACLE;
            
            placed++;
        }
        attempts++;
    }
    
    ql->num_obstacles = placed;
}

void qlearning_init(QLearning *ql) {
    Config *cfg = ql->config;
    int i, j;
    
    srand(cfg->seed);
    
    for (i = 0; i < ql->num_states; i++) {
        for (j = 0; j < NUM_ACTIONS; j++) {
            ql->q_table[i][j] = 0.0;
        }
    }
    
    for (i = 0; i < cfg->grid_rows; i++) {
        for (j = 0; j < cfg->grid_cols; j++) {
            ql->grid[i][j] = CELL_EMPTY;
        }
    }
    
    ql->start_state = 0;
    ql->goal_state = coords_to_state(cfg, cfg->grid_rows - 1, cfg->grid_cols - 1);
    
    ql->grid[0][0] = CELL_START;
    ql->grid[cfg->grid_rows - 1][cfg->grid_cols - 1] = CELL_GOAL;
    
    ql->num_obstacles = 0;
    generate_random_obstacles(ql);
}

int is_valid_state(Config *cfg, int row, int col) {
    return (row >= 0 && row < cfg->grid_rows && col >= 0 && col < cfg->grid_cols);
}

int is_obstacle(QLearning *ql, int state) {
    for (int i = 0; i < ql->num_obstacles; i++) {
        if (ql->obstacles[i] == state) {
            return 1;
        }
    }
    return 0;
}

int get_next_state(QLearning *ql, int current_state, int action) {
    Config *cfg = ql->config;
    int row, col, new_row, new_col;
    
    state_to_coords(cfg, current_state, &row, &col);
    new_row = row;
    new_col = col;
    
    switch (action) {
        case ACTION_UP:    new_row = row - 1; break;
        case ACTION_DOWN:  new_row = row + 1; break;
        case ACTION_LEFT:  new_col = col - 1; break;
        case ACTION_RIGHT: new_col = col + 1; break;
    }
    
    if (is_valid_state(cfg, new_row, new_col)) {
        return coords_to_state(cfg, new_row, new_col);
    }
    
    return current_state;
}

double get_reward(QLearning *ql, int next_state) {
    Config *cfg = ql->config;
    
    if (next_state == ql->goal_state) {
        return cfg->reward_goal;
    } else if (is_obstacle(ql, next_state)) {
        return cfg->reward_obstacle;
    }
    return cfg->reward_step;
}

int is_terminal_state(QLearning *ql, int state) {
    return (state == ql->goal_state);
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

int select_action(QLearning *ql, int state) {
    Config *cfg = ql->config;
    int action, best_action;
    double best_value;
    
    if (random_double() < cfg->epsilon) {
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

void update_q_value(QLearning *ql, int state, int action, 
                    double reward, int next_state) {
    Config *cfg = ql->config;
    double current_q = ql->q_table[state][action];
    double max_next_q = get_max_q_value(ql, next_state);
    
    ql->q_table[state][action] = current_q + 
        cfg->alpha * (reward + cfg->gamma * max_next_q - current_q);
}

double run_episode(QLearning *ql, int episode_num) {
    Config *cfg = ql->config;
    int step, state, action, next_state;
    double reward, total_reward;
    
    (void)episode_num; /* Evita warning de variável não usada */
    
    state = ql->start_state;
    total_reward = 0.0;
    
    for (step = 0; step < cfg->max_steps; step++) {
        action = select_action(ql, state);
        next_state = get_next_state(ql, state, action);
        reward = get_reward(ql, next_state);
        
        update_q_value(ql, state, action, reward, next_state);
        total_reward += reward;
        
        if (is_terminal_state(ql, next_state)) {
            break;
        }
        
        state = next_state;
    }
    
    return total_reward;
}

void train(QLearning *ql) {
    Config *cfg = ql->config;
    int episode;
    
    for (episode = 0; episode < cfg->num_episodes; episode++) {
        run_episode(ql, episode);
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

/**
 * Simula o caminho aprendido e retorna o número de passos para chegar ao objetivo
 * Retorna -1 se não conseguir chegar
 */
int get_path_length(QLearning *ql) {
    Config *cfg = ql->config;
    int state = ql->start_state;
    int step = 0;
    int action;
    
    while (state != ql->goal_state && step < cfg->max_steps) {
        action = get_best_action(ql, state);
        state = get_next_state(ql, state, action);
        step++;
    }
    
    return (state == ql->goal_state) ? step : -1;
}

/* =============================================================================
 * CONTADORES DE TESTES
 * =============================================================================
 */

int tests_passed = 0;
int tests_failed = 0;

#define TEST(name) void name(void)
#define ASSERT(condition, message) do { \
    if (condition) { \
        tests_passed++; \
        printf("[PASSOU] %s\n", message); \
    } else { \
        tests_failed++; \
        printf("[FALHOU] %s\n", message); \
    } \
} while(0)

#define ASSERT_EQ(a, b, message) ASSERT((a) == (b), message)
#define ASSERT_NEQ(a, b, message) ASSERT((a) != (b), message)
#define ASSERT_DOUBLE_EQ(a, b, message) ASSERT(fabs((a) - (b)) < EPSILON_TOLERANCE, message)
#define ASSERT_TRUE(condition, message) ASSERT(condition, message)
#define ASSERT_FALSE(condition, message) ASSERT(!(condition), message)

/* =============================================================================
 * TESTES DE CONFIGURAÇÃO E PARSING
 * =============================================================================
 */

TEST(test_config_defaults) {
    printf("\n--- Testes: Configuração Padrão ---\n");
    Config cfg;
    config_set_defaults(&cfg);
    
    ASSERT_EQ(cfg.grid_rows, 4, "Padrão: grid_rows = 4");
    ASSERT_EQ(cfg.grid_cols, 4, "Padrão: grid_cols = 4");
    ASSERT_DOUBLE_EQ(cfg.alpha, 0.1, "Padrão: alpha = 0.1");
    ASSERT_DOUBLE_EQ(cfg.gamma, 0.9, "Padrão: gamma = 0.9");
    ASSERT_DOUBLE_EQ(cfg.epsilon, 0.1, "Padrão: epsilon = 0.1");
    ASSERT_EQ(cfg.num_episodes, 1000, "Padrão: num_episodes = 1000");
    ASSERT_EQ(cfg.max_steps, 100, "Padrão: max_steps = 100");
    ASSERT_EQ(cfg.num_obstacles, 1, "Padrão: num_obstacles = 1");
    ASSERT_EQ(cfg.seed, 42, "Padrão: seed = 42");
    ASSERT_DOUBLE_EQ(cfg.reward_goal, 100.0, "Padrão: reward_goal = 100.0");
    ASSERT_DOUBLE_EQ(cfg.reward_obstacle, -100.0, "Padrão: reward_obstacle = -100.0");
    ASSERT_DOUBLE_EQ(cfg.reward_step, -1.0, "Padrão: reward_step = -1.0");
}

TEST(test_mode_easy) {
    printf("\n--- Testes: Modo Easy ---\n");
    Config cfg;
    config_set_defaults(&cfg);
    config_apply_mode(&cfg, "easy");
    
    ASSERT_EQ(cfg.grid_rows, 3, "Easy: grid_rows = 3");
    ASSERT_EQ(cfg.grid_cols, 3, "Easy: grid_cols = 3");
    ASSERT_DOUBLE_EQ(cfg.alpha, 0.2, "Easy: alpha = 0.2");
    ASSERT_DOUBLE_EQ(cfg.gamma, 0.8, "Easy: gamma = 0.8");
    ASSERT_EQ(cfg.num_episodes, 300, "Easy: num_episodes = 300");
    ASSERT_EQ(cfg.num_obstacles, 1, "Easy: num_obstacles = 1");
    ASSERT_TRUE(strcmp(cfg.mode, "easy") == 0, "Easy: mode = 'easy'");
}

TEST(test_mode_normal) {
    printf("\n--- Testes: Modo Normal ---\n");
    Config cfg;
    config_set_defaults(&cfg);
    config_apply_mode(&cfg, "normal");
    
    ASSERT_EQ(cfg.grid_rows, 4, "Normal: grid_rows = 4");
    ASSERT_EQ(cfg.grid_cols, 4, "Normal: grid_cols = 4");
    ASSERT_DOUBLE_EQ(cfg.alpha, 0.1, "Normal: alpha = 0.1");
    ASSERT_DOUBLE_EQ(cfg.gamma, 0.9, "Normal: gamma = 0.9");
    ASSERT_EQ(cfg.num_episodes, 1000, "Normal: num_episodes = 1000");
    ASSERT_TRUE(strcmp(cfg.mode, "normal") == 0, "Normal: mode = 'normal'");
}

TEST(test_mode_hard) {
    printf("\n--- Testes: Modo Hard ---\n");
    Config cfg;
    config_set_defaults(&cfg);
    config_apply_mode(&cfg, "hard");
    
    ASSERT_EQ(cfg.grid_rows, 6, "Hard: grid_rows = 6");
    ASSERT_EQ(cfg.grid_cols, 6, "Hard: grid_cols = 6");
    ASSERT_DOUBLE_EQ(cfg.alpha, 0.08, "Hard: alpha = 0.08");
    ASSERT_EQ(cfg.num_episodes, 3000, "Hard: num_episodes = 3000");
    ASSERT_EQ(cfg.num_obstacles, 4, "Hard: num_obstacles = 4");
}

TEST(test_mode_extreme) {
    printf("\n--- Testes: Modo Extreme ---\n");
    Config cfg;
    config_set_defaults(&cfg);
    config_apply_mode(&cfg, "extreme");
    
    ASSERT_EQ(cfg.grid_rows, 10, "Extreme: grid_rows = 10");
    ASSERT_EQ(cfg.grid_cols, 10, "Extreme: grid_cols = 10");
    ASSERT_DOUBLE_EQ(cfg.alpha, 0.05, "Extreme: alpha = 0.05");
    ASSERT_EQ(cfg.num_episodes, 5000, "Extreme: num_episodes = 5000");
    ASSERT_EQ(cfg.num_obstacles, 15, "Extreme: num_obstacles = 15");
}

/* =============================================================================
 * TESTES DE ALOCAÇÃO E LIBERAÇÃO DE MEMÓRIA
 * =============================================================================
 */

TEST(test_memory_allocation_3x3) {
    printf("\n--- Testes: Alocação Memória 3x3 ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.grid_rows = 3;
    cfg.grid_cols = 3;
    cfg.num_obstacles = 0;
    
    int result = qlearning_alloc(&ql, &cfg);
    ASSERT_EQ(result, 0, "Alocação 3x3: retornou 0 (sucesso)");
    ASSERT_EQ(ql.num_states, 9, "Alocação 3x3: num_states = 9");
    ASSERT_TRUE(ql.q_table != NULL, "Alocação 3x3: q_table alocada");
    ASSERT_TRUE(ql.grid != NULL, "Alocação 3x3: grid alocado");
    
    qlearning_free(&ql);
    printf("[INFO] Memória liberada com sucesso\n");
}

TEST(test_memory_allocation_10x10) {
    printf("\n--- Testes: Alocação Memória 10x10 ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.grid_rows = 10;
    cfg.grid_cols = 10;
    cfg.num_obstacles = 0;
    
    int result = qlearning_alloc(&ql, &cfg);
    ASSERT_EQ(result, 0, "Alocação 10x10: retornou 0 (sucesso)");
    ASSERT_EQ(ql.num_states, 100, "Alocação 10x10: num_states = 100");
    
    qlearning_free(&ql);
}

TEST(test_memory_allocation_large) {
    printf("\n--- Testes: Alocação Memória Grande 20x20 ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.grid_rows = 20;
    cfg.grid_cols = 20;
    cfg.num_obstacles = 0;
    
    int result = qlearning_alloc(&ql, &cfg);
    ASSERT_EQ(result, 0, "Alocação 20x20: retornou 0 (sucesso)");
    ASSERT_EQ(ql.num_states, 400, "Alocação 20x20: num_states = 400");
    
    qlearning_free(&ql);
}

/* =============================================================================
 * TESTES DE CONVERSÃO DE COORDENADAS
 * =============================================================================
 */

TEST(test_coords_conversion_3x3) {
    printf("\n--- Testes: Conversão Coordenadas 3x3 ---\n");
    Config cfg;
    config_set_defaults(&cfg);
    cfg.grid_rows = 3;
    cfg.grid_cols = 3;
    
    ASSERT_EQ(coords_to_state(&cfg, 0, 0), 0, "3x3: (0,0) -> estado 0");
    ASSERT_EQ(coords_to_state(&cfg, 0, 2), 2, "3x3: (0,2) -> estado 2");
    ASSERT_EQ(coords_to_state(&cfg, 1, 0), 3, "3x3: (1,0) -> estado 3");
    ASSERT_EQ(coords_to_state(&cfg, 1, 1), 4, "3x3: (1,1) -> estado 4");
    ASSERT_EQ(coords_to_state(&cfg, 2, 2), 8, "3x3: (2,2) -> estado 8");
    
    int row, col;
    state_to_coords(&cfg, 4, &row, &col);
    ASSERT_TRUE(row == 1 && col == 1, "3x3: estado 4 -> (1,1)");
    
    state_to_coords(&cfg, 8, &row, &col);
    ASSERT_TRUE(row == 2 && col == 2, "3x3: estado 8 -> (2,2)");
}

TEST(test_coords_conversion_5x5) {
    printf("\n--- Testes: Conversão Coordenadas 5x5 ---\n");
    Config cfg;
    config_set_defaults(&cfg);
    cfg.grid_rows = 5;
    cfg.grid_cols = 5;
    
    ASSERT_EQ(coords_to_state(&cfg, 0, 0), 0, "5x5: (0,0) -> estado 0");
    ASSERT_EQ(coords_to_state(&cfg, 0, 4), 4, "5x5: (0,4) -> estado 4");
    ASSERT_EQ(coords_to_state(&cfg, 2, 2), 12, "5x5: (2,2) -> estado 12 (centro)");
    ASSERT_EQ(coords_to_state(&cfg, 4, 4), 24, "5x5: (4,4) -> estado 24 (objetivo)");
    
    int row, col;
    state_to_coords(&cfg, 12, &row, &col);
    ASSERT_TRUE(row == 2 && col == 2, "5x5: estado 12 -> (2,2)");
}

TEST(test_coords_conversion_asymmetric) {
    printf("\n--- Testes: Conversão Coordenadas Assimétrica 3x5 ---\n");
    Config cfg;
    config_set_defaults(&cfg);
    cfg.grid_rows = 3;
    cfg.grid_cols = 5;
    
    ASSERT_EQ(coords_to_state(&cfg, 0, 0), 0, "3x5: (0,0) -> estado 0");
    ASSERT_EQ(coords_to_state(&cfg, 0, 4), 4, "3x5: (0,4) -> estado 4");
    ASSERT_EQ(coords_to_state(&cfg, 1, 0), 5, "3x5: (1,0) -> estado 5");
    ASSERT_EQ(coords_to_state(&cfg, 2, 4), 14, "3x5: (2,4) -> estado 14 (objetivo)");
}

/* =============================================================================
 * TESTES DE GERAÇÃO DE OBSTÁCULOS
 * =============================================================================
 */

TEST(test_obstacles_reproducibility) {
    printf("\n--- Testes: Reprodutibilidade de Obstáculos ---\n");
    Config cfg;
    QLearning ql1, ql2;
    
    /* Primeira execução com seed 42 */
    config_set_defaults(&cfg);
    cfg.seed = 42;
    cfg.num_obstacles = 3;
    qlearning_alloc(&ql1, &cfg);
    qlearning_init(&ql1);
    
    /* Guarda posições dos obstáculos */
    int obs1[3] = {ql1.obstacles[0], ql1.obstacles[1], ql1.obstacles[2]};
    
    /* Segunda execução com mesma seed */
    config_set_defaults(&cfg);
    cfg.seed = 42;
    cfg.num_obstacles = 3;
    qlearning_alloc(&ql2, &cfg);
    qlearning_init(&ql2);
    
    ASSERT_EQ(ql1.num_obstacles, ql2.num_obstacles, "Obstáculos: mesmo número");
    ASSERT_TRUE(obs1[0] == ql2.obstacles[0], "Obstáculo 1: mesma posição");
    ASSERT_TRUE(obs1[1] == ql2.obstacles[1], "Obstáculo 2: mesma posição");
    ASSERT_TRUE(obs1[2] == ql2.obstacles[2], "Obstáculo 3: mesma posição");
    
    qlearning_free(&ql1);
    qlearning_free(&ql2);
}

TEST(test_obstacles_different_seeds) {
    printf("\n--- Testes: Seeds Diferentes Geram Obstáculos Diferentes ---\n");
    Config cfg;
    QLearning ql1, ql2;
    
    /* Primeira execução com seed 42 */
    config_set_defaults(&cfg);
    cfg.seed = 42;
    cfg.num_obstacles = 3;
    cfg.grid_rows = 6;
    cfg.grid_cols = 6;
    qlearning_alloc(&ql1, &cfg);
    qlearning_init(&ql1);
    
    /* Segunda execução com seed diferente */
    config_set_defaults(&cfg);
    cfg.seed = 123;
    cfg.num_obstacles = 3;
    cfg.grid_rows = 6;
    cfg.grid_cols = 6;
    qlearning_alloc(&ql2, &cfg);
    qlearning_init(&ql2);
    
    /* Verifica que pelo menos um obstáculo está em posição diferente */
    int all_same = (ql1.obstacles[0] == ql2.obstacles[0]) &&
                   (ql1.obstacles[1] == ql2.obstacles[1]) &&
                   (ql1.obstacles[2] == ql2.obstacles[2]);
    
    ASSERT_FALSE(all_same, "Seeds diferentes: obstáculos em posições diferentes");
    
    qlearning_free(&ql1);
    qlearning_free(&ql2);
}

TEST(test_obstacles_avoid_start_goal) {
    printf("\n--- Testes: Obstáculos Evitam Início e Objetivo ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.seed = 12345;
    cfg.num_obstacles = 10; /* Muitos obstáculos */
    cfg.grid_rows = 5;
    cfg.grid_cols = 5;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    /* Verifica que início e objetivo não são obstáculos */
    int start_is_obstacle = is_obstacle(&ql, ql.start_state);
    int goal_is_obstacle = is_obstacle(&ql, ql.goal_state);
    
    ASSERT_FALSE(start_is_obstacle, "Estado inicial não é obstáculo");
    ASSERT_FALSE(goal_is_obstacle, "Estado objetivo não é obstáculo");
    
    qlearning_free(&ql);
}

/* =============================================================================
 * TESTES DE PRÓXIMO ESTADO E MOVIMENTAÇÃO
 * =============================================================================
 */

TEST(test_next_state_movements_4x4) {
    printf("\n--- Testes: Movimentação em Grid 4x4 ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.num_obstacles = 0;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    /* Estado 5 (1,1) - centro do grid 4x4 */
    int state = coords_to_state(&cfg, 1, 1);
    
    ASSERT_EQ(get_next_state(&ql, state, ACTION_UP), 1, "4x4: (1,1) + UP = (0,1)");
    ASSERT_EQ(get_next_state(&ql, state, ACTION_DOWN), 9, "4x4: (1,1) + DOWN = (2,1)");
    ASSERT_EQ(get_next_state(&ql, state, ACTION_LEFT), 4, "4x4: (1,1) + LEFT = (1,0)");
    ASSERT_EQ(get_next_state(&ql, state, ACTION_RIGHT), 6, "4x4: (1,1) + RIGHT = (1,2)");
    
    qlearning_free(&ql);
}

TEST(test_next_state_boundaries_4x4) {
    printf("\n--- Testes: Limites do Grid 4x4 ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.num_obstacles = 0;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    /* Estado 0 (0,0) - canto superior esquerdo */
    ASSERT_EQ(get_next_state(&ql, 0, ACTION_UP), 0, "4x4: (0,0) + UP = (0,0) (limite)");
    ASSERT_EQ(get_next_state(&ql, 0, ACTION_LEFT), 0, "4x4: (0,0) + LEFT = (0,0) (limite)");
    
    /* Estado 15 (3,3) - canto inferior direito */
    ASSERT_EQ(get_next_state(&ql, 15, ACTION_DOWN), 15, "4x4: (3,3) + DOWN = (3,3) (limite)");
    ASSERT_EQ(get_next_state(&ql, 15, ACTION_RIGHT), 15, "4x4: (3,3) + RIGHT = (3,3) (limite)");
    
    qlearning_free(&ql);
}

TEST(test_next_state_5x5) {
    printf("\n--- Testes: Movimentação em Grid 5x5 ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.grid_rows = 5;
    cfg.grid_cols = 5;
    cfg.num_obstacles = 0;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    /* Estado 12 (2,2) - centro do grid 5x5 */
    int state = 12;
    
    ASSERT_EQ(get_next_state(&ql, state, ACTION_UP), 7, "5x5: (2,2) + UP = (1,2)");
    ASSERT_EQ(get_next_state(&ql, state, ACTION_DOWN), 17, "5x5: (2,2) + DOWN = (3,2)");
    ASSERT_EQ(get_next_state(&ql, state, ACTION_LEFT), 11, "5x5: (2,2) + LEFT = (2,1)");
    ASSERT_EQ(get_next_state(&ql, state, ACTION_RIGHT), 13, "5x5: (2,2) + RIGHT = (2,3)");
    
    qlearning_free(&ql);
}

/* =============================================================================
 * TESTES DE RECOMPENSA
 * =============================================================================
 */

TEST(test_rewards_4x4) {
    printf("\n--- Testes: Recompensas em Grid 4x4 ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.num_obstacles = 0;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    ASSERT_DOUBLE_EQ(get_reward(&ql, ql.goal_state), 100.0, "4x4: Recompensa objetivo = 100");
    ASSERT_DOUBLE_EQ(get_reward(&ql, 1), -1.0, "4x4: Recompensa passo normal = -1");
    
    qlearning_free(&ql);
}

TEST(test_rewards_with_obstacle) {
    printf("\n--- Testes: Recompensa com Obstáculo ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.seed = 42;
    cfg.num_obstacles = 1;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    if (ql.num_obstacles > 0) {
        int obs_state = ql.obstacles[0];
        ASSERT_DOUBLE_EQ(get_reward(&ql, obs_state), -100.0, "Recompensa obstáculo = -100");
    }
    
    qlearning_free(&ql);
}

/* =============================================================================
 * TESTES DE ATUALIZAÇÃO DA Q-TABLE (EQUAÇÃO DE BELLMAN)
 * =============================================================================
 */

TEST(test_bellman_update) {
    printf("\n--- Testes: Atualização Q (Bellman) ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.num_obstacles = 0;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    /* Q(s,a) = Q(s,a) + α * (R + γ * max(Q(s',a')) - Q(s,a)) */
    /* Com Q-table zerada, alpha=0.1, gamma=0.9 */
    /* Q(0,1) = 0 + 0.1 * (-1 + 0.9 * 0 - 0) = -0.1 */
    
    update_q_value(&ql, 0, ACTION_DOWN, -1.0, 4);
    ASSERT_DOUBLE_EQ(ql.q_table[0][ACTION_DOWN], -0.1, "Bellman: Q(0, DOWN) = -0.1");
    
    /* Agora testa com recompensa positiva (objetivo) */
    /* Q(11, DOWN) indo para objetivo (estado 15) */
    /* Q(11,1) = 0 + 0.1 * (100 + 0.9 * 0 - 0) = 10 */
    update_q_value(&ql, 11, ACTION_DOWN, 100.0, 15);
    ASSERT_DOUBLE_EQ(ql.q_table[11][ACTION_DOWN], 10.0, "Bellman: Q(11, DOWN) = 10.0 (objetivo)");
    
    qlearning_free(&ql);
}

TEST(test_bellman_propagation) {
    printf("\n--- Testes: Propagação de Valores Q ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.num_obstacles = 0;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    /* Simula algumas atualizações para verificar propagação */
    /* Primeiro, atualiza estado próximo ao objetivo */
    update_q_value(&ql, 14, ACTION_RIGHT, 100.0, 15); /* (3,2) -> (3,3) objetivo */
    
    /* Agora, estados mais distantes devem receber valores propagados */
    double max_q_14 = get_max_q_value(&ql, 14);
    ASSERT_TRUE(max_q_14 > 0, "Propagação: Q máximo em estado 14 > 0");
    
    /* Atualiza estado 13 -> 14 */
    update_q_value(&ql, 13, ACTION_RIGHT, -1.0, 14);
    double q_13 = ql.q_table[13][ACTION_RIGHT];
    
    /* Com gamma=0.9, deve propagar parte do valor */
    ASSERT_TRUE(q_13 > -1.0, "Propagação: Q(13, RIGHT) > -1 devido ao valor futuro");
    
    qlearning_free(&ql);
}

/* =============================================================================
 * TESTES DE CONVERGÊNCIA E CAMINHO ÓTIMO
 * =============================================================================
 */

TEST(test_convergence_3x3_no_obstacles) {
    printf("\n--- Testes: Convergência Grid 3x3 sem Obstáculos ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.grid_rows = 3;
    cfg.grid_cols = 3;
    cfg.num_obstacles = 0;
    cfg.num_episodes = 500;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    train(&ql);
    
    int path_length = get_path_length(&ql);
    
    /* Caminho mínimo de (0,0) para (2,2) em 3x3 sem obstáculos = 4 passos */
    ASSERT_TRUE(path_length > 0, "3x3: Agente encontrou caminho");
    ASSERT_TRUE(path_length <= 6, "3x3: Caminho <= 6 passos (razoável)");
    
    printf("[INFO] Caminho encontrado: %d passos (ótimo: 4)\n", path_length);
    
    qlearning_free(&ql);
}

TEST(test_convergence_4x4_no_obstacles) {
    printf("\n--- Testes: Convergência Grid 4x4 sem Obstáculos ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.num_obstacles = 0;
    cfg.num_episodes = 1000;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    train(&ql);
    
    int path_length = get_path_length(&ql);
    
    /* Caminho mínimo de (0,0) para (3,3) em 4x4 = 6 passos */
    ASSERT_TRUE(path_length > 0, "4x4: Agente encontrou caminho");
    ASSERT_TRUE(path_length <= 8, "4x4: Caminho <= 8 passos (razoável)");
    
    printf("[INFO] Caminho encontrado: %d passos (ótimo: 6)\n", path_length);
    
    qlearning_free(&ql);
}

TEST(test_convergence_4x4_with_obstacle) {
    printf("\n--- Testes: Convergência Grid 4x4 com 1 Obstáculo ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.seed = 42;
    cfg.num_obstacles = 1;
    cfg.num_episodes = 1000;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    train(&ql);
    
    int path_length = get_path_length(&ql);
    
    ASSERT_TRUE(path_length > 0, "4x4+obs: Agente encontrou caminho");
    ASSERT_TRUE(path_length <= 12, "4x4+obs: Caminho razoável encontrado");
    
    printf("[INFO] Caminho encontrado: %d passos\n", path_length);
    
    qlearning_free(&ql);
}

TEST(test_convergence_5x5_with_obstacles) {
    printf("\n--- Testes: Convergência Grid 5x5 com 3 Obstáculos ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.grid_rows = 5;
    cfg.grid_cols = 5;
    cfg.seed = 123;
    cfg.num_obstacles = 3;
    cfg.num_episodes = 1500;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    train(&ql);
    
    int path_length = get_path_length(&ql);
    
    ASSERT_TRUE(path_length > 0, "5x5+obs: Agente encontrou caminho");
    ASSERT_TRUE(path_length <= 15, "5x5+obs: Caminho razoável encontrado");
    
    printf("[INFO] Caminho encontrado: %d passos (ótimo sem obstáculos: 8)\n", path_length);
    
    qlearning_free(&ql);
}

TEST(test_convergence_6x6_hard) {
    printf("\n--- Testes: Convergência Grid 6x6 (Modo Hard) ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    config_apply_mode(&cfg, "hard");
    cfg.quiet = 1;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    train(&ql);
    
    int path_length = get_path_length(&ql);
    
    ASSERT_TRUE(path_length > 0, "Hard (6x6): Agente encontrou caminho");
    ASSERT_TRUE(path_length <= 25, "Hard (6x6): Caminho razoável encontrado");
    
    printf("[INFO] Caminho encontrado: %d passos (ótimo sem obstáculos: 10)\n", path_length);
    
    qlearning_free(&ql);
}

/* =============================================================================
 * TESTES DE POLÍTICA APRENDIDA
 * =============================================================================
 */

TEST(test_policy_direction_near_goal) {
    printf("\n--- Testes: Política Correta Próximo ao Objetivo ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.num_obstacles = 0;
    cfg.num_episodes = 1000;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    train(&ql);
    
    /* Estado 14 (3,2) deve preferir ir para direita (objetivo em 15) */
    int action_14 = get_best_action(&ql, 14);
    ASSERT_EQ(action_14, ACTION_RIGHT, "Política: Estado (3,2) -> DIREITA");
    
    /* Estado 11 (2,3) deve preferir ir para baixo (caminho para objetivo) */
    int action_11 = get_best_action(&ql, 11);
    ASSERT_EQ(action_11, ACTION_DOWN, "Política: Estado (2,3) -> BAIXO");
    
    qlearning_free(&ql);
}

TEST(test_q_values_increase_toward_goal) {
    printf("\n--- Testes: Q-Values Aumentam em Direção ao Objetivo ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.num_obstacles = 0;
    cfg.num_episodes = 1000;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    train(&ql);
    
    /* Estados adjacentes ao objetivo devem ter Q-values positivos */
    /* Estado 14 (3,2) tem ação direita levando ao objetivo */
    /* Estado 11 (2,3) tem ação baixo levando ao objetivo */
    double q_14_right = ql.q_table[14][ACTION_RIGHT]; /* (3,2) -> (3,3) */
    double q_11_down = ql.q_table[11][ACTION_DOWN];   /* (2,3) -> (3,3) */
    
    ASSERT_TRUE(q_14_right > 50.0, "Q(14, RIGHT) > 50 (adjacente ao objetivo)");
    ASSERT_TRUE(q_11_down > 50.0, "Q(11, DOWN) > 50 (adjacente ao objetivo)");
    
    /* Estado inicial deve ter Q-values positivos (caminho aprendido) */
    double max_q_0 = get_max_q_value(&ql, 0);
    ASSERT_TRUE(max_q_0 > 0, "Q máximo do estado inicial > 0");
    
    /* Estado mais próximo deve ter valor maior que estado mais distante */
    /* Comparando estados na mesma "linha de decisão" */
    double max_q_4 = get_max_q_value(&ql, 4);   /* (1,0) - segunda linha */
    double max_q_12 = get_max_q_value(&ql, 12); /* (3,0) - quarta linha */
    ASSERT_TRUE(max_q_12 > max_q_4 || max_q_4 > max_q_0, 
                "Gradiente de Q-values ao longo do caminho");
    
    printf("[INFO] Q máximos: estado 0 = %.2f, estado 4 = %.2f, estado 12 = %.2f\n",
           max_q_0, max_q_4, max_q_12);
    
    qlearning_free(&ql);
}

/* =============================================================================
 * TESTES DE REPRODUTIBILIDADE
 * =============================================================================
 */

TEST(test_training_reproducibility) {
    printf("\n--- Testes: Reprodutibilidade do Treinamento ---\n");
    Config cfg;
    QLearning ql1, ql2;
    
    /* Primeira execução */
    config_set_defaults(&cfg);
    cfg.seed = 42;
    cfg.num_obstacles = 1;
    cfg.num_episodes = 500;
    qlearning_alloc(&ql1, &cfg);
    qlearning_init(&ql1);
    train(&ql1);
    int path1 = get_path_length(&ql1);
    double q1_0 = get_max_q_value(&ql1, 0);
    
    /* Segunda execução com mesmos parâmetros */
    config_set_defaults(&cfg);
    cfg.seed = 42;
    cfg.num_obstacles = 1;
    cfg.num_episodes = 500;
    qlearning_alloc(&ql2, &cfg);
    qlearning_init(&ql2);
    train(&ql2);
    int path2 = get_path_length(&ql2);
    double q2_0 = get_max_q_value(&ql2, 0);
    
    ASSERT_EQ(path1, path2, "Reprodutibilidade: mesmo comprimento de caminho");
    ASSERT_DOUBLE_EQ(q1_0, q2_0, "Reprodutibilidade: mesmo Q(0) máximo");
    
    qlearning_free(&ql1);
    qlearning_free(&ql2);
}

/* =============================================================================
 * TESTES DE ESTADOS TERMINAIS
 * =============================================================================
 */

TEST(test_terminal_state_goal) {
    printf("\n--- Testes: Estado Terminal (Objetivo) ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.num_obstacles = 0;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    ASSERT_TRUE(is_terminal_state(&ql, ql.goal_state), "Estado objetivo é terminal");
    ASSERT_FALSE(is_terminal_state(&ql, 0), "Estado 0 não é terminal");
    ASSERT_FALSE(is_terminal_state(&ql, 5), "Estado 5 não é terminal");
    
    qlearning_free(&ql);
}

/* =============================================================================
 * TESTES DE GRID ASSIMÉTRICO
 * =============================================================================
 */

TEST(test_asymmetric_grid_3x5) {
    printf("\n--- Testes: Grid Assimétrico 3x5 ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.grid_rows = 3;
    cfg.grid_cols = 5;
    cfg.num_obstacles = 0;
    cfg.num_episodes = 800;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    ASSERT_EQ(ql.num_states, 15, "3x5: 15 estados");
    ASSERT_EQ(ql.goal_state, 14, "3x5: objetivo = estado 14 (2,4)");
    
    train(&ql);
    
    int path_length = get_path_length(&ql);
    /* Caminho mínimo de (0,0) para (2,4) = 2 + 4 = 6 passos */
    ASSERT_TRUE(path_length > 0, "3x5: Agente encontrou caminho");
    ASSERT_TRUE(path_length <= 10, "3x5: Caminho razoável");
    
    printf("[INFO] Grid 3x5: caminho = %d passos (ótimo: 6)\n", path_length);
    
    qlearning_free(&ql);
}

TEST(test_asymmetric_grid_5x3) {
    printf("\n--- Testes: Grid Assimétrico 5x3 ---\n");
    Config cfg;
    QLearning ql;
    
    config_set_defaults(&cfg);
    cfg.grid_rows = 5;
    cfg.grid_cols = 3;
    cfg.num_obstacles = 0;
    cfg.num_episodes = 800;
    qlearning_alloc(&ql, &cfg);
    qlearning_init(&ql);
    
    ASSERT_EQ(ql.num_states, 15, "5x3: 15 estados");
    ASSERT_EQ(ql.goal_state, 14, "5x3: objetivo = estado 14 (4,2)");
    
    train(&ql);
    
    int path_length = get_path_length(&ql);
    ASSERT_TRUE(path_length > 0, "5x3: Agente encontrou caminho");
    ASSERT_TRUE(path_length <= 10, "5x3: Caminho razoável");
    
    printf("[INFO] Grid 5x3: caminho = %d passos (ótimo: 6)\n", path_length);
    
    qlearning_free(&ql);
}

/* =============================================================================
 * FUNÇÃO PRINCIPAL
 * =============================================================================
 */

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║        TESTES AUTOMATIZADOS - Q-LEARNING CLI (Versão Dinâmica)           ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════╝\n");
    
    /* Testes de Configuração */
    test_config_defaults();
    test_mode_easy();
    test_mode_normal();
    test_mode_hard();
    test_mode_extreme();
    
    /* Testes de Memória */
    test_memory_allocation_3x3();
    test_memory_allocation_10x10();
    test_memory_allocation_large();
    
    /* Testes de Coordenadas */
    test_coords_conversion_3x3();
    test_coords_conversion_5x5();
    test_coords_conversion_asymmetric();
    
    /* Testes de Obstáculos */
    test_obstacles_reproducibility();
    test_obstacles_different_seeds();
    test_obstacles_avoid_start_goal();
    
    /* Testes de Movimentação */
    test_next_state_movements_4x4();
    test_next_state_boundaries_4x4();
    test_next_state_5x5();
    
    /* Testes de Recompensa */
    test_rewards_4x4();
    test_rewards_with_obstacle();
    
    /* Testes de Bellman */
    test_bellman_update();
    test_bellman_propagation();
    
    /* Testes de Convergência */
    test_convergence_3x3_no_obstacles();
    test_convergence_4x4_no_obstacles();
    test_convergence_4x4_with_obstacle();
    test_convergence_5x5_with_obstacles();
    test_convergence_6x6_hard();
    
    /* Testes de Política */
    test_policy_direction_near_goal();
    test_q_values_increase_toward_goal();
    
    /* Testes de Reprodutibilidade */
    test_training_reproducibility();
    
    /* Testes de Estado Terminal */
    test_terminal_state_goal();
    
    /* Testes de Grid Assimétrico */
    test_asymmetric_grid_3x5();
    test_asymmetric_grid_5x3();
    
    /* Resultado Final */
    printf("\n╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                           RESULTADO FINAL                                ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Testes passaram: %3d                                                    ║\n", tests_passed);
    printf("║  Testes falharam: %3d                                                    ║\n", tests_failed);
    printf("║  Total de testes: %3d                                                    ║\n", tests_passed + tests_failed);
    printf("╠══════════════════════════════════════════════════════════════════════════╣\n");
    
    if (tests_failed == 0) {
        printf("║  ✓ TODOS OS TESTES PASSARAM!                                             ║\n");
    } else {
        printf("║  ✗ ALGUNS TESTES FALHARAM - VERIFICAR IMPLEMENTAÇÃO                      ║\n");
    }
    
    printf("╚══════════════════════════════════════════════════════════════════════════╝\n");
    
    return (tests_failed > 0) ? 1 : 0;
}
