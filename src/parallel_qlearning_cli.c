/**
 * =============================================================================
 * Q-LEARNING PARALELO EM C - VERSÃO COM ARGUMENTOS DE LINHA DE COMANDO
 * =============================================================================
 *
 * Este código implementa o algoritmo Q-Learning paralelo com OpenMP.
 * Threads treinam de forma independente por intervalos e sincronizam
 * periodicamente pela média das Q-tables (federated averaging).
 *
 * USO:
 * ----
 *   ./qlearning_parallel [opções]
 *
 * MODOS PREDEFINIDOS:
 *   --mode easy      Grid 3x3, poucos obstáculos, aprendizado rápido
 *   --mode normal    Grid 4x4, configuração balanceada (padrão)
 *   --mode hard      Grid 6x6, mais obstáculos, mais episódios
 *   --mode extreme   Grid 10x10, muitos obstáculos, desafiador
 *
 * OPÇÕES INDIVIDUAIS (sobrescrevem o modo):
 *   --gridx N           Largura do grid (colunas)
 *   --gridy N           Altura do grid (linhas)
 *   --obstacles N       Número de obstáculos aleatórios
 *   --seed N            Seed para geração aleatória (reprodutibilidade)
 *   --alpha F           Taxa de aprendizado (0.0 a 1.0)
 *   --gamma F           Fator de desconto (0.0 a 1.0)
 *   --epsilon F         Taxa de exploração (0.0 a 1.0)
 *   --episodes N        Número de episódios de treinamento
 *   --maxsteps N        Máximo de passos por episódio
 *   --sync-interval N   Episódios entre cada sincronização (padrão: automático)
 *   --verbose           Mostra progresso detalhado a cada episódio
 *   --step              Mostra cada passo do treinamento (muito detalhado!)
 *   --quiet             Modo silencioso (apenas resultado final)
 *   --no-table          Não mostra a Q-table
 *   --no-policy         Não mostra a política visual
 *   --help              Mostra esta ajuda
 *
 * EXEMPLOS:
 *   ./qlearning_parallel --mode normal
 *   ./qlearning_parallel --gridx 5 --gridy 5 --obstacles 3 --seed 42
 *   ./qlearning_parallel --mode hard --alpha 0.2 --sync-interval 50
 *   ./qlearning_parallel --episodes 2000 --sync-interval 100
 *
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <getopt.h>
#include <omp.h>

/* =============================================================================
 * CONSTANTES FIXAS
 * =============================================================================
 */

#define NUM_ACTIONS 4            /* Ações: cima, baixo, esquerda, direita */

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

/* =============================================================================
 * ESTRUTURA DE CONFIGURAÇÃO
 * =============================================================================
 */

/**
 * Estrutura que armazena todos os parâmetros configuráveis
 */
typedef struct {
    /* Dimensões do grid */
    int grid_rows;
    int grid_cols;

    /* Hiperparâmetros do Q-Learning */
    double alpha;           /* Taxa de aprendizado */
    double gamma;           /* Fator de desconto */
    double epsilon;         /* Taxa de exploração */

    /* Parâmetros de treinamento */
    int num_episodes;       /* Número de episódios */
    int max_steps;          /* Máximo de passos por episódio */

    /* Obstáculos */
    int num_obstacles;      /* Quantidade de obstáculos */
    unsigned int seed;      /* Seed para aleatoriedade */

    /* Sincronização entre threads */
    char sync_mode[20];     /* "sqrt", "statespace", "hogwild" */
    int sync_interval;      /* Sobrescreve o intervalo calculado (0 = automático) */

    /* Recompensas */
    double reward_goal;
    double reward_obstacle;
    double reward_step;

    /* Opções de saída */
    int verbose;            /* Mostra progresso a cada 100 episódios */
    int step_by_step;       /* Mostra cada passo do treinamento */
    int quiet;              /* Modo silencioso */
    int show_table;         /* Mostra Q-table */
    int show_policy;        /* Mostra política visual */

    /* Modo predefinido */
    char mode[20];
} Config;

/**
 * Estrutura principal do Q-Learning
 */
typedef struct {
    double **q_table;       /* Tabela Q: valores Q(s,a) - alocada dinamicamente */
    int **grid;             /* Representação do grid - alocada dinamicamente */
    int *obstacles;         /* Lista de estados com obstáculos */
    int num_obstacles;      /* Quantidade real de obstáculos */
    int goal_state;         /* Estado objetivo */
    int start_state;        /* Estado inicial */
    int num_states;         /* Total de estados */
    Config *config;         /* Ponteiro para configuração */
    unsigned int rand_seed; /* Semente per-thread para rand_r (sem lock global) */
} QLearning;

/* =============================================================================
 * VALORES PADRÃO E MODOS PREDEFINIDOS
 * =============================================================================
 */

/**
 * Configura valores padrão
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
    strcpy(cfg->sync_mode, "sqrt"); /* padrão: heurística sqrt(N/M) */
    cfg->sync_interval = 0;         /* 0 = calcular automaticamente */
    cfg->reward_goal = 100.0;
    cfg->reward_obstacle = -100.0;
    cfg->reward_step = -1.0;
    cfg->verbose = 1;
    cfg->step_by_step = 0;
    cfg->quiet = 0;
    cfg->show_table = 1;
    cfg->show_policy = 1;
    strcpy(cfg->mode, "normal");
}

/**
 * Aplica modo predefinido
 */
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
        cfg->verbose = 1;
    }
    else {
        fprintf(stderr, "Aviso: Modo '%s' desconhecido. Usando 'normal'.\n", mode);
        config_apply_mode(cfg, "normal");
    }
}

/* =============================================================================
 * HELP E PARSING DE ARGUMENTOS
 * =============================================================================
 */

/**
 * Mostra mensagem de ajuda
 */
void print_help(const char *program_name) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║               Q-LEARNING PARALELO (OpenMP) - GRID WORLD                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("USO: %s [opções]\n", program_name);
    printf("\n");
    printf("MODOS PREDEFINIDOS:\n");
    printf("  --mode easy       Grid 3x3, aprendizado rápido\n");
    printf("  --mode normal     Grid 4x4, configuração balanceada\n");
    printf("  --mode hard       Grid 6x6, mais desafiador\n");
    printf("  --mode extreme    Grid 10x10, muito difícil\n");
    printf("  --mode debug      Grid 3x3, para testes rápidos\n");
    printf("\n");
    printf("CONFIGURAÇÃO DO GRID:\n");
    printf("  --gridx N         Largura do grid em colunas (padrão: 4)\n");
    printf("  --gridy N         Altura do grid em linhas (padrão: 4)\n");
    printf("  --obstacles N     Número de obstáculos aleatórios (padrão: 1)\n");
    printf("  --seed N          Seed para geração aleatória (padrão: 42)\n");
    printf("\n");
    printf("HIPERPARÂMETROS DO Q-LEARNING:\n");
    printf("  --alpha F         Taxa de aprendizado 0.0-1.0 (padrão: 0.1)\n");
    printf("  --gamma F         Fator de desconto 0.0-1.0 (padrão: 0.9)\n");
    printf("  --epsilon F       Taxa de exploração 0.0-1.0 (padrão: 0.1)\n");
    printf("  --episodes N      Número de episódios (padrão: 1000)\n");
    printf("  --maxsteps N      Máximo passos por episódio (padrão: 100)\n");
    printf("\n");
    printf("SINCRONIZAÇÃO PARALELA:\n");
    printf("  --sync-mode M     Método de sincronização entre threads (padrão: sqrt)\n");
    printf("                    sqrt        Intervalo = sqrt(N/T) — heurística paralela\n");
    printf("                    statespace  Intervalo = num_estados (exploração completa)\n");
    printf("                    hogwild     Sem sync — Q-table compartilhada lock-free\n");
    printf("                                (Niu et al., NIPS 2011)\n");
    printf("  --sync-interval N Sobrescreve o intervalo calculado (só sqrt/statespace)\n");
    printf("\n");
    printf("OPÇÕES DE SAÍDA:\n");
    printf("  --verbose         Mostra progresso a cada 100 episódios\n");
    printf("  --step            Mostra cada passo do treinamento (muito detalhado!)\n");
    printf("  --quiet           Modo silencioso (apenas resultado final)\n");
    printf("  --no-table        Não mostra a Q-table\n");
    printf("  --no-policy       Não mostra a política visual\n");
    printf("  --help, -h        Mostra esta ajuda\n");
    printf("\n");
    printf("EXEMPLOS:\n");
    printf("  %s --mode normal\n", program_name);
    printf("  %s --gridx 5 --gridy 5 --obstacles 3 --seed 42\n", program_name);
    printf("  %s --mode hard --alpha 0.2 --sync-interval 100\n", program_name);
    printf("  %s --episodes 2000 --sync-interval 200 --quiet\n", program_name);
    printf("\n");
}

/**
 * Parseia argumentos de linha de comando
 */
int parse_arguments(int argc, char *argv[], Config *cfg) {
    static struct option long_options[] = {
        {"mode",          required_argument, 0, 'm'},
        {"gridx",         required_argument, 0, 'x'},
        {"gridy",         required_argument, 0, 'y'},
        {"obstacles",     required_argument, 0, 'o'},
        {"seed",          required_argument, 0, 's'},
        {"alpha",         required_argument, 0, 'a'},
        {"gamma",         required_argument, 0, 'g'},
        {"epsilon",       required_argument, 0, 'e'},
        {"episodes",      required_argument, 0, 'n'},
        {"maxsteps",      required_argument, 0, 't'},
        {"sync-mode",     required_argument, 0, 'M'},
        {"sync-interval", required_argument, 0, 'i'},
        {"verbose",       no_argument,       0, 'v'},
        {"step",          no_argument,       0, 'S'},
        {"quiet",         no_argument,       0, 'q'},
        {"no-table",      no_argument,       0, 'T'},
        {"no-policy",     no_argument,       0, 'P'},
        {"help",          no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    int mode_set = 0;

    /* Primeira passagem: procura por --mode para aplicar primeiro */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            config_apply_mode(cfg, argv[i + 1]);
            mode_set = 1;
            break;
        }
        if (strncmp(argv[i], "--mode=", 7) == 0) {
            config_apply_mode(cfg, argv[i] + 7);
            mode_set = 1;
            break;
        }
    }

    if (!mode_set) {
        config_apply_mode(cfg, "normal");
    }

    /* Reset getopt */
    optind = 1;

    /* Segunda passagem: processa todos os argumentos */
    while ((opt = getopt_long(argc, argv, "m:x:y:o:s:a:g:e:n:t:M:i:vSqTPh",
                              long_options, &option_index)) != -1) {
        switch (opt) {
            case 'm':
                /* Modo já processado na primeira passagem */
                break;
            case 'x':
                cfg->grid_cols = atoi(optarg);
                if (cfg->grid_cols < 2) {
                    fprintf(stderr, "Erro: gridx deve ser pelo menos 2\n");
                    return -1;
                }
                break;
            case 'y':
                cfg->grid_rows = atoi(optarg);
                if (cfg->grid_rows < 2) {
                    fprintf(stderr, "Erro: gridy deve ser pelo menos 2\n");
                    return -1;
                }
                break;
            case 'o':
                cfg->num_obstacles = atoi(optarg);
                if (cfg->num_obstacles < 0) {
                    fprintf(stderr, "Erro: obstacles deve ser pelo menos 0\n");
                    return -1;
                }
                break;
            case 's':
                cfg->seed = (unsigned int)atoi(optarg);
                break;
            case 'a':
                cfg->alpha = atof(optarg);
                if (cfg->alpha <= 0.0 || cfg->alpha > 1.0) {
                    fprintf(stderr, "Erro: alpha deve estar entre 0.0 e 1.0\n");
                    return -1;
                }
                break;
            case 'g':
                cfg->gamma = atof(optarg);
                if (cfg->gamma < 0.0 || cfg->gamma >= 1.0) {
                    fprintf(stderr, "Erro: gamma deve estar entre 0.0 e 1.0 (exclusive)\n");
                    return -1;
                }
                break;
            case 'e':
                cfg->epsilon = atof(optarg);
                if (cfg->epsilon < 0.0 || cfg->epsilon > 1.0) {
                    fprintf(stderr, "Erro: epsilon deve estar entre 0.0 e 1.0\n");
                    return -1;
                }
                break;
            case 'n':
                cfg->num_episodes = atoi(optarg);
                if (cfg->num_episodes < 1) {
                    fprintf(stderr, "Erro: episodes deve ser pelo menos 1\n");
                    return -1;
                }
                break;
            case 't':
                cfg->max_steps = atoi(optarg);
                if (cfg->max_steps < 1) {
                    fprintf(stderr, "Erro: maxsteps deve ser pelo menos 1\n");
                    return -1;
                }
                break;
            case 'M':
                if (strcmp(optarg, "sqrt")       != 0 &&
                    strcmp(optarg, "statespace") != 0 &&
                    strcmp(optarg, "hogwild")    != 0) {
                    fprintf(stderr,
                        "Erro: sync-mode deve ser 'sqrt', 'statespace' ou 'hogwild'\n");
                    return -1;
                }
                strcpy(cfg->sync_mode, optarg);
                break;
            case 'i':
                cfg->sync_interval = atoi(optarg);
                if (cfg->sync_interval < 1) {
                    fprintf(stderr, "Erro: sync-interval deve ser pelo menos 1\n");
                    return -1;
                }
                break;
            case 'v':
                cfg->verbose = 1;
                cfg->quiet = 0;
                break;
            case 'S':
                cfg->step_by_step = 1;
                cfg->verbose = 1;
                cfg->quiet = 0;
                break;
            case 'q':
                cfg->quiet = 1;
                cfg->verbose = 0;
                cfg->step_by_step = 0;
                break;
            case 'T':
                cfg->show_table = 0;
                break;
            case 'P':
                cfg->show_policy = 0;
                break;
            case 'h':
                print_help(argv[0]);
                return 1;
            default:
                print_help(argv[0]);
                return -1;
        }
    }

    /* Valida que há espaço suficiente para objetivo e início */
    int total_cells = cfg->grid_rows * cfg->grid_cols;
    if (cfg->num_obstacles >= total_cells - 1) {
        fprintf(stderr, "Erro: Muitos obstáculos (%d) para o grid %dx%d (máximo: %d)\n",
                cfg->num_obstacles, cfg->grid_cols, cfg->grid_rows, total_cells - 2);
        return -1;
    }

    return 0;
}

/* =============================================================================
 * FUNÇÕES AUXILIARES
 * =============================================================================
 */

/**
 * Converte coordenadas (linha, coluna) para índice de estado único
 */
int coords_to_state(Config *cfg, int row, int col) {
    return row * cfg->grid_cols + col;
}

/**
 * Converte índice de estado para coordenadas (linha, coluna)
 */
void state_to_coords(Config *cfg, int state, int *row, int *col) {
    *row = state / cfg->grid_cols;
    *col = state % cfg->grid_cols;
}

/**
 * Gera número aleatório entre 0 e 1 (usa rand_r — reentrant, sem lock global)
 */
double random_double(QLearning *ql) {
    return (double)rand_r(&ql->rand_seed) / RAND_MAX;
}

/**
 * Gera número inteiro aleatório entre 0 e max-1 (usa rand_r — reentrant)
 */
int random_int(QLearning *ql, int max) {
    return rand_r(&ql->rand_seed) % max;
}

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

/* =============================================================================
 * ALOCAÇÃO E LIBERAÇÃO DE MEMÓRIA
 * =============================================================================
 */

/**
 * Aloca memória para a estrutura Q-Learning
 */
int qlearning_alloc(QLearning *ql, Config *cfg) {
    int i;
    int num_states = cfg->grid_rows * cfg->grid_cols;

    ql->config = cfg;
    ql->num_states = num_states;

    /* Aloca Q-table */
    ql->q_table = (double **)malloc(num_states * sizeof(double *));
    if (!ql->q_table) return -1;

    for (i = 0; i < num_states; i++) {
        ql->q_table[i] = (double *)calloc(NUM_ACTIONS, sizeof(double));
        if (!ql->q_table[i]) return -1;
    }

    /* Aloca grid */
    ql->grid = (int **)malloc(cfg->grid_rows * sizeof(int *));
    if (!ql->grid) return -1;

    for (i = 0; i < cfg->grid_rows; i++) {
        ql->grid[i] = (int *)calloc(cfg->grid_cols, sizeof(int));
        if (!ql->grid[i]) return -1;
    }

    /* Aloca lista de obstáculos (tamanho máximo = total de estados) */
    ql->obstacles = (int *)malloc(num_states * sizeof(int));
    if (!ql->obstacles) return -1;

    return 0;
}

/**
 * Libera memória da estrutura Q-Learning
 */
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

/* =============================================================================
 * INICIALIZAÇÃO E CONFIGURAÇÃO DO AMBIENTE
 * =============================================================================
 */

/**
 * Verifica se uma posição é válida para colocar obstáculo
 */
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

/**
 * Gera obstáculos aleatoriamente
 */
void generate_random_obstacles(QLearning *ql) {
    Config *cfg = ql->config;
    int placed = 0;
    int max_attempts = cfg->grid_rows * cfg->grid_cols * 10;
    int attempts = 0;

    while (placed < cfg->num_obstacles && attempts < max_attempts) {
        int state = random_int(ql, ql->num_states);

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

    if (placed < cfg->num_obstacles && !cfg->quiet) {
        printf("Aviso: Só foi possível colocar %d obstáculos (pedidos: %d)\n",
               placed, cfg->num_obstacles);
    }
}

/**
 * Inicializa a estrutura Q-Learning
 */
void qlearning_init(QLearning *ql) {
    Config *cfg = ql->config;
    int i, j;

    /* Inicializa semente de aleatoriedade (rand_r é reentrant, sem lock global) */
    ql->rand_seed = cfg->seed;

    /* Inicializa Q-table com zeros */
    for (i = 0; i < ql->num_states; i++) {
        for (j = 0; j < NUM_ACTIONS; j++) {
            ql->q_table[i][j] = 0.0;
        }
    }

    /* Inicializa o grid */
    for (i = 0; i < cfg->grid_rows; i++) {
        for (j = 0; j < cfg->grid_cols; j++) {
            ql->grid[i][j] = CELL_EMPTY;
        }
    }

    /* Define posições de início e objetivo */
    ql->start_state = 0;
    ql->goal_state = coords_to_state(cfg, cfg->grid_rows - 1, cfg->grid_cols - 1);

    ql->grid[0][0] = CELL_START;
    ql->grid[cfg->grid_rows - 1][cfg->grid_cols - 1] = CELL_GOAL;

    /* Gera obstáculos aleatórios */
    ql->num_obstacles = 0;
    generate_random_obstacles(ql);
}

/* =============================================================================
 * FUNÇÕES DO Q-LEARNING
 * =============================================================================
 */

/**
 * Verifica se um estado é válido (dentro dos limites do grid)
 */
int is_valid_state(Config *cfg, int row, int col) {
    return (row >= 0 && row < cfg->grid_rows && col >= 0 && col < cfg->grid_cols);
}

/**
 * Verifica se um estado é obstáculo
 */
int is_obstacle(QLearning *ql, int state) {
    for (int i = 0; i < ql->num_obstacles; i++) {
        if (ql->obstacles[i] == state) {
            return 1;
        }
    }
    return 0;
}

/**
 * Calcula o próximo estado dado o estado atual e uma ação
 */
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

/**
 * Calcula a recompensa para transição de estado
 */
double get_reward(QLearning *ql, int next_state) {
    Config *cfg = ql->config;

    if (next_state == ql->goal_state) {
        return cfg->reward_goal;
    } else if (is_obstacle(ql, next_state)) {
        return cfg->reward_obstacle;
    }
    return cfg->reward_step;
}

/**
 * Verifica se o episódio terminou
 */
int is_terminal_state(QLearning *ql, int state) {
    return (state == ql->goal_state);
}

/**
 * Encontra o valor máximo Q para um dado estado
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
 * Seleciona ação usando política ε-greedy
 * (cada thread chama com sua própria local_ql — sem race condition)
 */
int select_action(QLearning *ql, int state) {
    Config *cfg = ql->config;
    int action, best_action;
    double best_value;

    /* Com probabilidade ε, explora (ação aleatória) */
    if (random_double(ql) < cfg->epsilon) {
        return random_int(ql, NUM_ACTIONS);
    }

    /* Caso contrário, explota (melhor ação conhecida) */
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

/**
 * Atualiza o valor Q usando a equação de Bellman
 */
void update_q_value(QLearning *ql, int state, int action,
                    double reward, int next_state) {
    Config *cfg = ql->config;
    double current_q = ql->q_table[state][action];
    double max_next_q = get_max_q_value(ql, next_state);

    ql->q_table[state][action] = current_q +
        cfg->alpha * (reward + cfg->gamma * max_next_q - current_q);
}

/**
 * Executa um episódio completo de treinamento
 */
double run_episode(QLearning *ql, int episode_num) {
    Config *cfg = ql->config;
    int step, state, action, next_state;
    double reward, total_reward;
    int row, col;

    state = ql->start_state;
    total_reward = 0.0;

    for (step = 0; step < cfg->max_steps; step++) {
        action = select_action(ql, state);
        next_state = get_next_state(ql, state, action);
        reward = get_reward(ql, next_state);

        if (cfg->step_by_step) {
            state_to_coords(cfg, state, &row, &col);
            printf("  Ep %4d | Passo %3d | (%d,%d) -> %s -> ",
                   episode_num + 1, step + 1, row, col, action_to_string(action));
            state_to_coords(cfg, next_state, &row, &col);
            printf("(%d,%d) | R=%.1f | Q=%.2f\n",
                   row, col, reward, ql->q_table[state][action]);
        }

        update_q_value(ql, state, action, reward, next_state);
        total_reward += reward;

        if (is_terminal_state(ql, next_state)) {
            if (cfg->step_by_step) {
                printf("  >>> Objetivo alcançado em %d passos!\n\n", step + 1);
            }
            break;
        }

        state = next_state;
    }

    return total_reward;
}

/* =============================================================================
 * MÉTODOS DE TREINAMENTO PARALELO
 * =============================================================================
 */

/**
 * train_federated — federated averaging com sincronização periódica e
 * merge baseado em delta (corrige diluição por threads ociosas).
 *
 * BUG do merge absoluto:
 *   Com sync_interval < num_threads, apenas algumas threads recebem
 *   episódios do #pragma omp for. As outras ficam com a tabela do último
 *   broadcast (= snapshot). Se somarmos todas e dividirmos por num_threads,
 *   cada update real é diluído: Q_new = Q_old + Δ/num_threads.
 *   Ex: 1 episódio, 4 threads → alpha efetivo = alpha/4. [ERRADO]
 *
 * Correção delta:
 *   delta_t   = local[t] - snapshot  (0 para threads ociosas)
 *   active    = min(chunk_size, num_threads)
 *   Q_new     = snapshot + sum(delta_t) / active
 *   → threads ociosas contribuem delta=0, sem diluição.
 *   Ex: 1 episódio, 4 threads → Q_new = Q_old + Δ/1 = sequencial. [CORRETO]
 */
static void train_federated(QLearning *ql, int num_threads, int sync_interval) {
    Config *cfg = ql->config;
    int episode, t, state, action;
    double ***local_q_tables = NULL;

    int total_syncs_expected = (cfg->num_episodes + sync_interval - 1) / sync_interval;
    int num_syncs = 0;

    if (!cfg->quiet)
        printf("Sync a cada %d episodios (%d syncs no total)\n\n",
               sync_interval, total_syncs_expected);

    /* Aloca Q-tables locais — uma por thread */
    local_q_tables = (double ***)malloc(num_threads * sizeof(double **));
    if (!local_q_tables) {
        fprintf(stderr, "Erro: Falha ao alocar Q-tables locais\n");
        return;
    }
    for (t = 0; t < num_threads; t++) {
        local_q_tables[t] = (double **)malloc(ql->num_states * sizeof(double *));
        for (state = 0; state < ql->num_states; state++)
            local_q_tables[t][state] = (double *)calloc(NUM_ACTIONS, sizeof(double));
    }

    /* ==========================================================================
     * REGIÃO PARALELA: chunk → merge delta → broadcast → repete
     * ========================================================================== */
    #pragma omp parallel private(episode, state, action)
    {
        int tid = omp_get_thread_num();
        QLearning local_ql = *ql;
        local_ql.q_table   = local_q_tables[tid];
        local_ql.rand_seed = cfg->seed + (unsigned int)tid;

        for (state = 0; state < ql->num_states; state++)
            for (action = 0; action < NUM_ACTIONS; action++)
                local_ql.q_table[state][action] = ql->q_table[state][action];
        #pragma omp barrier

        int episode_start = 0;
        while (episode_start < cfg->num_episodes) {
            int episode_end = episode_start + sync_interval;
            if (episode_end > cfg->num_episodes) episode_end = cfg->num_episodes;

            /* ----------------------------------------------------------------
             * FASE 1: Treinamento — cada thread recebe seu lote via omp for
             * ---------------------------------------------------------------- */
            #pragma omp for schedule(static)
            for (episode = episode_start; episode < episode_end; episode++) {
                run_episode(&local_ql, episode);

                if (!cfg->quiet && cfg->verbose && !cfg->step_by_step
                        && (episode + 1) % 100 == 0) {
                    #pragma omp critical
                    printf("  Ep %5d | Thread %d/%d\n",
                           episode + 1, tid + 1, num_threads);
                }
            }
            /* barrier implícito após omp for */

            /* ----------------------------------------------------------------
             * FASE 2: Merge com delta
             *   active = threads que efetivamente rodaram episódios neste chunk.
             *   Com schedule(static), a distribuição é determinística:
             *   threads 0..active-1 recebem trabalho se chunk < num_threads.
             *   Threads ociosas têm local == snapshot → delta == 0 → correto.
             * ---------------------------------------------------------------- */
            int chunk_size = episode_end - episode_start;
            int active     = (chunk_size < num_threads) ? chunk_size : num_threads;

            #pragma omp for collapse(2)
            for (state = 0; state < ql->num_states; state++) {
                for (action = 0; action < NUM_ACTIONS; action++) {
                    double snapshot  = ql->q_table[state][action];
                    double delta_sum = 0.0;
                    int tt;
                    for (tt = 0; tt < num_threads; tt++)
                        delta_sum += local_q_tables[tt][state][action] - snapshot;
                    ql->q_table[state][action] = snapshot + delta_sum / active;
                }
            }
            /* barrier implícito após omp for */

            /* ----------------------------------------------------------------
             * FASE 3: Broadcast — todas as threads recebem a média atualizada
             * ---------------------------------------------------------------- */
            for (state = 0; state < ql->num_states; state++)
                for (action = 0; action < NUM_ACTIONS; action++)
                    local_ql.q_table[state][action] = ql->q_table[state][action];
            #pragma omp barrier

            #pragma omp single
            {
                num_syncs++;
                if (!cfg->quiet)
                    printf("  [Sync %3d/%d] eps %d-%d | threads ativas: %d/%d\n",
                           num_syncs, total_syncs_expected,
                           episode_start + 1, episode_end,
                           active, num_threads);
            }
            /* barrier implícito após omp single */

            episode_start = episode_end;
        }
    }
    /* ========================================================================== */

    for (t = 0; t < num_threads; t++) {
        for (state = 0; state < ql->num_states; state++)
            free(local_q_tables[t][state]);
        free(local_q_tables[t]);
    }
    free(local_q_tables);

    if (!cfg->quiet)
        printf("\nTreinamento concluido! (%d syncs realizados)\n", num_syncs);
}

/**
 * train_hogwild — treinamento lock-free com Q-table compartilhada.
 *
 * Todas as threads lêem e escrevem na mesma Q-table sem nenhum lock.
 * As corridas de dados são intencionais e inofensivas porque:
 *   1. Updates são esparsos — em grids grandes, conflitos simultâneos
 *      em Q(s,a) são raros.
 *   2. Updates são pequenos (escalados por alpha) — um overwrite ocasional
 *      é desprezível para a convergência.
 *   3. Ambos os valores concorrentes são direções válidas de aprendizado.
 *
 * schedule(dynamic) distribui episódios sob demanda para evitar que
 * múltiplas threads se concentrem na mesma região do grid ao mesmo tempo.
 *
 * Ref: Niu et al., "Hogwild!: A Lock-Free Approach to Parallelizing
 *      Stochastic Gradient Descent", NIPS 2011.
 */
static void train_hogwild(QLearning *ql, int num_threads) {
    Config *cfg = ql->config;
    int episode;

    if (!cfg->quiet)
        printf("Modo Hogwild! — Q-table compartilhada, sem barreiras\n\n");

    #pragma omp parallel private(episode)
    {
        int tid = omp_get_thread_num();

        /* Cópia local da estrutura apenas para ter rand_seed independente.
         * IMPORTANTE: q_table ainda aponta para a Q-table COMPARTILHADA. */
        QLearning local_ql    = *ql;
        local_ql.rand_seed    = cfg->seed + (unsigned int)tid * 1000003U;

        #pragma omp for schedule(dynamic)
        for (episode = 0; episode < cfg->num_episodes; episode++) {
            run_episode(&local_ql, episode);   /* lê/escreve shared q_table sem lock */

            if (!cfg->quiet && cfg->verbose && !cfg->step_by_step
                    && (episode + 1) % 100 == 0) {
                #pragma omp critical
                printf("  Ep %5d | Thread %d/%d\n",
                       episode + 1, tid + 1, num_threads);
            }
        }
    }

    if (!cfg->quiet)
        printf("\nTreinamento Hogwild! concluido!\n");
}

/**
 * train — dispatcher: seleciona o método com base em cfg->sync_mode,
 * calcula o sync_interval para modos federados e delega.
 */
void train(QLearning *ql) {
    Config *cfg = ql->config;
    int num_threads;

    if (!cfg->quiet) {
        printf("\nIniciando treinamento: %d episodios\n", cfg->num_episodes);
        printf("Hiperparametros: alpha=%.2f, gamma=%.2f, epsilon=%.2f\n",
               cfg->alpha, cfg->gamma, cfg->epsilon);
    }

    /* Determina número de threads */
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    if (!cfg->quiet)
        printf("Threads OpenMP: %d  |  Modo sync: %s\n\n",
               num_threads, cfg->sync_mode);

    /* Hogwild! não precisa de sync_interval */
    if (strcmp(cfg->sync_mode, "hogwild") == 0) {
        train_hogwild(ql, num_threads);
        return;
    }

    /* Modos federados: calcula sync_interval */
    int sync_interval = cfg->sync_interval;   /* override manual tem prioridade */
    if (sync_interval <= 0) {
        if (strcmp(cfg->sync_mode, "sqrt") == 0) {
            /* Heurística paralela: sqrt(N/T) minimiza regret bound */
            sync_interval = (int)sqrt((double)cfg->num_episodes
                                      / (double)num_threads);
            if (!cfg->quiet)
                printf("Intervalo auto (sqrt): sqrt(%d/%d) = %d\n",
                       cfg->num_episodes, num_threads, sync_interval);
        } else {
            /* statespace: um round = explorar todos os estados uma vez */
            sync_interval = ql->num_states;
            if (!cfg->quiet)
                printf("Intervalo auto (statespace): %d estados = %d\n",
                       ql->num_states, sync_interval);
        }
        if (sync_interval < num_threads) sync_interval = num_threads;
        if (sync_interval < 1)          sync_interval = 1;
    } else {
        if (!cfg->quiet)
            printf("Intervalo (manual): %d\n", sync_interval);
    }

    train_federated(ql, num_threads, sync_interval);
}

/* =============================================================================
 * FUNÇÕES DE VISUALIZAÇÃO
 * =============================================================================
 */

/**
 * Imprime a configuração atual
 */
void print_config(Config *cfg) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║               Q-LEARNING PARALELO (OpenMP) - GRID WORLD                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("CONFIGURAÇÃO:\n");
    printf("  Modo: %s\n", cfg->mode);
    printf("  Grid: %dx%d (%d estados)\n", cfg->grid_cols, cfg->grid_rows,
           cfg->grid_rows * cfg->grid_cols);
    printf("  Obstáculos: %d\n", cfg->num_obstacles);
    printf("  Seed: %u\n", cfg->seed);
    printf("  Alpha: %.3f | Gamma: %.3f | Epsilon: %.3f\n",
           cfg->alpha, cfg->gamma, cfg->epsilon);
    printf("  Episódios: %d | Max passos: %d\n", cfg->num_episodes, cfg->max_steps);

    if (strcmp(cfg->sync_mode, "hogwild") == 0) {
        printf("  Sync: hogwild (lock-free, sem barreiras)\n");
    } else if (cfg->sync_interval > 0) {
        int total = (cfg->num_episodes + cfg->sync_interval - 1) / cfg->sync_interval;
        printf("  Sync: %s | intervalo manual = %d eps (%d syncs)\n",
               cfg->sync_mode, cfg->sync_interval, total);
    } else {
        printf("  Sync: %s | intervalo automatico (calculado em train)\n",
               cfg->sync_mode);
    }
}

/**
 * Imprime o grid com posições dos obstáculos
 */
void print_grid(QLearning *ql) {
    Config *cfg = ql->config;
    int row, col, state;

    printf("\n=== GRID %dx%d ===\n", cfg->grid_cols, cfg->grid_rows);
    printf("(S=início, G=objetivo, X=obstáculo)\n\n");

    printf("+");
    for (col = 0; col < cfg->grid_cols; col++) printf("---+");
    printf("\n");

    for (row = 0; row < cfg->grid_rows; row++) {
        printf("|");
        for (col = 0; col < cfg->grid_cols; col++) {
            state = coords_to_state(cfg, row, col);

            if (state == ql->start_state) {
                printf(" S |");
            } else if (state == ql->goal_state) {
                printf(" G |");
            } else if (is_obstacle(ql, state)) {
                printf(" X |");
            } else {
                printf("   |");
            }
        }
        printf("\n+");
        for (col = 0; col < cfg->grid_cols; col++) printf("---+");
        printf("\n");
    }

    if (ql->num_obstacles > 0) {
        printf("\nObstáculos em: ");
        for (int i = 0; i < ql->num_obstacles; i++) {
            state_to_coords(cfg, ql->obstacles[i], &row, &col);
            printf("(%d,%d)", row, col);
            if (i < ql->num_obstacles - 1) printf(", ");
        }
        printf("\n");
    }
}

/**
 * Imprime a Q-table formatada
 */
void print_q_table(QLearning *ql) {
    Config *cfg = ql->config;
    int state, action;

    printf("\n=== Q-TABLE ===\n");
    printf("Estado |   CIMA   |  BAIXO   |   ESQ    |   DIR    | Melhor\n");
    printf("-------|----------|----------|----------|----------|-------\n");

    for (state = 0; state < ql->num_states; state++) {
        int row, col, best_action = 0;
        double best_value = ql->q_table[state][0];

        state_to_coords(cfg, state, &row, &col);
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
 */
void print_policy(QLearning *ql) {
    Config *cfg = ql->config;
    int row, col, state, action;
    double best_value;
    int best_action;
    char symbols[4] = {'^', 'v', '<', '>'};

    printf("\n=== POLITICA APRENDIDA ===\n");
    printf("(^ = cima, v = baixo, < = esq, > = dir)\n\n");

    printf("+");
    for (col = 0; col < cfg->grid_cols; col++) printf("---+");
    printf("\n");

    for (row = 0; row < cfg->grid_rows; row++) {
        for (col = 0; col < cfg->grid_cols; col++) {
            state = coords_to_state(cfg, row, col);

            if (state == ql->goal_state) {
                printf("| G ");
            } else if (is_obstacle(ql, state)) {
                printf("| X ");
            } else {
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
        printf("|\n+");
        for (col = 0; col < cfg->grid_cols; col++) printf("---+");
        printf("\n");
    }
}

/**
 * Obtém a melhor ação para um estado
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
 * Simula e mostra o caminho que o agente seguiria após treinamento
 */
void demonstrate_path(QLearning *ql) {
    Config *cfg = ql->config;
    int state = ql->start_state;
    int step = 0;
    int action;
    int row, col;

    printf("\n=== DEMONSTRACAO DO CAMINHO ===\n");
    printf("Caminho do agente do inicio ao objetivo:\n\n");

    while (state != ql->goal_state && step < cfg->max_steps) {
        state_to_coords(cfg, state, &row, &col);
        action = get_best_action(ql, state);

        printf("Passo %2d: Estado (%d,%d) -> Acao: %s\n",
               step + 1, row, col, action_to_string(action));

        state = get_next_state(ql, state, action);
        step++;
    }

    if (state == ql->goal_state) {
        state_to_coords(cfg, state, &row, &col);
        printf("Passo %2d: Estado (%d,%d) -> OBJETIVO ALCANCADO!\n",
               step + 1, row, col);
        printf("\nAgente encontrou o caminho em %d passos.\n", step);
    } else {
        printf("\nAgente nao conseguiu encontrar o objetivo em %d passos.\n", step);
    }
}

/* =============================================================================
 * FUNÇÃO PRINCIPAL
 * =============================================================================
 */

int main(int argc, char *argv[]) {
    Config cfg;
    QLearning ql;
    int parse_result;

    config_set_defaults(&cfg);

    parse_result = parse_arguments(argc, argv, &cfg);
    if (parse_result != 0) {
        return (parse_result > 0) ? 0 : 1;
    }

    if (!cfg.quiet) {
        print_config(&cfg);
    }

    if (qlearning_alloc(&ql, &cfg) != 0) {
        fprintf(stderr, "Erro: Falha ao alocar memória\n");
        return 1;
    }

    qlearning_init(&ql);

    if (!cfg.quiet) {
        print_grid(&ql);
    }

    train(&ql);

    if (cfg.show_table && !cfg.quiet) {
        print_q_table(&ql);
    }

    if (cfg.show_policy) {
        print_policy(&ql);
    }

    demonstrate_path(&ql);

    if (!cfg.quiet) {
        printf("\n");
        printf("════════════════════════════════════════════════════════════════════════════\n");
        printf("                              FIM DA EXECUCAO\n");
        printf("════════════════════════════════════════════════════════════════════════════\n");
    }

    qlearning_free(&ql);

    return 0;
}
