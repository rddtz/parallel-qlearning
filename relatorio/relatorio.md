# Apresentação do algoritmo
## Explicação do algoritmo
O algoritmo Q-Learning é um método de aprendizado por reforço no qual um agente aprende a alcançar um objetivo final utilizando uma tabela chamada Q-Table. Essa tabela armazena valores para cada par de estado e ação, indicando o quão boa é determinada ação em um estado específico. Com base nesses valores, o agente tende a escolher as ações que possuem as maiores recompensas esperadas.

Para construir essa tabela, o agente precisa passar por um processo de treinamento composto por vários episódios. Em cada episódio, o agente executa uma sequência de passos (steps). A cada passo, uma ação é escolhida, o ambiente retorna uma recompensa, e o valor correspondente na Q-Table é atualizado. Ao longo do treinamento, o agente aprende quais ações produzem os melhores resultados para alcançar o objetivo desejado.

## Problema escolhido
O algoritmo Q-Learning foi escolhido para resolver um problema de pathfinding em um mundo em formato de grade. O desafio consiste em fazer um agente chegar ao ponto final utilizando apenas quatro ações de movimento: mover-se para cima, para baixo, para a esquerda e para a direita.

## Pseudocódigo

```
    1 ALGORITMO: Q-Learning Sequencial
    2 -------------------------------------------------------------------------
    3 ENTRADAS:
    4     Grid(L, C), Obstáculos(O), Episódios(E), Max_Passos(P)
    5     α (Aprendizado), γ (Desconto), ε (Exploração)
    6
    7 1. INICIALIZAÇÃO E AMBIENTE
    8    |  Alocar dinamicamente Q_Table[Estados][Ações]
    9    |  Preencher Q_Table com zeros
   10    |  Gerar Grid(L, C) com:
   11    |    - Estado_Inicial s0 em (0, 0)
   12    |    - Estado_Objetivo g em (L-1, C-1)
   13    |    - Posicionar O obstáculos aleatórios (evitando s0 e g)
   14    |  Configurar Recompensas:
   15    |    - R_objetivo (+100)
   16    |    - R_obstáculo (-100)
   17    |    - R_passo (-1)
   18    |    - R_loop (-10)
   19
   20 2. NÚCLEO DE APRENDIZADO (Treinamento)
   21    |  PARA cada episódio de 1 até E:
   22    |  |  s ← s0 (Estado Inicial)
   23    |  |  PARA cada passo de 1 até P:
   24    |  |  |  
   25    |  |  |  // Seleção de Ação (ε-Greedy)
   26    |  |  |  SE Random(0, 1) < ε ENTÃO:
   27    |  |  |  |  a ← Escolher_Ação_Aleatória()
   28    |  |  |  SENÃO:
   29    |  |  |  |  a ← argmax_a' Q[s][a'] (Melhor ação conhecida)
   30    |  |  |  
   31    |  |  |  // Interação com o Ambiente
   32    |  |  |  s' ← Calcular_Próximo_Estado(s, a)
   33    |  |  |  r  ← Avaliar_Recompensa(s')
   34    |  |  |  
   35    |  |  |  SE s' já foi visitado neste episódio ENTÃO:
   36    |  |  |  |  r ← r + R_loop  // Penalidade de ciclo
   37    |  |  |  
   38    |  |  |  // Atualização de Bellman (Aprendizado Temporal)
   39    |  |  |  Max_Q_Futuro ← Máximo(Q[s'][todas_ações])
   40    |  |  |  Q[s][a] ← Q[s][a] + α * (r + γ * Max_Q_Futuro - Q[s][a])
   41    |  |  |  
   42    |  |  |  // Transição de Estado
   43    |  |  |  s ← s'
   44    |  |  |  
   45    |  |  |  SE s == g ENTÃO:
   46    |  |  |  |  Encerrar Episódio (Objetivo Alcançado)
   47    |  |  FIM PARA
   48    |  FIM PARA
   49
   50 3. PÓS-PROCESSAMENTO E SAÍDA
   51    |  Política_Ótima π(s) ← argmax_a Q[s][a]
   52    |  Imprimir Visualização do Grid com π(s)
   53    |  Simular Caminho Final do Agente
   54    |  Liberar Memória Alocada
   55 -------------------------------------------------------------------------

```
   
# Possibilidade de Paralelização
Com base no código do Q-Learning sequencial, foram identificadas três possíveis abordagens de paralelização:

Paralelização da obtenção da melhor ação;
Paralelização dos steps;
Paralelização dos episódios.

A paralelização da obtenção da melhor ação foi descartada, pois o número de ações disponíveis para o agente é relativamente pequeno, não proporcionando ganho de desempenho significativo que justificasse o custo da paralelização.

A paralelização dos steps também foi descartada, uma vez que cada step depende diretamente do estado atual, e o estado atual é resultado da ação executada no step anterior. Dessa forma, existe uma dependência verdadeira entre as execuções, tornando necessária a sincronização entre as threads e reduzindo os benefícios da paralelização.

Por fim, a paralelização dos episódios foi escolhida como a abordagem mais adequada, pois cada episódio pode ser executado de forma independente. Isso reduz problemas de dependência entre threads e permite melhor aproveitamento do paralelismo durante o treinamento do agente.

# Maneiras de Paralelizar os Episódios
Como dito anteriormente, a paralelização dos episódios foi escolhida como a mais adequada, porém existe duas possíveis de parelizar os episódios:
    - A primeira maneira é que cada thread vai ter sua Q-Table Local e dado tantos episódios, vai sincronizar com a Q-Table Global (a que será aprendida no final);
    - A segunda maneira é que cada thread vai acessar e atualizar diretamente na Q-Table Global. Essa método é chamado de Hogwild;

## Com sincronização (Treinamento Federado)
A primeira abordagem de paralelização utiliza o conceito de **Treinamento Federado (Federated Averaging)**. Nesta técnica, cada thread mantém sua própria cópia local da Q-Table, permitindo que o treinamento ocorra de forma totalmente independente durante um intervalo definido de episódios.

### Funcionamento do Algoritmo
O processo segue um ciclo de três fases:

1.  **Fase de Treinamento Local**: Cada thread recebe um lote de episódios (definido pelo `sync_interval`) e atualiza sua Q-Table local independentemente.
2.  **Fase de Agregação (Merge)**: Após o término do lote, as threads sincronizam seus conhecimentos. A Q-Table Global é atualizada calculando-se a média das variações (deltas) aprendidas por cada thread.
3.  **Fase de Transmissão (Broadcast)**: A nova Q-Table Global consolidada é copiada de volta para todas as Q-Tables locais, garantindo que o próximo lote comece com o conhecimento compartilhado.

### Implementação com OpenMP
A função `train_federated` gerencia esse ciclo. Abaixo, detalhamos os principais blocos de código:

**1. Inicialização e Barreira de Entrada:**
A região paralela é iniciada e cada thread configura seu ambiente local. Uma barreira é utilizada para garantir que todas as threads iniciem o treinamento simultaneamente.
```C
#pragma omp parallel private(episode, state, action)
{
    int tid = omp_get_thread_num();
    QLearning local_ql = *ql; // Copia a estrutura (parâmetros)
    local_ql.q_table = local_q_tables[tid]; // Aponta para a tabela local dedicada

    // Sincronização inicial: cópia da global para a local
    for (state = 0; state < ql->num_states; state++)
        for (action = 0; action < NUM_ACTIONS; action++)
            local_ql.q_table[state][action] = ql->q_table[state][action];
    
    #pragma omp barrier
```

**2. Escalonamento de Episódios:**
Utilizamos a diretiva `#pragma omp for schedule(runtime)` para distribuir os episódios do lote atual entre as threads. A cláusula `runtime` permite ajustar o escalonamento (static, dynamic ou guided) via variáveis de ambiente.
```C
#pragma omp for schedule(runtime)
for (episode = episode_start; episode < episode_end; episode++) {
    run_episode(&local_ql, episode);
}
// Barreira implícita aqui garante que o treino do lote termine
```

**3. Sincronização por Delta :**
Após a execução dos episódios, é realizada a atualização da Q-Table global a partir das Q-Tables locais. Para isso, foi utilizada a diretiva #pragma omp for collapse(2), responsável por achatar os dois laços for em um único laço e distribuir suas iterações entre as threads.

Durante o desenvolvimento, um dos principais problemas encontrados foi a diluição do aprendizado. Como cada thread mantém sua própria Q-Table local, descobertas relevantes feitas por uma thread acabavam sendo reduzidas durante o processo de merge devido à influência dos valores das demais Q-Tables.

Para solucionar esse problema, foi implementado um método de merge baseado na diferença entre as Q-Tables locais e a Q-Table global, considerando apenas o número de threads ativas no cálculo. Dessa forma, as atualizações produzidas por uma thread não são penalizadas pelas demais, tornando o comportamento do algoritmo mais próximo da versão sequencial e preservando melhor o aprendizado obtido durante a execução paralela.
```C
#pragma omp for collapse(2)
for (state = 0; state < ql->num_states; state++) {
    for (action = 0; action < NUM_ACTIONS; action++) {
        double snapshot = ql->q_table[state][action];
        double delta_sum = 0.0;
        for (int tt = 0; tt < num_threads; tt++)
            delta_sum += local_q_tables[tt][state][action] - snapshot;
        
        // Atualiza a global pela média dos deltas das threads ativas
        ql->q_table[state][action] = snapshot + (delta_sum / active);
    }
}
```

## Heurísticas de Intervalo de Sincronização
O sucesso do treinamento paralelo depende do equilíbrio entre a frequência de sincronização e a velocidade de execução. Sincronizar muito gera *overhead* de comunicação; sincronizar pouco pode levar à divergência. Implementamos duas heurísticas automáticas que tentam encontrar esse equilíbrio:

### 1. Heurística SQRT (Raiz Quadrada)
Inspirada em limites teóricos de arrependimento (*regret bounds*) em aprendizado paralelo, esta heurística calcula o intervalo como:
$$Intervalo = \sqrt{\frac{N}{T}}$$
Onde $N$ é o número total de episódios e $T$ é o número de threads. Ela busca um equilíbrio matemático que minimiza o erro acumulado ao longo do tempo.

### 2. Heurística Statespace (Espaço de Estados)
Baseia-se na complexidade do ambiente. O intervalo de sincronização é igual ao número total de estados no grid:
$$Intervalo = Largura \times Altura$$
A intuição é que cada thread deve ter a oportunidade de (em média) explorar cada estado uma vez antes de fundir seu conhecimento com as demais, promovendo uma exploração mais diversificada no início do treino.

## Hogwild 

# Resultados 

