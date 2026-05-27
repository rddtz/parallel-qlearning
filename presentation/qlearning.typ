#let _ = ```typ
exec ~/.local/typst-0.12/bin/typst c "$0" --root "$(readlink -f "$0" | xargs dirname)/./"
⁠```
#set document(title: "Paralelização do Q-Learning com OpenMP", date: datetime(year: 2026, month: 5, day: 27), author: "Rayan Raddatz de Matos, Eduardo Magnus Lazuta, Enzo Lisbôa Peixoto")
#set text(lang: "pt")
#show link: set text(fill: blue, weight: 700)
#show link: underline
#set heading(numbering: "1.")
#import "@preview/touying:0.7.3": *
#import themes.simple: *
#import "@preview/numbly:0.1.0": numbly
#let talk-title = [Paralelização do Algoritmo Q-Learning com OpenMP]
#let talk-authors = "Eduardo Magnus Lazuta, Enzo Lisbôa Peixoto"
#let talk-main-author = "Rayan Raddatz de Matos"
#let talk-event = [INF/UFRGS -- Computação Paralela]
#let talk-date = "2026-05-27"
#show: simple-theme.with(
aspect-ratio: "16-9",
config-common(show-notes-on-second-screen: none),
primary: rgb("#04364A"),
config-colors(
secondary: rgb("#176B87"),
tertiary: rgb("#448C95"),
neutral-lightest: rgb("#ffffff"),
neutral-darkest: rgb("#000000"),
),
footer: self => self.info.title,
footer-right: self => context utils.slide-counter.display() + " / " + utils.last-slide-number,
config-info(
title: talk-title,
author: talk-main-author,
date: talk-date,
institution: talk-event,
),
)
#set heading(numbering: numbly("{1}.", default: "1.1"))
#slide(
config: utils.merge-dicts(
config-common(freeze-slide-counter: true),
config-store(
footer: none,
footer-right: none,
),
),
)[
#align(left)[
#stack(
dir: ttb,
spacing: 0.8em,
text(size: 0.6em, weight: "bold", fill: rgb("#A0A0A0"), talk-event + " | " + talk-date),
text(size: 1em, weight: "bold", fill: rgb("#A00000"), talk-title),
text(size: 0.9em, weight: "bold", fill: rgb("#000000"), talk-main-author) + ", " + text(size: 0.9em, talk-authors),
grid(
columns: (auto, 2.2em, auto),
align: left + horizon,
image("logos/UFRGS.png", height: 1.9cm),
[],
image("logos/INF.png", height: 1.9cm),
),
)
]
]
#set text(size: 22pt)
#heading(level: 2, outlined: false, numbering: none)[Agenda] #label("org998cd1f")
#list(list.item[Problema e Algoritmo])#list(list.item[Paralelização])#list(list.item[Metodologia Experimental])#list(list.item[Resultados])#list(list.item[Conclusão])
#heading(level: 1, outlined: false, numbering: none)[Problema e Algoritmo] #label("org873f564")
#heading(level: 2, outlined: false, numbering: none)[Q\u{2d}Learning: pathfinding em grid] #label("orgd37eac5")
Agente aprende a navegar de $(0,0)$ até $(L-1, C-1)$ em um grid $L times C$
com obstáculos. Ações: cima, baixo, esquerda, direita.

#v(0.8em)
#pause

A cada passo, a Q\u{2d}Table é atualizada pela equação de Bellman:

#v(0.4em)
#align(center, math.equation(
  $Q(s,a) <- Q(s,a) + alpha [r + gamma max_(a') Q(s',a') - Q(s,a)]$
))

#v(0.8em)
#pause

Após o treinamento, a política ótima é extraída como
$pi(s) = arg max_a Q(s,a)$.
#heading(level: 2, outlined: false, numbering: none)[Estrutura de um episódio] #label("orgcdbe9b0")
#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
[
#set list(spacing: 0.7em)
#list(
  [Parte do estado inicial $s_0$],
  [Seleciona ação via epsilon-greedy],
  [Transita para $s'$, recebe recompensa $r$],
  [Atualiza $Q(s,a)$ pela equação de Bellman],
  [Repete até atingir o objetivo ou esgotar passos],
)
],
[
#set text(size: 0.75em)
Recompensas configuradas:
#v(0.4em)
- $R_"objetivo"$ = $+100$
- $R_"obstaculo"$ = $-100$
- $R_"passo"$ = $-1$
- $R_"loop"$ = $-10$
#v(0.6em)
Grids testados: *10x10, 50x50, 100x100*
]
)
#heading(level: 1, outlined: false, numbering: none)[Paralelização] #label("orga50fff6")
#heading(level: 2, outlined: false, numbering: none)[Qual nível de paralelismo?] #label("org43eae18")
Três opções foram consideradas:

#v(0.6em)

#list(list.item[#text(weight: "bold", [Ações]) (argmax de 4 ações): descartada \u{2d}\u{2d} overhead supera o ganho])

#v(0.6em)
#pause

#list(list.item[#text(weight: "bold", [Steps]) (passos do episódio): descartada \u{2d}\u{2d} cada step depende do anterior])

#v(0.6em)
#pause

#list(list.item[#text(weight: "bold", [Episódios]): #text(weight: "bold", [escolhida]) \u{2d}\u{2d} episódios são independentes; distribuição direta via #raw("omp for")])

#v(0.6em)
#pause

Mas paralelizar sem sincronização não basta: com $T$ threads e $E$ episódios,
cada thread aprende como se tivesse apenas $E slash T$ episódios.
#heading(level: 2, outlined: false, numbering: none)[Sincronização Periódica] #label("org29f6539")
Cada thread mantém uma #text(weight: "bold", [Q\u{2d}Table local]), treina um lote de #raw("sync_interval") episódios
e funde os deltas na Q\u{2d}Table global:

#v(0.5em)

#text(size: 0.68em)[
```c
#pragma omp for schedule(runtime)
for (episode = ep_start; episode < ep_end; episode++)
    run_episode(&local_ql, episode);

#pragma omp for collapse(2)
for (state ...) for (action ...) {
    delta_sum = sum_t(local[t][s][a] - snapshot);
    ql->q_table[state][action] = snapshot + delta_sum / active;
}
```
]

#v(0.3em)

#text(weight: "bold", [Merge por delta]): divide pelas threads #text(weight: "bold", [ativas]) (#raw("active = min(chunk, T)")),
preservando o $alpha$ efetivo quando #raw("chunk_size < T").
#heading(level: 2, outlined: false, numbering: none)[Heurísticas de sync\u{5f}interval] #label("orgda020dc")
Duas estratégias automáticas implementadas em #raw("train()"):

#v(0.8em)

#list(list.item[#text(weight: "bold", [sqrt]): #raw("sync_interval = sqrt(N/T)") \u{2d}\u{2d} equilibra overhead de barreira e divergência local (inspirado em limites de regret para SGD paralelo)])

#v(0.8em)
#pause

#list(list.item[#text(weight: "bold", [statespace]): #raw("sync_interval = L * C") \u{2d}\u{2d} cada thread explora todos os estados antes de fundir])

#v(0.8em)
#pause

Além disso, testamos valores fixos: #text(weight: "bold", [every10]) (#raw("i=10")) e #text(weight: "bold", [every1000]) (#raw("i=1000")).
#heading(level: 2, outlined: false, numbering: none)[HOGWILD!] #label("org1a3f3b3")
Todas as threads leem e escrevem na #text(weight: "bold", [mesma Q\u{2d}Table sem nenhuma sincronização]):

#v(0.5em)

#text(size: 0.72em)[
```c
QLearning local_ql = *ql;
local_ql.rand_seed = seed + tid;
/* q_table aponta para a tabela COMPARTILHADA */

#pragma omp for schedule(runtime)
for (episode = 0; episode < N; episode++)
    run_episode(&local_ql, episode);  /* sem sincronizacao */
```
]

#v(0.5em)
#pause

Race conditions são \u{22}intencionais\u{22} e \u{22}inofensivas\u{22} para casos esparsos.
#heading(level: 1, outlined: false, numbering: none)[Metodologia Experimental] #label("orgcdd4f6c")
#heading(level: 2, outlined: false, numbering: none)[Fatores do experimento] #label("orga8171e7")
#{
  set text(size: 0.85em)
  table(
    columns: (auto, auto, 1fr),
    stroke: none,
    table.hline(),
    table.header([*Fator*], [*Níveis*], [*Motivação*]),
    table.hline(),
    [Threads],        [1, 2, 4, 8, 16, 20, 40],        [Do serial ao hyperthreading completo],
    [Tamanho grid],   [10, 50, 100],                    [Espaço de estados pequeno/médio/grande],
    [Modo de sync],   [sqrt, statespace, hogwild, 10, 1000],  [5 estratégias de paralelização],
    [Episódios],      [5000, 10000],                    [5k instável, 10k converge melhor],
    [Obstáculos],     [0%, 50%],                        [Grid livre vs. alta densidade],
    [Schedule OMP],   [static, guided, dynamic],        [Políticas de distribuição],
    [Chunk],          [0 (auto), 1, 10],                [Granularidade do chunking],
    table.hline(),
  )
}

#v(0.6em)

#text(weight: "bold", [Baseline]): binário sequencial puro (sem runtime OpenMP) para $T=1$.
#heading(level: 2, outlined: false, numbering: none)[Afinidade de threads e medição de tempo] #label("orgce8a827")
#text(weight: "bold", [Afinidade:])

#list(list.item[$T <= 20$: #raw("OMP_PLACES=cores"), #raw("OMP_PROC_BIND=close") \u{2d}\u{2d} 1 thread por núcleo físico])#list(list.item[$T = 40$: #raw("OMP_PLACES=threads") \u{2d}\u{2d} hyperthreading completo])

#v(1em)

#text(weight: "bold", [Medição de tempo:])

#list(list.item[Apenas o tempo de #raw("train()") \u{2d}\u{2d} exclui alocação, init e I\u{2f}O])#list(list.item[Paralelo: #raw("omp_get_wtime()") | Sequencial: #raw("clock_gettime(CLOCK_MONOTONIC)")])#list(list.item[Múltiplos blocos de replicação para estimar variância do escalonador])
#heading(level: 1, outlined: false, numbering: none)[Resultados] #label("org8252352")
#heading(level: 2, outlined: false, numbering: none)[Speedup por modo de sincronização] #label("org1fa8fba")
#align(center, image("figs/speedup_blocks.svg", width: 200%, height: 10cm, fit: "contain"))
#heading(level: 2, outlined: false, numbering: none)[Speedup: observações] #label("orge7cc8db")
#list(list.item[Quase ideal para 8 threads ou menos em alguns casos (hogwild, every1000).])

#v(0.7em)
#pause

#list(list.item[#text(weight: "bold", [10x10]): satura em ~4\u{2d}\u{2d}5x a partir de 8 threads; trabalho por thread insuficiente para amortizar overhead])

#v(0.7em)
#pause

#list(list.item[#text(weight: "bold", [100x100]): HOGWILD! atinge até #text(weight: "bold", [17x com 40 threads]); modos sincronizados ficam abaixo de 10x])

#v(0.7em)
#pause

#list(list.item[Barreiras periódicas serializam parcialmente a execução nos modos sincronizados])
#heading(level: 2, outlined: false, numbering: none)[Eficiência paralela S(T)\u{2f}T] #label("org20b9459")
#align(center, image("figs/efficiency.svg", width: 200%, height: 10cm, fit: "contain"))
#heading(level: 2, outlined: false, numbering: none)[Eficiência: observações] #label("org8b5d592")
#list(list.item[HOGWILD! mantém a #text(weight: "bold", [melhor eficiência]) em todos os grids: ~35\u{2d}\u{2d}45% com 40 threads no 100x100])

#v(0.7em)
#pause

#list(list.item[#text(weight: "bold", [every10]) (sync a cada 10 eps): #text(weight: "bold", [pior eficiência]) pois overhead de barreiras frequentes supera o ganho])

#v(0.7em)
#pause

#list(list.item[every1000 e statespace ficam no intermediário: sincronizam menos e amortizam o custo])

#v(0.7em)
#pause

#list(list.item[Nenhuma configuração sustenta 50% de eficiência além de 8 threads no grid 10x10])
#heading(level: 2, outlined: false, numbering: none)[Convergência] #label("orgd5c8ced")
#align(center, image("figs/conv_obstacles.svg", width: 200%, height: 11cm, fit: "contain"))
#heading(level: 2, outlined: false, numbering: none)[Convergência: causa da degradação] #label("org75b2e7b")
Com $T$ threads e $E$ episódios fixos, cada thread treina com $E slash T$ episódios.
Quando $E slash T < E_"min"$ o agente não aprende.

#v(0.8em)
#pause

#list(list.item[#text(weight: "bold", [0% de obstáculos]): todos os modos convergem para poucos threads; degrada com o aumento])#list(list.item[#text(weight: "bold", [50% de obstáculos]): nenhuma configuração paralela convergiu para grids grandes])

#v(0.8em)
#pause

Mesmo HOGWILD! (que acumula updates de #text(weight: "bold", [todas]) as threads) falha com 50%
de obstáculos, provavelmente pela perda de \u{22}esparsidade\u{22}.
#heading(level: 2, outlined: false, numbering: none)[Causa da falha com 50% de obstáculos (10.000 episódios)] #label("orge4144f7")
#text(weight: "bold", [Causa raiz: desconto gamma anula o sinal de recompensa])

Com 5000 obstáculos, o caminho mínimo tem ~300 passos. Com $gamma = 0.9$:

#align(center, math.equation($gamma^300 = 0.9^300 approx 0$))

A recompensa do goal #text(weight: "bold", [não chega ao estado inicial]). Mesmo em episódios bem\u{2d}sucedidos, Q\u{2d}target em $s_0 approx -10$ (ainda negativo) — aumentar episódios não resolve.

#v(0.3em)
#pause

#text(weight: "bold", [Medido empiricamente com 10.000 episódios e T=2:])

#list(list.item[570 episódios chegaram ao goal — Q($s_0$) ficou preso em $-83$ a $-87$ durante todas as 143 sincronizações])#list(list.item[Episódios falhos (9430) dominam: cada um empurra Q($s_0$) para $-90$; cada bem\u{2d}sucedido empurra para $-10$])

#v(0.3em)
#pause

#text(weight: "bold", [Por que o paralelo falha universalmente e o sequencial às vezes não:])

#list(list.item[Sequencial com certas seeds tem sequência de exploração \u{22}sortuda\u{22} que aprende uma rota local próxima ao goal])#list(list.item[O paralelo divide episódios entre threads e funde Q\u{2d}tables, quebrando essa sequência e tornando a falha consistente])
#heading(level: 2, outlined: false, numbering: none)[Speedup: execuções convergidas] #label("orge5bf48e")
#{
  set text(size: 0.62em)
  v(0.3em)
  grid(
    columns: (1fr, 1fr),
    gutter: 1.4em,
    table(
      columns: (auto, auto, auto, auto, auto, auto, auto, auto),
      stroke: none,
      table.hline(),
      table.header([*Grid*],[*Config*],[*2*],[*4*],[*8*],[*16*],[*20*],[*40*]),
      table.hline(),
      [10x10],[every10],   [1,54],[1,63],[1,47],[0,59],[0,51],[0,49],
      [10x10],[every1000], [*2,95*],[*4,77*],[*7,11*],[7,09],[6,69],[4,73],
      [10x10],[hogwild],   [*2,43*],[*3,67*],[5,11],[3,00],[2,91],[2,78],
      [10x10],[sqrt],      [*2,62*],[3,37],[3,42],[1,29],[0,97],[1,32],
      [10x10],[statespace],[*2,75*],[*3,95*],[5,07],[3,51],[3,07],[2,34],
      table.hline(),
    ),
    table(
      columns: (auto, auto, auto, auto, auto, auto, auto, auto),
      stroke: none,
      table.hline(),
      table.header([*Grid*],[*Config*],[*2*],[*4*],[*8*],[*16*],[*20*],[*40*]),
      table.hline(),
      [50x50],[every10],   [1,08],[1,11],[0,98],[0,65],[0,59],[0,38],
      [50x50],[every1000], [*2,21*],[2,96],[3,76],[---], [---], [---],
      [50x50],[hogwild],   [*2,32*],[*4,04*],[*7,40*],[9,19],[10,12],[10,65],
      [50x50],[sqrt],      [1,98],[2,33],[2,20],[1,19],[---], [---],
      [50x50],[statespace],[*2,20*],[2,91],[3,58],[4,16],[---], [---],
      table.hline(),
      [100x100],[every10],   [0,90],[0,85],[---],[---],[---],[---],
      [100x100],[every1000], [*1,97*],[2,49],[---],[---],[---],[---],
      [100x100],[hogwild],   [*2,43*],[*4,24*],[*8,00*],[11,45],[13,88],[18,14],
      [100x100],[sqrt],      [*1,77*],[1,99],[---],[---],[---],[---],
      table.hline(),
    ),
  )
  v(0.35em)
  text(size: 0.55em, fill: luma(130))[Speedup medio (10k ep., 0% obstaculos). "---" = nenhuma execucao convergiu.]
}
#heading(level: 2, outlined: false, numbering: none)[VTune: IPC e Memory Bound] #label("org6146611")
#align(center, image("figs/vtune.svg", width: 200%, height: 11cm, fit: "contain"))
#heading(level: 2, outlined: false, numbering: none)[Análise de Cache (10x10)] #label("org828b56a")
Q\u{2d}Table do 10x10 ocupa #text(weight: "bold", [3,2 KB]) (100 estados x 4 ações x 8 bytes) \u{2d}\u{2d}
cabe inteira em L1. Taxa de LLC\u{2d}miss medida com #raw("perf stat"):

#v(0.3em)
#{
  set text(size: 0.72em)
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    stroke: none,
    table.hline(),
    table.header([*T*], [*sq-L1*], [*sq-LLC*], [*spc-L1*], [*spc-LLC*], [*hog-L1*], [*hog-LLC*]),
    table.hline(),
    [2],  [1,12%], [0,80%], [1,10%], [1,11%], [7,42%],  [0,44%],
    [4],  [3,87%], [0,70%], [4,10%], [0,79%], [8,53%],  [0,18%],
    [8],  [6,37%], [0,17%], [5,15%], [0,18%], [23,20%], [0,11%],
    table.hline(),
    [16], [3,95%], [30,9%], [4,64%], [23,1%], [5,99%],  [36,7%],
    [20], [5,08%], [29,1%], [11,09%],[22,9%], [1,38%],  [36,4%],
    [40], [1,99%], [40,7%], [1,98%], [28,5%], [4,17%],  [35,9%],
    table.hline(),
  )
}

Piora quando o número de threads ultrapassa 8, pois o gargalo se torna #text(weight: "bold", [tráfego de cache]), não falta de trabalho.
#heading(level: 2, outlined: false, numbering: none)[Análise de Cache: interpretação] #label("orgc7111f6")
#text(weight: "bold", [Para $T <= 8$]): Todos modos tem LLC\u{2d}miss #text(weight: "bold", [menor]) pois Q\u{2d}Table única de 3,2 KB
fica estável em L1\u{2f}L2. Modos sincronizados tem um miss L1 um pouco
maior pois tem várias cópias das tableas.

#pause

#text(weight: "bold", [Sockets]): Hype é dual\u{2d}socket (2x Xeon E5\u{2d}2650 v3, 10
núcleos\u{2f}socket). Com #raw("OMP_PROC_BIND=close"), $T <= 10$ fica inteiro no
socket 0; $T = 16$ já usa 6 núcleos do socket 1. Por isso vemos mais
LLC\u{2d}miss.

#pause

#text(weight: "bold", [Para $T >= 16$]): hogwild é o pior (36,7%), sem cópia local, acessa
frequentemente o socket 1. Statespace fica em 23,1% porque os threads
do socket 1 carregam a cópia local uma vez e a reutilizam por 100
episódios.
#heading(level: 2, outlined: false, numbering: none)[Análise de Cache (100x100)] #label("orgb7a2e6e")
Q\u{2d}Table ocupa #text(weight: "bold", [320 KB]), excede L1 (32 KB) e L2 (~256 KB), cabe no LLC
(~20 MB). Baseline sequencial: 2,82% de L1\u{2d}miss (vs. 0,18% no 10x10).

#v(0.3em)
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
[
#set text(size: 0.65em)
*L1-miss*
#table(
  columns: (auto, auto, auto, auto, auto, auto),
  stroke: none,
  table.hline(),
  table.header([*T*], [*sqrt*], [*ev10*], [*ev1k*], [*spc*], [*hog*]),
  table.hline(),
  [2],  [5,63%],  [10,27%], [3,39%], [3,28%], [4,50%],
  [4],  [6,27%],  [10,77%], [2,91%], [2,62%], [4,30%],
  [8],  [7,80%],  [11,59%], [2,64%], [2,09%], [5,41%],
  table.hline(),
  [16], [11,79%], [13,74%], [1,98%], [2,18%], [4,36%],
  [20], [12,85%], [14,59%], [2,83%], [2,17%], [3,78%],
  [40], [11,03%], [12,43%], [3,01%], [2,01%], [2,56%],
  table.hline(),
)
],
[
#set text(size: 0.65em)
*LLC-miss*
#table(
  columns: (auto, auto, auto, auto, auto, auto),
  stroke: none,
  table.hline(),
  table.header([*T*], [*sqrt*], [*ev10*], [*ev1k*], [*spc*], [*hog*]),
  table.hline(),
  [2],  [0,06%], [0,01%], [0,08%], [0,08%], [0,00%],
  [4],  [0,12%], [0,03%], [0,29%], [0,05%], [0,00%],
  [8],  [0,01%], [0,02%], [0,04%], [0,05%], [0,20%],
  table.hline(),
  [16], [7,01%],  [6,81%],  [3,12%],  [1,59%],  [35,44%],
  [20], [22,43%], [23,00%], [14,03%], [5,43%],  [33,49%],
  [40], [27,65%], [28,38%], [16,47%], [2,61%],  [60,77%],
  table.hline(),
)
],
)

Media dos tres schedules.
#heading(level: 2, outlined: false, numbering: none)[Análise de Cache 100x100: interpretação] #label("orgb1ee384")
#list(list.item[$T <= 8$: todos no socket 0, LLC\u{2d}miss $<$ 0,3% em todos os modos])

#v(0.45em)
#pause

#list(list.item[#text(weight: "bold", [Hogwild]) (60,8% com 40T): sem cópia local, cada escrita do socket 1
invalida linhas no socket 0])

#v(0.45em)
#pause

#list(list.item[#text(weight: "bold", [Statespace]) (2,6% com 40T): intervalo $= 10^4$, episódios = uma sync
total; threads do socket 1 carregam cópia local uma vez e não voltam])

#v(0.45em)
#pause

#list(list.item[#text(weight: "bold", [Every1000]) (16,5%) melhor que sqrt (27,7%) \u{2d}\u{2d} menos cruzamentos de QPI por merge])

#v(0.45em)
#pause

#list(list.item[L1\u{2d}miss alto no #text(weight: "bold", [every10]) (10\u{2d}15%): copia 320 KB a cada 10 episódios,
expulsa dados úteis do L1 repetidamente])
#heading(level: 2, outlined: false, numbering: none)[Schedule, chunk e obstáculos] #label("orgd4a8db3")
#grid(
  columns: (1fr, 1fr),
  gutter: 0.8em,
  align(center, image("figs/sched_impact.svg", width: 100%, height: 9cm, fit: "contain")),
  align(center, image("figs/obs_speedup.svg",  width: 100%, height: 9cm, fit: "contain")),
)
#heading(level: 2, outlined: false, numbering: none)[Schedule e chunk: não são fatores dominantes] #label("orgafd4439")
#list(list.item[Os três schedules produzem speedup #text(weight: "bold", [praticamente idêntico]): durações de episódio
homogêneas, carga equilibrada independente da política])

#v(0.7em)

#list(list.item[Diferença geral permanece #text(weight: "bold", [\u{3c}2% no speedup médio])])

#v(0.7em)

#list(list.item[Chunk=1: leve degradação em grids pequenos (overhead por episódio);
chunk=10 e auto se comportam de forma equivalente])

#v(0.7em)

#list(list.item[Obstáculos: comparar speedup entre 0% e 50% é #text(weight: "bold", [injusto]), ganho aparente com 50%
reflete apenas que há mais trabalho total a dividir])
#heading(level: 1, outlined: false, numbering: none)[Conclusão] #label("org5a11f42")
#heading(level: 2, outlined: false, numbering: none)[Conclusão] #label("org48d09f4")
#list(list.item[#text(weight: "bold", [Gargalos principal]): divisão de episódios e miss. Com $T$ threads e $E$ fixos,
cada thread treina com $E slash T$ \u{2d}\u{2d} abaixo do limiar, agente não converge])#list(list.item[#text(weight: "bold", [Problemas de memória entre cache]): gargalo é coerência de cache, LLC\u{2d}miss explode
de \u{3c}1,2% para 23\u{2d}40% após $T=8$])#list(list.item[#text(weight: "bold", [HOGWILD!]) obteve maior speedup bruto (até #text(weight: "bold", [17x]) com 40 threads em 100x100),
mas não converge com 50% de obstáculos])#list(list.item[Modos sincronizados com intervalos longos (#text(weight: "bold", [every1000]), #text(weight: "bold", [statespace])) preservam
melhor o comportamento de cache entre syncs, mas convergem menos])#list(list.item[Schedule e chunk: #text(weight: "bold", [não são fatores dominantes]) (\u{3c}2% de impacto no speedup)])
#heading(level: 2, outlined: false, numbering: none)[] #label("orgba79afb")
#text(size: 2.8em, weight: "bold", "Perguntas?")
