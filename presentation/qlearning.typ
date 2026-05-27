#let _ = ```typ
exec ~/.local/typst-0.12/bin/typst c "$0" --root "$(readlink -f "$0" | xargs dirname)/./"
⁠```
#set document(title: "Paralelizacao do Q-Learning com OpenMP", date: datetime(year: 2026, month: 5, day: 27), author: "Rayan Raddatz de Matos, Eduardo Magnus Lazuta, Enzo Lisboa Peixoto")
#set text(lang: "pt")
#show link: set text(fill: blue, weight: 700)
#show link: underline
#set heading(numbering: "1.")
#import "@preview/touying:0.7.3": *
#import themes.simple: *
#import "@preview/numbly:0.1.0": numbly
#let talk-title = [Paralelizacao do Algoritmo Q-Learning com OpenMP]
#let talk-authors = "Eduardo Magnus Lazuta, Enzo Lisboa Peixoto"
#let talk-main-author = "Rayan Raddatz de Matos"
#let talk-event = [INF/UFRGS -- Computacao Paralela]
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
#heading(level: 2, outlined: false, numbering: none)[Agenda] #label("orgff3b879")
#list(list.item[Problema e Algoritmo])#list(list.item[Paralelizacao])#list(list.item[Metodologia Experimental])#list(list.item[Resultados])#list(list.item[Conclusao])
#heading(level: 1, outlined: false, numbering: none)[Problema e Algoritmo] #label("org36a4a5b")
#heading(level: 2, outlined: false, numbering: none)[Q\u{2d}Learning: pathfinding em grid] #label("orgfe46c10")
Agente aprende a navegar de $(0,0)$ ate $(L-1, C-1)$ em um grid $L times C$
com obstaculos. Acoes: cima, baixo, esquerda, direita.

#v(0.8em)
#pause

A cada passo, a Q\u{2d}Table e atualizada pela equacao de Bellman:

#v(0.4em)
#align(center, math.equation(
  $Q(s,a) <- Q(s,a) + alpha [r + gamma max_(a') Q(s',a') - Q(s,a)]$
))

#v(0.8em)
#pause

Apos o treinamento, a politica otima e extraida como
$pi(s) = arg max_a Q(s,a)$.
#heading(level: 2, outlined: false, numbering: none)[Estrutura de um episodio] #label("orgb12a093")
#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
[
#set list(spacing: 0.7em)
#list(
  [Parte do estado inicial $s_0$],
  [Seleciona acao via epsilon-greedy],
  [Transita para $s'$, recebe recompensa $r$],
  [Atualiza $Q(s,a)$ pela equacao de Bellman],
  [Repete ate atingir o objetivo ou esgotar passos],
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
#heading(level: 1, outlined: false, numbering: none)[Paralelizacao] #label("org2163fd6")
#heading(level: 2, outlined: false, numbering: none)[Qual nivel de paralelismo?] #label("org560b712")
Tres opcoes foram consideradas:

#v(0.6em)

#list(list.item[#text(weight: "bold", [Acoes]) (argmax de 4 acoes): descartada \u{2d}\u{2d} overhead supera o ganho])

#v(0.6em)
#pause

#list(list.item[#text(weight: "bold", [Steps]) (passos do episodio): descartada \u{2d}\u{2d} cada step depende do anterior])

#v(0.6em)
#pause

#list(list.item[#text(weight: "bold", [Episodios]): #text(weight: "bold", [escolhida]) \u{2d}\u{2d} episodios sao independentes; distribuicao direta via #raw("omp for")])

#v(0.6em)
#pause

Mas paralelizar sem sincronizacao nao basta: com $T$ threads e $E$ episodios,
cada thread aprende como se tivesse apenas $E slash T$ episodios.
#heading(level: 2, outlined: false, numbering: none)[Sincronizacao Periodica] #label("org1986adf")
Cada thread mantem uma #text(weight: "bold", [Q\u{2d}Table local]), treina um lote de #raw("sync_interval") episodios
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
#heading(level: 2, outlined: false, numbering: none)[Heuristicas de sync\u{5f}interval] #label("orgce6cf90")
Duas estrategias automaticas implementadas em #raw("train()"):

#v(0.8em)

#list(list.item[#text(weight: "bold", [sqrt]): #raw("sync_interval = sqrt(N/T)") \u{2d}\u{2d} equilibra overhead de barreira e divergencia local (inspirado em limites de regret para SGD paralelo)])

#v(0.8em)
#pause

#list(list.item[#text(weight: "bold", [statespace]): #raw("sync_interval = L * C") \u{2d}\u{2d} cada thread explora todos os estados antes de fundir])

#v(0.8em)
#pause

Alem disso, testamos valores fixos: #text(weight: "bold", [every10]) (#raw("i=10")) e #text(weight: "bold", [every1000]) (#raw("i=1000")).
#heading(level: 2, outlined: false, numbering: none)[HOGWILD!] #label("org238bdf7")
Todas as threads leem e escrevem na #text(weight: "bold", [mesma Q\u{2d}Table sem nenhum lock]):

#v(0.5em)

#text(size: 0.72em)[
```c
QLearning local_ql = *ql;
local_ql.rand_seed = seed + tid;
/* q_table aponta para a tabela COMPARTILHADA */

#pragma omp for schedule(runtime)
for (episode = 0; episode < N; episode++)
    run_episode(&local_ql, episode);  /* sem lock */
```
]

#v(0.5em)
#pause

Race conditions sao #text(weight: "bold", [intencionais e inofensivas]) porque:

#list(list.item[#text(weight: "bold", [Esparsidade]): cada episodio visita fracao pequena dos $L times C$ estados])#list(list.item[#text(weight: "bold", [Escala]): updates ponderados por $alpha <= 0.1$ \u{2d}\u{2d} sobrescrever um e ruido $O(alpha)$])#list(list.item[#text(weight: "bold", [Simetria]): ambos os valores concorrentes sao gradientes validos])

Ref: Niu et al., #text(weight: "bold", [Hogwild!]), NIPS 2011.
#heading(level: 1, outlined: false, numbering: none)[Metodologia Experimental] #label("org2440e77")
#heading(level: 2, outlined: false, numbering: none)[Fatores do experimento] #label("org1c682aa")
#{
  set text(size: 0.85em)
  table(
    columns: (auto, auto, 1fr),
    stroke: none,
    table.hline(),
    table.header([*Fator*], [*Niveis*], [*Motivacao*]),
    table.hline(),
    [Threads],        [1, 2, 4, 8, 16, 20, 40],        [Do serial ao hyperthreading completo],
    [Tamanho grid],   [10, 50, 100],                    [Espaco de estados pequeno/medio/grande],
    [Modo de sync],   [sqrt, statespace, hogwild, 10, 1000],  [5 estrategias de paralelizacao],
    [Episodios],      [5000, 10000],                    [5k instavel, 10k converge melhor],
    [Obstaculos],     [0%, 50%],                        [Grid livre vs. alta densidade],
    [Schedule OMP],   [static, guided, dynamic],        [Politicas de distribuicao],
    [Chunk],          [0 (auto), 1, 10],                [Granularidade do chunking],
    table.hline(),
  )
}

#v(0.6em)

#text(weight: "bold", [Baseline]): binario sequencial puro (sem runtime OpenMP) para $T=1$.
#heading(level: 2, outlined: false, numbering: none)[Afinidade de threads e medicao de tempo] #label("org40fd541")
#text(weight: "bold", [Afinidade:])

#list(list.item[$T <= 20$: #raw("OMP_PLACES=cores"), #raw("OMP_PROC_BIND=close") \u{2d}\u{2d} 1 thread por nucleo fisico])#list(list.item[$T = 40$: #raw("OMP_PLACES=threads") \u{2d}\u{2d} hyperthreading completo])

#v(1em)

#text(weight: "bold", [Medicao de tempo:])

#list(list.item[Apenas o tempo de #raw("train()") \u{2d}\u{2d} exclui alocacao, init e I\u{2f}O])#list(list.item[Paralelo: #raw("omp_get_wtime()") | Sequencial: #raw("clock_gettime(CLOCK_MONOTONIC)")])#list(list.item[Multiplos blocos de replicacao para estimar variancia do escalonador])
#heading(level: 1, outlined: false, numbering: none)[Resultados] #label("org9a8b895")
#heading(level: 2, outlined: false, numbering: none)[Speedup por modo de sincronizacao] #label("orgb1ab2e8")
#align(center, image("figs/speedup_blocks.svg", width: 200%, height: 10cm, fit: "contain"))
#heading(level: 2, outlined: false, numbering: none)[Speedup: observacoes] #label("org26b7c27")
#list(list.item[#text(weight: "bold", [10x10]): satura em ~4\u{2d}\u{2d}5x a partir de 8 threads; trabalho por thread insuficiente para amortizar overhead])

#v(0.7em)
#pause

#list(list.item[#text(weight: "bold", [100x100]): HOGWILD! atinge ate #text(weight: "bold", [17x com 40 threads]); modos sincronizados ficam abaixo de 10x])

#v(0.7em)
#pause

#list(list.item[Barreiras periodicas serializam parcialmente a execucao nos modos sincronizados])

#v(0.7em)
#pause

#list(list.item[Grid 10x10 com T=2: eficiencia #text(weight: "bold", [super\u{2d}linear]) (efeito de cache com Q\u{2d}Table de 3,2 KB)])
#heading(level: 2, outlined: false, numbering: none)[Eficiencia paralela S(T)\u{2f}T] #label("org133af47")
#align(center, image("figs/efficiency.svg", width: 200%, height: 10cm, fit: "contain"))
#heading(level: 2, outlined: false, numbering: none)[Eficiencia: observacoes] #label("org972a08c")
#list(list.item[HOGWILD! mantem a #text(weight: "bold", [melhor eficiencia]) em todos os grids: ~35\u{2d}\u{2d}45% com 40 threads no 100x100])

#v(0.7em)
#pause

#list(list.item[#text(weight: "bold", [every10]) (sync a cada 10 eps): #text(weight: "bold", [pior eficiencia]) \u{2d}\u{2d} overhead de barreiras frequentes supera o ganho])

#v(0.7em)
#pause

#list(list.item[every1000 e statespace ficam no intermediario: sincronizam menos e amortizam o custo])

#v(0.7em)
#pause

#list(list.item[Nenhuma configuracao sustenta 50% de eficiencia alem de 8 threads no grid 10x10])
#heading(level: 2, outlined: false, numbering: none)[Analise de Cache (10x10)] #label("org222b43c")
Q\u{2d}Table do 10x10 ocupa #text(weight: "bold", [3,2 KB]) (100 estados x 4 acoes x 8 bytes) \u{2d}\u{2d} cabe inteira em L1.

#v(0.6em)

Taxa de LLC\u{2d}miss medida com #raw("perf stat"):

#v(0.4em)
#{
  set text(size: 0.82em)
  table(
    columns: (auto, auto, auto, auto),
    stroke: none,
    table.hline(),
    table.header([*Threads*], [*sqrt*], [*statespace*], [*hogwild*]),
    table.hline(),
    [2],  [0,80%], [1,11%], [0,44%],
    [4],  [0,70%], [0,79%], [0,18%],
    [8],  [0,17%], [0,18%], [0,11%],
    table.hline(),
    [16], [30,9%], [23,1%], [36,7%],
    [20], [29,1%], [22,9%], [36,4%],
    [40], [40,7%], [28,5%], [35,9%],
    table.hline(),
  )
}

#v(0.4em)

Ponto de ruptura em $T in [8, 16]$ confirma: gargalo e #text(weight: "bold", [trafego de cache]), nao falta de trabalho.
#heading(level: 2, outlined: false, numbering: none)[Analise de Cache: interpretacao] #label("orgdb457c1")
#text(weight: "bold", [Para $T <= 8$]): hogwild tem LLC\u{2d}miss #text(weight: "bold", [menor]) \u{2d}\u{2d} Q\u{2d}Table unica de 3,2 KB fica
estavel em L1\u{2f}L2. Modos federados tem copias separadas por thread, aumentando o footprint.

#v(0.8em)
#pause

#text(weight: "bold", [Para $T >= 16$]): hogwild passa a ter #text(weight: "bold", [mais]) LLC\u{2d}miss (36,7%) \u{2d}\u{2d} cada escrita invalida
a linha de cache nos demais nucleos (protocolo MESI). Statespace aguenta melhor (23,1%)
pois Q\u{2d}Tables locais ficam isoladas por mais tempo entre syncs.

#v(0.8em)
#pause

Confirmado pelo VTune: #text(weight: "bold", [Memory Bound alto]) e #text(weight: "bold", [IPC baixo]) no 10x10 com muitas threads.
#heading(level: 2, outlined: false, numbering: none)[VTune: IPC e Memory Bound] #label("org98c1031")
#align(center, image("figs/vtune.svg", width: 200%, height: 11cm, fit: "contain"))
#heading(level: 2, outlined: false, numbering: none)[Convergencia] #label("org44ef33f")
#align(center, image("figs/conv_obstacles.svg", width: 200%, height: 11cm, fit: "contain"))
#heading(level: 2, outlined: false, numbering: none)[Convergencia: causa da degradacao] #label("org156d496")
Com $T$ threads e $E$ episodios fixos, cada thread treina com $E slash T$ episodios.
Quando $E slash T < E_"min"$ o agente nao aprende.

#v(0.8em)
#pause

#list(list.item[#text(weight: "bold", [0% de obstaculos]): todos os modos convergem para poucos threads; degrada com o aumento])#list(list.item[#text(weight: "bold", [50% de obstaculos]): nenhuma configuracao paralela convergiu para grids grandes])

#v(0.8em)
#pause

Mesmo HOGWILD! (que acumula updates de #text(weight: "bold", [todas]) as threads) falha com 50% de obstaculos:
race conditions corrompem o aprendizado em ambientes densos.
#heading(level: 2, outlined: false, numbering: none)[Schedule, chunk e obstaculos] #label("org5992ef3")
#grid(
  columns: (1fr, 1fr),
  gutter: 0.8em,
  align(center, image("figs/sched_impact.svg", width: 100%, height: 9cm, fit: "contain")),
  align(center, image("figs/obs_speedup.svg",  width: 100%, height: 9cm, fit: "contain")),
)
#heading(level: 2, outlined: false, numbering: none)[Schedule e chunk: nao sao fatores dominantes] #label("orgbdd7bc5")
#list(list.item[Os tres schedules produzem speedup #text(weight: "bold", [praticamente identico]): duracoes de episodio
homogeneas, carga equilibrada independente da politica])

#v(0.7em)

#list(list.item[Diferenca geral permanece #text(weight: "bold", [\u{3c}2% no speedup medio])])

#v(0.7em)

#list(list.item[Chunk=1: leve degradacao em grids pequenos (overhead por episodio);
chunk=10 e auto se comportam de forma equivalente])

#v(0.7em)

#list(list.item[Obstaculos: comparar speedup entre 0% e 50% e #text(weight: "bold", [injusto]) \u{2d}\u{2d} ganho aparente com 50%
reflete apenas que ha mais trabalho total a dividir])
#heading(level: 1, outlined: false, numbering: none)[Conclusao] #label("orgd7c8a39")
#heading(level: 2, outlined: false, numbering: none)[Conclusao] #label("orgbfea466")
#list(list.item[#text(weight: "bold", [Gargalo principal]): divisao de episodios. Com $T$ threads e $E$ fixos,
cada thread treina com $E slash T$ \u{2d}\u{2d} abaixo do limiar, agente nao converge])

#v(0.7em)

#list(list.item[#text(weight: "bold", [Grid 10x10]): gargalo e coerencia de cache \u{2d}\u{2d} LLC\u{2d}miss explode
de \u{3c}1,2% para 23\u{2d}\u{2d}40% entre $T=8$ e $T=16$])

#v(0.7em)

#list(list.item[#text(weight: "bold", [HOGWILD!]) obteve maior speedup bruto (ate #text(weight: "bold", [17x]) com 40 threads em 100x100),
mas nao converge com 50% de obstaculos])

#v(0.7em)

#list(list.item[Modos sincronizados com intervalos longos (#text(weight: "bold", [every1000]), #text(weight: "bold", [statespace])) preservam
melhor o comportamento de cache entre syncs, mas convergem menos])

#v(0.7em)

#list(list.item[Schedule e chunk: #text(weight: "bold", [nao sao fatores dominantes]) (\u{3c}2% de impacto no speedup)])
#heading(level: 2, outlined: false, numbering: none)[] #label("orgc7b78c7")
#text(size: 2.8em, weight: "bold", "Perguntas?")
