# =============================================================================
# MAKEFILE - Q-LEARNING
# =============================================================================
#
# Comandos disponíveis:
#   make all        - Compila todos os programas
#   make cli        - Compila versão com linha de comando (recomendado!)
#   make run        - Executa versão CLI no modo normal
#   make run-easy   - Executa versão CLI no modo fácil
#   make run-hard   - Executa versão CLI no modo difícil
#   make run-extreme- Executa versão CLI no modo extremo
#   make demo       - Demonstração com vários exemplos
#   make help       - Mostra ajuda do programa CLI
#   make clean      - Remove arquivos compilados
#
# =============================================================================

CC = gcc
CFLAGS = -Wall -Wextra -O2 -fopenmp
LDFLAGS = -lm

# Arquivos executáveis
CLI = bin/qlearning
PARALLEL = bin/qlearning_parallel
TEST_CLI = bin/test_qlearning_cli

# Regra padrão: compila tudo
all: $(CLI) $(PARALLEL) $(TEST) $(TEST_CLI)

# Create bin directory if it doesn't exist
$(CLI) $(PARALLEL) $(TEST_CLI): | bin

bin:
	mkdir -p bin

# Compila versão CLI (recomendado!)
$(CLI): src/qlearning_cli.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

cli: $(CLI)

# Compila os testes (versão CLI dinâmica)
$(TEST_CLI): src/test_qlearning_cli.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)


$(PARALLEL): src/parallel_qlearning_cli.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

parallel: $(PARALLEL)


# Executa versão CLI
run: $(CLI)
	./$(CLI) --mode normal

run-easy: $(CLI)
	./$(CLI) --mode easy

run-hard: $(CLI)
	./$(CLI) --mode hard

run-extreme: $(CLI)
	./$(CLI) --mode extreme --verbose

# Demonstração com exemplos
demo: $(CLI)
	@echo "=========================================="
	@echo "DEMO 1: Modo Easy (Grid 3x3)"
	@echo "=========================================="
	./$(CLI) --mode easy
	@echo ""
	@echo "=========================================="
	@echo "DEMO 2: Grid 5x5 com 3 obstáculos"
	@echo "=========================================="
	./$(CLI) --gridx 5 --gridy 5 --obstacles 3 --seed 123
	@echo ""
	@echo "=========================================="
	@echo "DEMO 3: Grid 6x6 com 5 obstáculos (seed diferente)"
	@echo "=========================================="
	./$(CLI) --gridx 6 --gridy 6 --obstacles 5 --seed 999 --episodes 2000

# Mostra ajuda
help: $(CLI)
	./$(CLI) --help

# Compila e executa os testes CLI
test-cli: $(TEST_CLI)
	./$(TEST_CLI)

# Limpa arquivos compilados
clean:
	rm -rf bin

# Marca alvos que não são arquivos
.PHONY: all cli run run-easy run-hard run-extreme demo help test-cli test-all clean
