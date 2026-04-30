# =============================================================================
# MAKEFILE - Q-LEARNING SEQUENCIAL
# =============================================================================
# 
# Comandos disponíveis:
#   make all        - Compila todos os programas
#   make cli        - Compila versão com linha de comando (recomendado!)
#   make sequential - Compila versão antiga (parâmetros fixos)
#   make test       - Compila e executa os testes
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
CLI = qlearning
SEQUENTIAL = qlearning_sequential
TEST = test_qlearning
TEST_CLI = test_qlearning_cli

# Regra padrão: compila tudo
all: $(CLI) $(SEQUENTIAL) $(TEST) $(TEST_CLI)

# Compila versão CLI (recomendado!)
$(CLI): qlearning_cli.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

cli: $(CLI)

# Compila o programa sequencial (versão antiga)
$(SEQUENTIAL): qlearning_sequential.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

sequential: $(SEQUENTIAL)

# Compila os testes (versão original)
$(TEST): test_qlearning.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Compila os testes (versão CLI dinâmica)
$(TEST_CLI): test_qlearning_cli.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

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

# Compila e executa os testes (versão original)
test: $(TEST)
	./$(TEST)

# Compila e executa os testes CLI
test-cli: $(TEST_CLI)
	./$(TEST_CLI)

# Executa todos os testes
test-all: $(TEST) $(TEST_CLI)
	@echo "===== Executando testes da versão original ====="
	./$(TEST)
	@echo ""
	@echo "===== Executando testes da versão CLI ====="
	./$(TEST_CLI)

# Limpa arquivos compilados
clean:
	rm -f $(CLI) $(SEQUENTIAL) $(TEST) $(TEST_CLI)

# Marca alvos que não são arquivos
.PHONY: all cli sequential run run-easy run-hard run-extreme demo help test test-cli test-all clean
