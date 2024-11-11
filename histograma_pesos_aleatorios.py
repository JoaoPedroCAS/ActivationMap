import matplotlib.pyplot as plt

# Função para ler o arquivo de métricas e extrair as acurácias
def ler_acuracias(arquivo):
    acuracias = []
    
    with open(arquivo, 'r') as f:
        # Pular o cabeçalho
        next(f)
        
        for linha in f:
            # Dividir a linha por vírgula e pegar a acurácia (segunda coluna)
            dados = linha.strip().split(', ')
            acuracia = float(dados[1])  # A acurácia está na segunda posição
            acuracias.append(acuracia)
    
    return acuracias

# Ler acurácias do arquivo
arquivo = '/home/joao.p.c.a.sa/PreProjeto/Code/resultados_pesos_aleatorios/metricas_resultados.txt'
acuracias = ler_acuracias(arquivo)

# Criar o histograma das acurácias
plt.figure(figsize=(10, 6))
plt.hist(acuracias, bins=20, edgecolor='black', alpha=0.7)
plt.title('Histograma das Acurácias Obtidas')
plt.xlabel('Acurácia')
plt.ylabel('Frequência')
plt.grid(True)
plt.savefig("/home/joao.p.c.a.sa/PreProjeto/Code/resultados_pesos_aleatorios/metricas_resultados.png")
