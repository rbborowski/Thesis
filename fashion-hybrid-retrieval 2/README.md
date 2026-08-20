# Hybrid Retrieval for Fashion E-Commerce Search

Código do TCC **"Hybrid Retrieval for Fashion E-Commerce Search: Combining Semantic
Similarity and LLM-Based Query Understanding"** (INF/UFRGS).

> **Custo zero.** Nenhuma parte deste projeto usa serviço pago, assinatura ou API
> cobrada. O modelo de linguagem roda localmente na sua máquina (Ollama ou Hugging
> Face), os embeddings também, e o dataset é público e gratuito.

Compara dois sistemas de busca sobre o mesmo catálogo e o mesmo espaço de embeddings:

| Sistema | O que faz |
|---|---|
| **baseline** | Ranqueia todo o catálogo por similaridade de cosseno entre o embedding da consulta e o embedding da descrição do produto. |
| **hybrid** | Usa um LLM para separar a consulta em *hard filters* (restrições categóricas, validadas contra o vocabulário do catálogo) e *soft intent* (linguagem estilística); filtra os candidatos pelos metadados e só então ranqueia. |

Os dois só diferem no que acontece **antes** do ranqueamento — é isso que torna
qualquer diferença atribuível ao estágio de compreensão de consulta.

---

## 1. Início rápido (offline, sem dataset, sem modelo, sem conta)

```bash
git clone <seu-repo> && cd fashion-hybrid-retrieval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make test    # 21 testes
make demo    # pipeline inteiro com catálogo sintético
```

`make demo` gera um catálogo falso com o schema do H&M, usa um embedder por hashing
e um LLM mock baseado em regex. Serve **só** para verificar o encanamento — os
números daí não valem para o texto.

## 2. Configuração real (tudo gratuito)

### 2.1 Modelo de linguagem: Ollama (recomendado)

[Ollama](https://ollama.com) é software livre que roda modelos de pesos abertos na
sua máquina. Sem conta, sem chave, sem cobrança.

```bash
# instale o Ollama pelo site, depois:
ollama pull qwen2.5:3b-instruct     # ~2 GB, roda em CPU
make ollama-setup                    # atalho equivalente
```

Modelos sugeridos, em ordem crescente de exigência de hardware:

| Modelo | RAM aprox. | Comentário |
|---|---|---|
| `qwen2.5:3b-instruct` | 4 GB | roda em CPU; bom ponto de partida |
| `llama3.1:8b` | 8–10 GB | melhor extração, mais lento em CPU |
| `qwen2.5:7b-instruct` | 8 GB | bom equilíbrio se houver GPU |

Alternativas, se não puder instalar o Ollama:

- `--llm transformers` — carrega o modelo direto pelo Hugging Face
  (`pip install -e ".[local-llm]"`). Mais pesado em RAM, mas não precisa de servidor.
- `--llm openai-compatible` — aponta para qualquer servidor local que fale o
  protocolo do OpenAI (`llama-server` do llama.cpp, vLLM, LM Studio). Use apenas
  com servidor local; não aponte para fornecedor pago.

### 2.2 Embeddings

`sentence-transformers` baixa o modelo uma vez e roda local, de graça:

```bash
pip install -e ".[full,dev]"
```

O padrão é `all-MiniLM-L6-v2` (~90 MB, roda bem em CPU).

### 2.3 Dataset

```bash
pip install kaggle                                       # cliente gratuito
python scripts/download_catalog.py                       # baixa o arquivo oficial
python scripts/download_catalog.py --source huggingface  # espelho, só para destravar
```

Setup do Kaggle, feito uma vez:

1. Conta gratuita em <https://www.kaggle.com>
2. Aceitar os termos da competição **no navegador** — é o único passo que não dá
   para automatizar
3. Em <https://www.kaggle.com/settings/account>, "Create New Token" → salvar o
   `kaggle.json` em `~/.kaggle/` (e `chmod 600 ~/.kaggle/kaggle.json`)

Feito isso, `python scripts/download_catalog.py` baixa **apenas** o `articles.csv`
(poucos MB) e valida o schema. O arquivo de imagens da competição (~25 GB) não é
necessário: o ranqueamento aqui é texto-texto. Se der erro 403, quase sempre
significa que os termos ainda não foram aceitos naquela conta.

**Use a rota do Kaggle para os resultados que vão no texto.** A conta é gratuita e
nada é cobrado; o único custo é aceitar os termos uma vez no navegador. Em troca você
fica com o arquivo oficial: 105.542 artigos, auditável e citável.

Os espelhos no Hugging Face não exigem conta, mas **não são cópias cruas** — são
versões processadas, das quais o uploader já removeu linhas com campos faltantes e às
quais adicionou embeddings pré-computados. Isso significa que o tamanho do catálogo
não é o oficial e os critérios de exclusão não são seus para documentar, o que
enfraquece a análise do espaço de rótulos (Seção 4.2) e a procedência de qualquer
número reportado. O script descarta as colunas de embedding pré-computado, porque o
experimento precisa montar a própria representação textual (Seção 4.3).

Não misture as duas fontes entre experimentos: escolha uma e reporte qual.

### 2.4 Rodando

```bash
python -m fashion_retrieval prepare
python -m fashion_retrieval label-space
python -m fashion_retrieval embed       --embeddings sentence-transformers
python -m fashion_retrieval gen-queries --llm ollama -n 500
python -m fashion_retrieval evaluate    --embeddings sentence-transformers --llm ollama
python -m fashion_retrieval search "a black midi dress for a summer wedding" \
    --embeddings sentence-transformers --llm ollama
```

Comece com `--max-articles 5000` e `-n 50` para validar o fluxo antes de rodar no
catálogo completo (~105 mil artigos). Com modelo local em CPU, gerar 500 consultas
leva de dezenas de minutos a algumas horas: rode de madrugada ou reduza o `n`.

---

## 3. Comandos

| Comando | Seção do TCC | O que faz |
|---|---|---|
| `prepare` | 4.2 | Lê `articles.csv`, descarta artigos sem descrição usável, monta `product_text`. |
| `label-space` | 4.2 | Conta valores distintos e cobertura por campo, decide quais servem como *hard filter*, escreve o vocabulário fechado e `results/label_space_report.md`. |
| `embed` | 2.2 | Codifica `product_text` e salva a matriz de embeddings. |
| `gen-queries` | 4.5 | Gera consultas sintéticas a partir de nome + descrição, **sem** mostrar os metadados ao gerador. |
| `evaluate` | 4.5 | Roda os dois sistemas e reporta Recall@k, MRR, candidatos, taxa de relaxamento e latência por estágio. |
| `search "..."` | — | Uma consulta pelos dois sistemas, lado a lado. |
| `scripts/run_case_study.py` | 4.5 | Estudo de caso qualitativo com *constraint violation rate*. |
| `scripts/download_catalog.py` | 4.2 | Baixa o catálogo de fonte gratuita. |
| `scripts/make_sample_catalog.py` | — | Catálogo sintético para testes. |

## 4. Estrutura

```
src/fashion_retrieval/
  config.py            Configuração central (caminhos, hiperparâmetros, schema)
  data.py              Carregamento e limpeza do catálogo
  label_space.py       Análise do espaço de rótulos e vocabulário de filtros
  embeddings.py        Backends de embedding + similaridade de cosseno
  llm.py               Clientes LLM (ollama | transformers | openai-compatible | mock)
  query_parser.py      Extração de hard filters + soft intent, com validação
  retrieval.py         BaselineRetriever e HybridRetriever (com relaxamento)
  synthetic_queries.py Geração de consultas sintéticas com os três controles
  evaluation.py        Recall@k, violation rate, acurácia de extração, relatórios
  cli.py               Interface de linha de comando
scripts/               download_catalog, make_sample_catalog, run_case_study
tests/test_pipeline.py 21 testes, todos offline
```

---

## 5. Decisões de projeto que vieram direto do texto

Cada uma existe para tapar um buraco metodológico específico:

**O baseline enxerga os mesmos metadados que o híbrido filtra.**
`build_product_text()` concatena os campos estruturados à descrição. Sem isso, o
baseline seria privado de informação que o híbrido tem, e a diferença medida
refletiria em parte essa assimetria, não a arquitetura.

**Valores fora do vocabulário são descartados, não repassados.**
`QueryParser._validate()` só aceita pares campo/valor existentes no catálogo — o que
importa ainda mais com modelo pequeno, que alucina valores com mais frequência.

**Filtro errado não é fatal: existe relaxamento.**
Abaixo de `min_candidates`, `apply_filters()` descarta filtros um a um em ordem
crescente de confiança e registra o que descartou. A taxa de relaxamento é reportada.

**O gerador de consultas sintéticas não vê os metadados nem o schema de filtros.**
Caso contrário as consultas sairiam no vocabulário exato dos filtros e o híbrido
venceria por construção do experimento.

**A plausibilidade das consultas sintéticas é medida, não assumida.**
`export_plausibility_sample()` gera CSV com coluna `plausible` para anotação manual.

**Falha de parsing é resultado, não exceção.**
`extract_json()` é tolerante e, quando falha, marca `parse_failed`; a taxa entra no
relatório. Com modelo local pequeno isso não é detalhe: confiabilidade de saída
estruturada é parte do custo da escolha, e vale discutir na Seção 4.7.

**A eficiência é medida, mas não é alegada como benefício.**
Latência por estágio e tamanho do conjunto de candidatos são caracterização
descritiva (Seção 2.4 do texto).

## 6. Métricas

- **Recall@k / MRR** (H2) — o produto que originou a consulta aparece no top-k?
- **Constraint violation rate** (H1) — fração dos top-k cujos metadados contradizem
  uma restrição declarada. Não precisa de julgamento de relevância.
- **Precisão/revocação de extração por campo** (H3) — falso positivo e falso negativo
  contados separadamente: filtro errado remove o produto certo, filtro faltante só
  deixa de ajudar.
- **Descritivas** — candidatos, relaxamento, falha de parsing, latência por estágio.

## 7. Estudo de caso qualitativo

Copie `data/case_study.example.csv` para `data/case_study.csv`, edite, e rode:

```bash
python scripts/run_case_study.py --llm ollama --embeddings sentence-transformers
```

Deixe a célula vazia quando a consulta não declara aquela restrição. Inclua casos que
estressam a distinção estudada: tipos com vizinhos próximos (dress/skirt) e consultas
que declaram uma cor enquanto descrevem estilo típico de outra.

---

## 8. Notas

- `min_candidates` (padrão 20) dispara relaxamento com frequência em catálogos
  pequenos, já que "Black + Dress" pode ter menos de 20 itens. Reduza em amostras.
- O backend `hashing` é determinístico e não semântico: existe para testes e CI.
- `data/`, `results/` e `*.npy` estão no `.gitignore` — o dataset não é redistribuído.
- Se um modelo local pequeno produzir JSON inválido com frequência, tente um modelo
  maior, ou baixe `llm_temperature` para 0, ou reduza o vocabulário no prompt via
  `min_value_frequency`. Reporte a taxa de falha em vez de escondê-la.

## Licença

Código sob licença MIT. O dataset da H&M é regido pelos termos da competição no
Kaggle e não é redistribuído aqui.
