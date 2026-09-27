# Categorizador Semântico de Produtos (v2 - MiniLM)

Microsserviço REST em FastAPI para classificação automática de nomes de produtos de faturas/notas fiscais ruidosos e com abreviações (ex: `SAB L MONANGE DETOX`), utilizando Similaridade de Cosseno com embeddings vetoriais profundos gerados pelo modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

### Principais Vantagens da v2:
- **Ultra leve:** Consome apenas ~400 MB a 600 MB de RAM (roda em máquinas de **1 GB de RAM / $6/mês** no DigitalOcean).
- **Sem necessidade de arquivos de 7 GB:** Não precisa mais baixar ou manter `cc.pt.300.bin` (o download do modelo é automático e tem menos de 470 MB).
- **Maior acurácia semântica:** Arquitetura Transformer multilíngue capaz de capturar contexto de palavras compostas e abreviações de supermercado.

---

## Requisitos
- Python 3.9+ ou Docker
- 1 vCPU e 1 GB de RAM (mínimo)

---

## Execução com Docker (Recomendado para Produção / DigitalOcean)

```bash
docker compose up -d --build
```
A API estará acessível em `http://localhost:8000`.

---

## Execução Local com Python

1. Crie e ative um ambiente virtual:
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Inicie o servidor:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Endpoints

### 1. `GET /health`
Verifica se o modelo foi carregado e se a API está pronta para receber requisições.

### 2. `POST /categorizar`
Recebe o nome do produto e retorna a categoria correspondente e o grau de confiança (similaridade de cosseno).

**Payload:**
```json
{
  "nome": "SAB L MONANGE DETOX"
}
```

**Resposta:**
```json
{
  "produto": "SAB L MONANGE DETOX",
  "categoria": "HIGIENE_E_BELEZA",
  "confianca": 0.7351
}
```
