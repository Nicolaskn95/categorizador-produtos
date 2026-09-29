# Categorizador Inteligente de Produtos (Linear SVM + TF-IDF)

Microsserviço REST de alta performance desenvolvido em **FastAPI** para classificação automática de itens de faturas/notas fiscais (NFC-e e CF-e). 

O serviço foi projetado especificamente para lidar com abreviações fiscais severas da SEFAZ, medidas de embalagem e ruídos de impressão (ex: `1 PE - BROCOLIS JAPON/NINJA`, `0.76 KG - ABOB ITALIA`, `GUARD FANI 23X22`, `1.006 KG - MUSCULO`, `1 PC - MAC DIVELLA CAPEL 11`).

---

## 🧠 Arquitetura do Modelo

O microsserviço utiliza uma abordagem híbrida de classificação em 3 estágios:

1. **Pré-processamento e Desabreviação Fiscal:**
   - Remoção de prefixos fiscais de pesagem e embalagens (`MA`, `KG`, `PE`, `PO`, `FR`, `PC`, `BJ`, etc.).
   - Remoção de dimensões soltas (`23X22`, `500ML`, `2L`).
   - Normalização léxica de termos truncados de supermercado (`MAC` $\rightarrow$ `macarrao`, `GUARD` $\rightarrow$ `guardanapo`, `ABOB` $\rightarrow$ `abobrinha`, `REQ` $\rightarrow$ `requeijao`, `LAV R` $\rightarrow$ `lava roupas`, `SARD` $\rightarrow$ `sardinha`, `MUSCULO` $\rightarrow$ `carne musculo`).

2. **Regras Determinísticas (Expressões Regulares):**
   - Validação imediata de palavras-chave canônicas e inequívocas com nível de confiança calibrado em `0.98`.

3. **Classificador de Aprendizado de Máquina (Linear SVM):**
   - **Vetorização Híbrida (`FeatureUnion`):** Combina TF-IDF de palavras (`ngram_range=(1, 2)`) e TF-IDF de subpalavras/caracteres (`analyzer='char_wb'`, `ngram_range=(3, 5)`), permitindo que fragmentos de palavras correspondam perfeitamente mesmo com truncamento.
   - **Modelo:** `LinearSVC` com balanceamento de classes (`class_weight='balanced'`) calibrado probabilisticamente via `CalibratedClassifierCV` para retornar probabilidades e escores de confiança reais.

---

## 🏷️ Categorias Suportadas

O classificador distribui os itens entre as 12 categorias canônicas da plataforma:

1. `ACOUGUE_E_PEIXARIA`
2. `HORTIFRUTI_FRUTAS`
3. `HORTIFRUTI_VERDURAS_E_LEGUMES`
4. `LATICINIOS_E_OVOS`
5. `PADARIA_E_CONFEITARIA`
6. `MERCEARIA_SECA`
7. `CONGELADOS`
8. `BEBIDAS`
9. `LIMPEZA`
10. `HIGIENE_E_BELEZA`
11. `PET_SHOP`
12. `UTILIDADES_DOMESTICAS`

---

## 🚀 Execução com Docker (Recomendado)

O container é extremamente leve (~180 MB) e não necessita de PyTorch ou placas de vídeo dedicadas:

```bash
docker compose up -d --build
```

A API estará acessível em `http://localhost:8000`.

Para visualizar os logs:
```bash
docker compose logs -f
```

---

## 💻 Execução Local com Python

1. Crie e ative um ambiente virtual:
```bash
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate
```

2. Instale as dependências leves:
```bash
pip install -r requirements.txt
```

3. Inicie o servidor FastAPI:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 Endpoints da API

### 1. `GET /health`
Verifica a integridade do serviço e o status do modelo Linear SVM.

**Resposta de Exemplo:**
```json
{
  "status": "ready",
  "modelo": "LinearSVM + TF-IDF (char_wb + word n-grams)",
  "categorias_disponiveis": 12
}
```

---

### 2. `POST /categorizar`
Recebe o nome bruto do produto como extraído da nota fiscal e retorna a categoria correspondente e a confiança estatística.

**Payload:**
```json
{
  "nome": "1 PE - BROCOLIS JAPON/NINJA"
}
```

**Resposta de Exemplo:**
```json
{
  "produto": "1 PE - BROCOLIS JAPON/NINJA",
  "categoria": "HORTIFRUTI_VERDURAS_E_LEGUMES",
  "confianca": 0.98
}
```

**Outro Exemplo (Desabreviação e Limpeza):**
```json
// Requisição:
{ "nome": "1 PC - GUARD FANI 23X22" }

// Resposta:
{
  "produto": "1 PC - GUARD FANI 23X22",
  "categoria": "LIMPEZA",
  "confianca": 0.98
}
```
