import os
import re
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuração de Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("categorizador-produtos")

# ---------------------------------------------------------------------------
# Configurações do Ambiente e Constantes
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Padrões determinísticos de alta precisão para itens comuns de supermercado / faturas
REGRAS_REGEX: Dict[str, str] = {
    "ACOUGUE_E_PEIXARIA": (
        r"\b(picanha|alcatra|contra|maminha|costela|frango|coxa|sobrecoxa|peito de frango|"
        r"carne|bov|suin|peixe|salmao|tilapia|camarao|linguica|salsicha|bife|acougue|bacon|"
        r"pernil|mignon|patinho|acem|cupim|bovino|suino|pescado|bacalhau)\b"
    ),
    "LATICINIOS_E_OVOS": (
        r"\b(leite|queijo|mussarela|mucarela|parmesao|iogurte|requeijao|manteiga|margarina|"
        r"ovo|ovos|nata|creme de leite|coalhada|ricota|gorgonzola|provolone)\b"
    ),
    "PADARIA_E_CONFEITARIA": (
        r"\b(pao|biscoito|bolacha|bolo|torta|croissant|baguete|salgado|coxinha|empada|"
        r"pastel|torrada|panetone|confeitaria|padaria)\b"
    ),
    "MERCEARIA_SECA": (
        r"\b(arroz|feijao|macarrao|massa|espaguete|oleo|azeite|farinha|acucar|cafe|sal|"
        r"molho|extrato|enlatado|milho|ervilha|sardinha|atum|lentilha|grao de bico)\b"
    ),
    "CONGELADOS": (
        r"\b(congelad|cong\b|sorvete|pizza|lasanha|nugget|hamburguer|steak|batata cong|"
        r"polpa de fruta|acai|gelo)\b"
    ),
    "BEBIDAS": (
        r"\b(refrigerante|refr\b|coca|coca-cola|pepsi|guarana|fanta|suco|cerveja|chopp|"
        r"vinho|vodka|whisky|gin|energetico|red bull|monster|agua|tonica|cha|ice)\b"
    ),
    "LIMPEZA": (
        r"\b(detergente|desinfetante|sabao em po|sabao barra|amaciante|agua sanitaria|"
        r"alvejante|cloro|esponja|bombril|ype|veja|limpador|desengordurante|multiuso|"
        r"vassoura|rodo|saco lixo|lustra|lixivia)\b"
    ),
    "HIGIENE_E_BELEZA": (
        r"\b(shampoo|shamp\b|condicionador|cond\b|sabonete|sab l\b|sab emb\b|creme dental|"
        r"pasta dente|escova dente|desodorante|desod\b|absorvente|cotonete|hidratante|"
        r"perfume|fio dental|papel higienico|fralda|protetor solar|esmalt)\b"
    ),
    "PET_SHOP": (
        r"\b(racao|petisco|pet\b|pedigree|whiskas|golden|premier|areia gato|coleira|"
        r"cachorro|caes|canino|felino)\b"
    ),
    "UTILIDADES_DOMESTICAS": (
        r"\b(lampada|pilha|panela|copo|prato|talher|guardanapo|papel toalha|bateria|"
        r"utensilio|bazar|filtro cafe)\b"
    ),
    "HORTIFRUTI_FRUTAS": (
        r"\b(maca|banana|laranja|uva|morango|abacaxi|melancia|melao|mamao|limao|pera|"
        r"manga|maracuja|kiwi|pessego|goiaba|tangerina|mexerica)\b"
    ),
    "HORTIFRUTI_VERDURAS_E_LEGUMES": (
        r"\b(alface|tomate|cebola|batata|cenoura|alho|pimentao|chuchu|abobrinha|couve|"
        r"brocolis|espinafre|repolho|mandioca|aipim|rucula|cheiro verde|hortalica)\b"
    )
}

CATEGORIAS_DESCRICAO: Dict[str, str] = {
    "ACOUGUE_E_PEIXARIA": "carnes bovinas, aves, frango, suino, peixes e frutos do mar de acougue",
    "LATICINIOS_E_OVOS": "laticinios, leite, queijos, iogurtes, manteiga e ovos",
    "PADARIA_E_CONFEITARIA": "padaria e confeitaria, paes, bolos, tortas, biscoitos e salgados",
    "MERCEARIA_SECA": "mercearia seca, graos, arroz, feijao, massas, oleo de cozinha, cafe e acucar",
    "CONGELADOS": "alimentos congelados, sorvetes, refeicoes prontas, pizzas e lasanhas congeladas",
    "BEBIDAS": "bebidas, refrigerantes, sucos, cervejas, vinhos, destilados e agua mineral",
    "LIMPEZA": "produtos de limpeza domestica, detergentes, desinfetantes, sabao e amaciante",
    "HIGIENE_E_BELEZA": "higiene pessoal e cosméticos, sabonetes, shampoos, desodorantes e cremes",
    "PET_SHOP": "produtos para animais e pet shop, racao e petiscos para caes e gatos",
    "UTILIDADES_DOMESTICAS": "utilidades domesticas e bazar, utensilios de cozinha, copos, pratos e lampadas",
    "HORTIFRUTI_FRUTAS": "frutas frescas de hortifruti",
    "HORTIFRUTI_VERDURAS_E_LEGUMES": "verduras, legumes e hortalicas frescas de hortifruti"
}

# ---------------------------------------------------------------------------
# Serviço de Classificação Híbrido (Regras + IA Semântica)
# ---------------------------------------------------------------------------
class ModelService:
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.category_names: List[str] = list(CATEGORIAS_DESCRICAO.keys())
        self.category_vectors: Dict[str, np.ndarray] = {}

    def initialize(self, model_name: str):
        logger.info(f"Carregando modelo Sentence-Transformer '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        logger.info("Modelo Sentence-Transformer carregado com sucesso.")

        logger.info("Indexando vetores semânticos das categorias...")
        for cat, desc in CATEGORIAS_DESCRICAO.items():
            vec = self.model.encode(desc, normalize_embeddings=True)
            self.category_vectors[cat] = np.array(vec, dtype=np.float32)
        logger.info(f"{len(self.category_names)} categorias semânticas indexadas.")

    def classify(self, text: str) -> Dict[str, any]:
        cleaned = text.strip()
        if not cleaned:
            return {"categoria": "DESCONHECIDO", "confianca": 0.0}

        cleaned_lower = cleaned.lower()

        # 1. Regra Determinística por Expressões Regulares (Rápido e 100% Preciso)
        for cat, pattern in REGRAS_REGEX.items():
            if re.search(pattern, cleaned_lower):
                return {
                    "categoria": cat,
                    "confianca": 0.95
                }

        # 2. Fallback Semântico com IA Transformer (para itens não cobertos pelas regras)
        product_vec = self.model.encode(f"produto de mercado: {cleaned_lower}", normalize_embeddings=True)
        best_cat = None
        best_score = -1.0

        for cat, cat_vec in self.category_vectors.items():
            score = float(np.dot(cat_vec, product_vec))
            if score > best_score:
                best_score = score
                best_cat = cat

        return {
            "categoria": best_cat or "MERCEARIA_SECA",
            "confianca": round(max(best_score, 0.0), 4)
        }

model_service = ModelService()

# ---------------------------------------------------------------------------
# Ciclo de Vida do FastAPI (Lifespan)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    model_service.initialize(MODEL_NAME)
    yield
    # Shutdown
    del model_service.model
    del model_service.category_vectors

# ---------------------------------------------------------------------------
# Aplicação FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Microsserviço de Categorização Semântica de Produtos",
    description="Classifica nomes de produtos extraídos de faturas/notas fiscais usando abordagem híbrida (Regras Regex + Sentence-Transformers).",
    version="2.1.0",
    lifespan=lifespan
)

# ---------------------------------------------------------------------------
# Schemas Pydantic
# ---------------------------------------------------------------------------
class ProdutoInput(BaseModel):
    nome: str = Field(..., min_length=1, example="SAB L MONANGE DETOX")

class CategorizacaoOutput(BaseModel):
    produto: str
    categoria: str
    confianca: float

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", status_code=status.HTTP_200_OK, tags=["Sistema"])
def health_check():
    is_ready = model_service.model is not None
    return {
        "status": "ready" if is_ready else "loading",
        "modelo": MODEL_NAME,
        "categorias_disponiveis": len(model_service.category_names)
    }

@app.post(
    "/categorizar",
    response_model=CategorizacaoOutput,
    status_code=status.HTTP_200_OK,
    tags=["Classificação"]
)
def categorizar_produto(payload: ProdutoInput):
    if not payload.nome.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="O nome do produto não pode ser vazio."
        )

    resultado = model_service.classify(payload.nome)

    return CategorizacaoOutput(
        produto=payload.nome,
        categoria=resultado["categoria"],
        confianca=resultado["confianca"]
    )
