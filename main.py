# -*- coding: utf-8 -*-
import os
import re
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion

# ---------------------------------------------------------------------------
# Configuração de Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("categorizador-produtos")

# ---------------------------------------------------------------------------
# Dicionário de Normalização e Expansão de Abreviações Fiscais (SEFAZ / NFC-e)
# ---------------------------------------------------------------------------
PADRAO_EMBALAGEM = (
    r"^\s*\d+([.,]\d+)?\s*"
    r"(MA|SH|TP|FR|PC|KG|UN|LT|CX|PT|GL|FD|BJ|PE|PO|BD|DZ|SC|TB|BL|BR|LATA|BARRA|M|G|GR)"
    r"\s*[-–—/]\s*"
)

ABREVIACOES_FISCAIS: Dict[str, str] = {
    r"\bagua sanit\w*\b": "agua sanitaria",
    r"\bag sanit\w*\b": "agua sanitaria",
    r"\bagua san\b": "agua sanitaria",
    r"\bpo gel\b": "gelatina",
    r"\bpo gelat\w*\b": "gelatina",
    r"\bfil pap\b": "filtro de papel",
    r"\bfiltro papel\b": "filtro de papel",
    r"\bfiltro cafe\b": "filtro de papel",
    r"\bnect\b": "suco nectar",
    r"\besp s brite\b": "esponja scotch brite",
    r"\besp\b": "esponja",
    r"\bs brite\b": "scotch brite",
    r"\bqjo\b": "queijo",
    r"\bmussarel\b": "mussarela",
    r"\bmol soja\b": "molho soja",
    r"\bmol\b": "molho",
    r"\bcr leite\b": "creme de leite",
    r"\bleite cond\b": "leite condensado",
    r"\bguar scott\b": "guardanapo scott",
    r"\blav r\b": "lava roupas",
    r"\btixan mac\b": "tixan maciez",
    r"\btixan\b": "lava roupas tixan",
    r"\bguard\b": "guardanapo",
    r"\babob\b": "abobrinha",
    r"\breq\b": "requeijao",
    r"\bsard\b": "sardinha",
    r"\bcer\b": "cereal",
    r"\bmaion\b": "maionese",
    r"\bling c\b": "linguica calabresa",
    r"\bling\b": "linguica",
    r"\bdes\b": "desinfetante",
    r"\bdet l\b": "detergente liquido",
    r"\bdet\b": "detergente",
    r"\brefri\b": "refrigerante",
    r"\brefr\b": "refrigerante",
    r"\bref\b": "refrigerante",
    r"\benerg\b": "energetico",
    r"\bsab l\b": "sabonete liquido",
    r"\bsab\b": "sabonete",
    r"\bchoc\b": "chocolate",
    r"\bch\b": "chocolate",
    r"\bbisc\b": "biscoito",
    r"\bacem\b": "carne acem",
    r"\bmusculo\b": "carne musculo",
    r"\bcost\b": "costela",
    r"\balcat\b": "alcatra",
    r"\bpican\b": "picanha",
    r"\btemp\b": "tempero",
    r"\bfar ma\b": "farinha mandioca",
    r"\bfar\b": "farinha",
    r"\bmac\b": "macarrao",
}

def limpar_nome_fiscal(nome: str) -> str:
    """
    Higieniza a descrição do produto de cupom fiscal:
    1. Remove prefixo fiscal de quantidade/embalagem ('1 MA - ', '0.34 KG - ', '1 PE - ', '1 PO - ', '1 BR - ')
    2. Remove unidades soltas e medidas tipo '23X22', '500ML', '2L', '1KG'
    3. Normaliza pontuações e expande abreviações fiscais conhecidas
    """
    texto = re.sub(PADRAO_EMBALAGEM, "", nome, flags=re.IGNORECASE)
    texto = re.sub(r"\b\d+([.,]\d+)?(kg|g|gr|l|ml|un|pc|m|cm|mm|x\d+)\b", " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\b\d+x\d+\b", " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"[/\\_\-]", " ", texto).lower().strip()

    for padrao, expansao in ABREVIACOES_FISCAIS.items():
        texto = re.sub(padrao, expansao, texto)

    return re.sub(r"\s+", " ", texto).strip()

# ---------------------------------------------------------------------------
# Regras Determinísticas de Altíssima Precisão (Expressões Regulares)
# Prioridade:
# 1. LIMPEZA antes de BEBIDAS (evita que água sanitária/alvejante caia em bebidas)
# 2. PADARIA_E_CONFEITARIA antes de LATICINIOS e FRUTAS (biscoito de nata cai em padaria)
# 3. MERCEARIA_SECA antes de FRUTAS (gelatina de uva/limão cai em mercearia)
# 4. BEBIDAS antes de FRUTAS (evita que suco/energético/néctar de uva caia em frutas)
# 5. UTILIDADES_DOMESTICAS (filtro de papel, fósforos, etc)
# 6. MERCEARIA_SECA antes de ACOUGUE (evita que caldo sazon de carne caia em açougue)
# ---------------------------------------------------------------------------
REGRAS_REGEX: Dict[str, str] = {
    "LIMPEZA": (
        r"\b(detergente|det\b|desinfetante|des\b|des lysoform|lysoform|sabao em po|sabao barra|"
        r"amaciante|agua sanitaria|alvejante|cloro|esponja|scotch brite|bombril|ype|veja|limpador|desengordurante|"
        r"multiuso|vassoura|rodo|saco lixo|lustra|lixivia|inseticida|guardanapo|guard\b|"
        r"papel toalha|toalha papel|lava roupas|lav r\b|omo|tixan|ariel|comfort|downy|brilhante)\b"
    ),
    "PADARIA_E_CONFEITARIA": (
        r"\b(pao|biscoito|bisc\b|bolacha|rosquinha|rosq\b|bolo|torta|croissant|baguete|salgado|coxinha|empada|"
        r"pastel|torrada|panetone|confeitaria|padaria)\b"
    ),
    "MERCEARIA_SECA": (
        r"\b(arroz|feijao|macarrao|mac\b|massa|espaguete|penne|oleo|azeite|farinha|biju|polvilho|fuba|"
        r"acucar|cafe|sal|molho|extrato|enlatado|milho|ervilha|sardinha|sard\b|atum|lentilha|grao de bico|"
        r"vinagre|mol tom|ext tom|bat pa\b|batata palha|yoki|salsaretti|conserva|maionese|maion\b|"
        r"ketchup|canjica|sucrilhos|cereal|cer\b|caldo|sazon|knorr|maggi|sabor ami|curry|tempero|temp\b|"
        r"siamar|kitano|colorau|cominho|oregano|chocolate|choc\b|ch\b|garoto|lacta|nestle tal|barra chocolate|bombom|"
        r"gelatina|po gel\b|ervas finas)\b"
    ),
    "BEBIDAS": (
        r"\b(refrigerante|refr\b|refri\b|ref\b|schweppes|coca|coca-cola|pepsi|guarana|fanta|"
        r"suco|nectar|cerveja|chopp|vinho|vodka|whisky|gin|energetico|energ\b|baly|red bull|monster|"
        r"tonica|cha|ice|gatorade|h2oh|agua(?! sanitaria))\b"
    ),
    "UTILIDADES_DOMESTICAS": (
        r"\b(lampada|pilha|panela|copo|prato|talher|bateria|utensilio|bazar|filtro de papel|filtro papel|filtro cafe)\b"
    ),
    "ACOUGUE_E_PEIXARIA": (
        r"\b(picanha|alcatra|contra|maminha|costela|frango|coxa|sobrecoxa|peito de frango|"
        r"carne|bov|suin|peixe|salmao|tilapia|camarao|linguica|ling\b|salsicha|bife|acougue|bacon|"
        r"pernil|mignon|patinho|acem|musculo|cupim|bovin\w*|suin\w*|paleta|pescado|bacalhau|toscana|calabresa|"
        r"miolo acem|file de frango|charque|carne seca)\b"
    ),
    "LATICINIOS_E_OVOS": (
        r"\b(leite|queijo|mussarela|mucarela|parmesao|iogurte|requeijao|req\b|manteiga|margarina|"
        r"ovo|ovos|nata|creme de leite|leite condensado|coalhada|ricota|gorgonzola|provolone|catupiry)\b"
    ),
    "CONGELADOS": (
        r"\b(congelad|cong\b|sorvete|pizza|lasanha|nugget|hamburguer|steak|batata cong|"
        r"polpa de fruta|acai|gelo)\b"
    ),
    "HIGIENE_E_BELEZA": (
        r"\b(shampoo|shamp\b|condicionador|cond\b|sabonete|sab l\b|sab emb\b|sab\b|creme dental|"
        r"pasta dente|escova dente|desodorante|desod\b|absorvente|cotonete|hidratante|"
        r"perfume|fio dental|papel higienico|fralda|protetor solar|esmalt)\b"
    ),
    "PET_SHOP": (
        r"\b(racao|petisco|pet\b|pedigree|whiskas|golden|premier|areia gato|coleira|"
        r"cachorro|caes|canino|felino)\b"
    ),
    "HORTIFRUTI_FRUTAS": (
        r"\b(maca|banana|laranja|uva|morango|abacaxi|melancia|melao|mamao|limao|pera|"
        r"manga|maracuja|kiwi|pessego|goiaba|tangerina|mexerica)\b"
    ),
    "HORTIFRUTI_VERDURAS_E_LEGUMES": (
        r"\b(alface|tomate|cebola|cebolinha|batata|cenoura|alho|pimentao|chuchu|abobrinha|abob\b|couve|"
        r"brocolis|espinafre|repolho|mandioca|aipim|rucula|cheiro verde|hortalica|"
        r"coentro|agriao|salsa|salsinha|manjericao|hortela|alecrim|acelga|escarola|chicoria|"
        r"beterraba|vagem|hidroponic\w*)\b"
    )
}

# ---------------------------------------------------------------------------
# Base de Conhecimento / Treinamento Especializada para Supermercados (SVM)
# ---------------------------------------------------------------------------
CORPUS_SUPERMERCADO: List[Tuple[str, str]] = [
    # ACOUGUE_E_PEIXARIA
    ("picanha bovina", "ACOUGUE_E_PEIXARIA"),
    ("alcatra bovina corte", "ACOUGUE_E_PEIXARIA"),
    ("contra file bovino", "ACOUGUE_E_PEIXARIA"),
    ("maminha bovina", "ACOUGUE_E_PEIXARIA"),
    ("costela bovina ripa", "ACOUGUE_E_PEIXARIA"),
    ("file de frango congelado", "ACOUGUE_E_PEIXARIA"),
    ("coxa e sobrecoxa frango", "ACOUGUE_E_PEIXARIA"),
    ("peito de frango", "ACOUGUE_E_PEIXARIA"),
    ("file de frango", "ACOUGUE_E_PEIXARIA"),
    ("miolo acem bovino", "ACOUGUE_E_PEIXARIA"),
    ("acem bovino", "ACOUGUE_E_PEIXARIA"),
    ("musculo bovino", "ACOUGUE_E_PEIXARIA"),
    ("carne musculo", "ACOUGUE_E_PEIXARIA"),
    ("musculo carnes", "ACOUGUE_E_PEIXARIA"),
    ("patinho bovino", "ACOUGUE_E_PEIXARIA"),
    ("cupim bovino", "ACOUGUE_E_PEIXARIA"),
    ("linguica toscana sadia", "ACOUGUE_E_PEIXARIA"),
    ("linguica perdigao na brasa", "ACOUGUE_E_PEIXARIA"),
    ("ling c sadia gra", "ACOUGUE_E_PEIXARIA"),
    ("ling t perdigao nabr", "ACOUGUE_E_PEIXARIA"),
    ("bacon fatiado defumado", "ACOUGUE_E_PEIXARIA"),
    ("pernil suino", "ACOUGUE_E_PEIXARIA"),
    ("costela suina", "ACOUGUE_E_PEIXARIA"),
    ("peixe salmao fresco", "ACOUGUE_E_PEIXARIA"),
    ("file tilapia", "ACOUGUE_E_PEIXARIA"),
    ("camarao cinza limpo", "ACOUGUE_E_PEIXARIA"),
    ("carne moida bovina", "ACOUGUE_E_PEIXARIA"),
    ("bife de chorizo", "ACOUGUE_E_PEIXARIA"),
    ("carne seca charque", "ACOUGUE_E_PEIXARIA"),

    # HORTIFRUTI_VERDURAS_E_LEGUMES
    ("couve manteiga", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("couve fresca", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("cenoura fresca", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("brocolis ninja", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("brocolis japones", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("brocolis japon ninja", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("tomate italiano", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("tomate carmem", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("beterraba fresca", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("beterraba", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("alface americana fechada", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("alface crespa hidroponica", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("vagem macarrao", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("vagem holandesa", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("vagem", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("hortela fresca", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("pimentao verde", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("batata inglesa", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("batata doce", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("abobrinha italia", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("abob italia", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("abobrinha menina", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("cebola branca", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("alho roxo", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("chuchu", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("espinafre", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("repolho verde", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("rucula hidroponica", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("coentro fresco", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("salsinha e cebolinha cheiro verde", "HORTIFRUTI_VERDURAS_E_LEGUMES"),
    ("mandioca aipim", "HORTIFRUTI_VERDURAS_E_LEGUMES"),

    # HORTIFRUTI_FRUTAS
    ("maca fuji", "HORTIFRUTI_FRUTAS"),
    ("maca gala", "HORTIFRUTI_FRUTAS"),
    ("mamao formosa", "HORTIFRUTI_FRUTAS"),
    ("mamao papaya", "HORTIFRUTI_FRUTAS"),
    ("laranja pera", "HORTIFRUTI_FRUTAS"),
    ("laranja bahia", "HORTIFRUTI_FRUTAS"),
    ("goiaba vermelha", "HORTIFRUTI_FRUTAS"),
    ("banana prata", "HORTIFRUTI_FRUTAS"),
    ("banana nanica", "HORTIFRUTI_FRUTAS"),
    ("abacaxi perola", "HORTIFRUTI_FRUTAS"),
    ("melancia fatiada", "HORTIFRUTI_FRUTAS"),
    ("melao amarelo", "HORTIFRUTI_FRUTAS"),
    ("morango bandeja", "HORTIFRUTI_FRUTAS"),
    ("uva thompson", "HORTIFRUTI_FRUTAS"),
    ("limao taiti", "HORTIFRUTI_FRUTAS"),
    ("pera williams", "HORTIFRUTI_FRUTAS"),
    ("manga tommy", "HORTIFRUTI_FRUTAS"),
    ("maracuja azedo", "HORTIFRUTI_FRUTAS"),

    # LATICINIOS_E_OVOS
    ("leite longa vida italac integral", "LATICINIOS_E_OVOS"),
    ("leite integral piracanjuba", "LATICINIOS_E_OVOS"),
    ("leite desnatado", "LATICINIOS_E_OVOS"),
    ("ovo branco cartela preti", "LATICINIOS_E_OVOS"),
    ("ovo e bco preti", "LATICINIOS_E_OVOS"),
    ("ovos vermelhos", "LATICINIOS_E_OVOS"),
    ("requeijao cremoso pocos de caldas", "LATICINIOS_E_OVOS"),
    ("req pocos calda", "LATICINIOS_E_OVOS"),
    ("requeijao catupiry", "LATICINIOS_E_OVOS"),
    ("queijo mussarela fatiado", "LATICINIOS_E_OVOS"),
    ("queijo prato fatiado", "LATICINIOS_E_OVOS"),
    ("queijo parmesao ralado", "LATICINIOS_E_OVOS"),
    ("manteiga com sal aviacao", "LATICINIOS_E_OVOS"),
    ("margarina qualy", "LATICINIOS_E_OVOS"),
    ("iogurte morango danone", "LATICINIOS_E_OVOS"),
    ("creme de leite nestle", "LATICINIOS_E_OVOS"),
    ("leite condensado moca", "LATICINIOS_E_OVOS"),

    # LIMPEZA
    ("guardanapo de papel fani", "LIMPEZA"),
    ("guard fani", "LIMPEZA"),
    ("guardanapo fani", "LIMPEZA"),
    ("guardanapo folha dupla", "LIMPEZA"),
    ("toalha papel fani", "LIMPEZA"),
    ("papel toalha kitchen", "LIMPEZA"),
    ("lava roupas tixan maciez", "LIMPEZA"),
    ("tixan maciez", "LIMPEZA"),
    ("lava roupas omo delic coco", "LIMPEZA"),
    ("lav r omo delic coco", "LIMPEZA"),
    ("sabao liquido omo", "LIMPEZA"),
    ("sabao em po ariel", "LIMPEZA"),
    ("amaciante confort concentrado", "LIMPEZA"),
    ("amaciante downy", "LIMPEZA"),
    ("detergente ype maca", "LIMPEZA"),
    ("detergente limpol", "LIMPEZA"),
    ("desinfetante lysoform suave", "LIMPEZA"),
    ("des lysoform", "LIMPEZA"),
    ("desinfetante pinho sol", "LIMPEZA"),
    ("agua sanitaria ype", "LIMPEZA"),
    ("alvejante vanish", "LIMPEZA"),
    ("limpador veja multiuso", "LIMPEZA"),
    ("esponja dupla face bombril", "LIMPEZA"),
    ("palha de aco bombril", "LIMPEZA"),
    ("saco de lixo reforçado", "LIMPEZA"),

    # MERCEARIA_SECA
    ("canjica cr yoki", "MERCEARIA_SECA"),
    ("canjica amarela yoki", "MERCEARIA_SECA"),
    ("maionese heinz tradicional", "MERCEARIA_SECA"),
    ("maion heinz tradicio", "MERCEARIA_SECA"),
    ("macarrao barilla ovos penne", "MERCEARIA_SECA"),
    ("mac barilla ovos pen", "MERCEARIA_SECA"),
    ("mac divella capel", "MERCEARIA_SECA"),
    ("macarrao divella capellini", "MERCEARIA_SECA"),
    ("macarrao espaguete dona benta", "MERCEARIA_SECA"),
    ("sardinha coqueiro em oleo", "MERCEARIA_SECA"),
    ("sard coqueiro oleo", "MERCEARIA_SECA"),
    ("atum solido gomes da costa", "MERCEARIA_SECA"),
    ("cereal sucrilhos kelloggs", "MERCEARIA_SECA"),
    ("cer sucrilhos", "MERCEARIA_SECA"),
    ("cereal matinal", "MERCEARIA_SECA"),
    ("arroz branco camil tipo 1", "MERCEARIA_SECA"),
    ("arroz tio joao", "MERCEARIA_SECA"),
    ("feijao carioca camil", "MERCEARIA_SECA"),
    ("oleo de soja liza", "MERCEARIA_SECA"),
    ("azeite de oliva andorinha", "MERCEARIA_SECA"),
    ("farinha de trigo dona benta", "MERCEARIA_SECA"),
    ("farinha de mandioca deusa biju", "MERCEARIA_SECA"),
    ("farinha mandioca biju", "MERCEARIA_SECA"),
    ("far ma deusa biju", "MERCEARIA_SECA"),
    ("caldo sazon carne", "MERCEARIA_SECA"),
    ("caldo knorr galinha", "MERCEARIA_SECA"),
    ("curry siamar", "MERCEARIA_SECA"),
    ("tempero baiano siamar", "MERCEARIA_SECA"),
    ("chocolate garoto tablete castanha", "MERCEARIA_SECA"),
    ("barra de chocolate garoto", "MERCEARIA_SECA"),
    ("acucar refinado uniao", "MERCEARIA_SECA"),
    ("cafe torrado e moido pilao", "MERCEARIA_SECA"),
    ("molho de tomate salsaretti", "MERCEARIA_SECA"),
    ("extrato de tomate elefante", "MERCEARIA_SECA"),
    ("vinagre de alcool castelo", "MERCEARIA_SECA"),
    ("sal refinado cisne", "MERCEARIA_SECA"),

    # BEBIDAS
    ("suco campo largo pessego", "BEBIDAS"),
    ("suco campo largo uva integral", "BEBIDAS"),
    ("refrigerante schweppes citrus", "BEBIDAS"),
    ("schweppes citrus pet", "BEBIDAS"),
    ("ref schweppes cit", "BEBIDAS"),
    ("energetico baly uva verde", "BEBIDAS"),
    ("energ baly uva v s a", "BEBIDAS"),
    ("refrigerante coca cola zero", "BEBIDAS"),
    ("refrigerante guaraná antarctica", "BEBIDAS"),
    ("cerveja heineken lata", "BEBIDAS"),
    ("cerveja amstel", "BEBIDAS"),
    ("vinho tinto chileno", "BEBIDAS"),
    ("agua mineral sem gas", "BEBIDAS"),
    ("energetico red bull", "BEBIDAS"),

    # HIGIENE_E_BELEZA
    ("shampoo loreal elseve", "HIGIENE_E_BELEZA"),
    ("condicionador pantene", "HIGIENE_E_BELEZA"),
    ("sabonete liquido monange detox", "HIGIENE_E_BELEZA"),
    ("sab l monange detox", "HIGIENE_E_BELEZA"),
    ("sabonete dove original", "HIGIENE_E_BELEZA"),
    ("creme dental colgate total 12", "HIGIENE_E_BELEZA"),
    ("desodorante rexona aerosol", "HIGIENE_E_BELEZA"),
    ("papel higienico neve folha dupla", "HIGIENE_E_BELEZA"),
    ("absorvente sempre livre", "HIGIENE_E_BELEZA"),
    ("fralda pampers confort sec", "HIGIENE_E_BELEZA"),

    # PET_SHOP
    ("racao premier caes adultos", "PET_SHOP"),
    ("racao golden filhotes", "PET_SHOP"),
    ("racao whiskas gatos castrados", "PET_SHOP"),
    ("petisco pedigree dentastix", "PET_SHOP"),
    ("areia sanitaria para gatos", "PET_SHOP"),
    ("coleira antipulgas seresto", "PET_SHOP"),

    # PADARIA_E_CONFEITARIA
    ("pao frances quentinho", "PADARIA_E_CONFEITARIA"),
    ("pao de forma wickbold", "PADARIA_E_CONFEITARIA"),
    ("bolo de cenoura com chocolate", "PADARIA_E_CONFEITARIA"),
    ("torta de frango com catupiry", "PADARIA_E_CONFEITARIA"),
    ("biscoito recheado passatempo", "PADARIA_E_CONFEITARIA"),
    ("torrada bauducco tradicional", "PADARIA_E_CONFEITARIA"),
    ("panetone bauducco frutas", "PADARIA_E_CONFEITARIA"),

    # CONGELADOS
    ("pizza congelada sadia calabresa", "CONGELADOS"),
    ("lasanha congelada perdigao quatro queijos", "CONGELADOS"),
    ("nuggets sadia crocante", "CONGELADOS"),
    ("hamburguer friboi congelado", "CONGELADOS"),
    ("sorvete kibon chicabon", "CONGELADOS"),
    ("batata congelada mccain corte tradicional", "CONGELADOS"),
    ("polpa de acai congelada", "CONGELADOS"),

    # UTILIDADES_DOMESTICAS
    ("lampada led philips 9w", "UTILIDADES_DOMESTICAS"),
    ("pilha alcalina duracell aa", "UTILIDADES_DOMESTICAS"),
    ("panela de pressao tramontina", "UTILIDADES_DOMESTICAS"),
    ("copo de vidro nadir figueiredo", "UTILIDADES_DOMESTICAS"),
    ("prato fundo duralex", "UTILIDADES_DOMESTICAS"),
    ("filtro de papel para cafe melitta", "UTILIDADES_DOMESTICAS"),
]

# ---------------------------------------------------------------------------
# Serviço de Classificação Inteligente (Linear SVM + TF-IDF)
# ---------------------------------------------------------------------------
class ModelService:
    def __init__(self):
        self.model: Optional[Pipeline] = None
        self.categorias: List[str] = list(REGRAS_REGEX.keys())

    def initialize(self):
        logger.info("Iniciando treinamento do modelo Linear SVM com TF-IDF (char + word n-grams)...")
        
        X_raw = [item[0] for item in CORPUS_SUPERMERCADO]
        X = [limpar_nome_fiscal(t) for t in X_raw]
        y = [item[1] for item in CORPUS_SUPERMERCADO]

        vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, analyzer="word")),
            ("char", TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True, analyzer="char_wb")),
        ])

        clf = CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=42),
            cv=3
        )

        self.model = Pipeline([
            ("tfidf", vectorizer),
            ("clf", clf)
        ])

        self.model.fit(X, y)
        logger.info(f"Modelo Linear SVM treinado com sucesso! {len(X)} amostras base indexadas.")

    def classify(self, text: str) -> Dict[str, any]:
        raw = text.strip()
        if not raw:
            return {"categoria": "DESCONHECIDO", "confianca": 0.0}

        cleaned = limpar_nome_fiscal(raw)

        # 1. Regras Determinísticas por Expressões Regulares (Confiança 0.98)
        for cat, pattern in REGRAS_REGEX.items():
            if re.search(pattern, cleaned):
                return {
                    "categoria": cat,
                    "confianca": 0.98
                }

        # 2. Classificador Inteligente Linear SVM
        if self.model:
            pred = self.model.predict([cleaned])[0]
            prob = float(np.max(self.model.predict_proba([cleaned])[0]))
            return {
                "categoria": pred,
                "confianca": round(prob, 4)
            }

        return {"categoria": "MERCEARIA_SECA", "confianca": 0.5}

model_service = ModelService()

# ---------------------------------------------------------------------------
# Ciclo de Vida do FastAPI (Lifespan)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.initialize()
    yield

# ---------------------------------------------------------------------------
# Aplicação FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Microsserviço de Categorização Inteligente de Produtos",
    description="Classifica nomes de produtos extraídos de faturas/notas fiscais usando abordagem híbrida de alta performance: Regras Fiscais + Linear SVM com TF-IDF.",
    version="3.1.0",
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
        "modelo": "LinearSVM + TF-IDF (char_wb + word n-grams)",
        "categorias_disponiveis": len(model_service.categorias)
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
