from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import Orange
import pandas as pd
import pickle
import os
from typing import Dict, Any

# --- Configuração da API ---
app = FastAPI(
    title="API de Predição com Múltiplos Modelos",
    description="Endpoint para prever o resultado de diferentes modelos de estratégia financeira.",
    version="1.0.0"
)

# --- Configuração dos Modelos ---
MODELOS_CONFIG = {
    # Modelo 1:
    "ModeloTAP_ETH_2_3.pkcls": {
        "target": "pnlToken",
        "variables": {
            'bestStrategy7Days': Orange.data.DiscreteVariable('bestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Trava de alta com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de baixa com PUT', 'Trava de baixa com CALL', 'Trava de alta com CALL', 'Calendário curto com PUT', 'Iron Fly']),
            'secondBestStrategy7Days': Orange.data.DiscreteVariable('secondBestStrategy7Days', values=['Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de baixa com PUT', 'Trava de baixa com CALL', 'Asymmetric Wings', 'Calendário curto com PUT']),
            'thirdBestStrategy7Days': Orange.data.DiscreteVariable('thirdBestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Calendário curto com PUT']),
            'fourthBestStrategy7Days': Orange.data.DiscreteVariable('fourthBestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de alta com PUT', 'Calendário curto com PUT']),
            'DiaEmissao': Orange.data.DiscreteVariable('DiaEmissao', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira']),
            'bestStrategy30Days': Orange.data.DiscreteVariable('bestStrategy30Days', values=['Asymmetric Wings', 'Calendário curto com PUT', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de alta com CALL', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Iron Condor', 'Iron Fly', 'Trava de baixa com PUT']),
            'DiaFechamento': Orange.data.DiscreteVariable('DiaFechamento', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira']),
            'secondBestStrategy30Days': Orange.data.DiscreteVariable('secondBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Calendário curto com PUT']),
            'thirdBestStrategy30Days': Orange.data.DiscreteVariable('thirdBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Venda Coberta OTM', 'Trava de alta com PUT', 'Trava de baixa com PUT', 'Calendário curto com PUT', 'Venda Coberta ATM', 'Trava de baixa com CALL']),
            'fourthBestStrategy30Days': Orange.data.DiscreteVariable('fourthBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Calendário curto com PUT'])
        }
    },
    # Modelo 2:
    "ModeloTAP_BTC_2_3_TH_90.pkcls": {
        "target": "Class_Threshold",
        "variables": {
            'bestStrategy7Days': Orange.data.DiscreteVariable('bestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Trava de alta com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de baixa com PUT', 'Trava de baixa com CALL', 'Trava de alta com CALL', 'Calendário curto com PUT']),
            'secondBestStrategy7Days': Orange.data.DiscreteVariable('secondBestStrategy7Days', values=['Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de baixa com PUT', 'Trava de baixa com CALL', 'Asymmetric Wings', 'Calendário curto com PUT']),
            'thirdBestStrategy7Days': Orange.data.DiscreteVariable('thirdBestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Calendário curto com PUT']),
            'fourthBestStrategy7Days': Orange.data.DiscreteVariable('fourthBestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de alta com PUT']),
            'DiaEmissao': Orange.data.DiscreteVariable('DiaEmissao', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira']),
            'bestStrategy30Days': Orange.data.DiscreteVariable('bestStrategy30Days', values=['Asymmetric Wings', 'Calendário curto com PUT', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de alta com CALL', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Iron Condor', 'Iron Fly', 'Trava de baixa com PUT']),
            'DiaFechamento': Orange.data.DiscreteVariable('DiaFechamento', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira']),
            'secondBestStrategy30Days': Orange.data.DiscreteVariable('secondBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Calendário curto com PUT','Trava de baixa com PUT']),
            'thirdBestStrategy30Days': Orange.data.DiscreteVariable('thirdBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Venda Coberta OTM', 'Trava de alta com PUT', 'Trava de baixa com PUT', 'Calendário curto com PUT', 'Venda Coberta ATM', 'Trava de baixa com CALL']),
            'fourthBestStrategy30Days': Orange.data.DiscreteVariable('fourthBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de alta com PUT', 'Trava de baixa com CALL'])
        }
    },
    # Modelo 3:
    "ModeloTBC_BTC_2_3_TH.pkcls": {
        "target": "pnlToken",
        "variables": {
            'bestStrategy7Days': Orange.data.DiscreteVariable('bestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Trava de alta com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de baixa com PUT', 'Trava de baixa com CALL', 'Trava de alta com CALL', 'Calendário curto com PUT']),
            'secondBestStrategy7Days': Orange.data.DiscreteVariable('secondBestStrategy7Days', values=['Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de baixa com PUT', 'Trava de baixa com CALL', 'Asymmetric Wings', 'Calendário curto com PUT']),
            'thirdBestStrategy7Days': Orange.data.DiscreteVariable('thirdBestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Calendário curto com PUT']),
            'fourthBestStrategy7Days': Orange.data.DiscreteVariable('fourthBestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de alta com PUT']),
            'DiaEmissao': Orange.data.DiscreteVariable('DiaEmissao', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira']),
            'bestStrategy30Days': Orange.data.DiscreteVariable('bestStrategy30Days', values=['Asymmetric Wings', 'Calendário curto com PUT', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de alta com CALL', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Iron Condor', 'Iron Fly', 'Trava de baixa com PUT']),
            'DiaFechamento': Orange.data.DiscreteVariable('DiaFechamento', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira']),
            'secondBestStrategy30Days': Orange.data.DiscreteVariable('secondBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Calendário curto com PUT','Trava de baixa com PUT']),
            'thirdBestStrategy30Days': Orange.data.DiscreteVariable('thirdBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Venda Coberta OTM', 'Trava de alta com PUT', 'Trava de baixa com PUT', 'Calendário curto com PUT', 'Venda Coberta ATM', 'Trava de baixa com CALL']),
            'fourthBestStrategy30Days': Orange.data.DiscreteVariable('fourthBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de alta com PUT', 'Trava de baixa com CALL'])
        }
    },
    # Modelo 4:
    "ModeloTAP_BTC_2_3.pkcls": {
        "target": "pnlToken",
        "variables": {
            'bestStrategy7Days': Orange.data.DiscreteVariable('bestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Trava de alta com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de baixa com PUT', 'Trava de baixa com CALL', 'Trava de alta com CALL', 'Calendário curto com PUT']),
            'secondBestStrategy7Days': Orange.data.DiscreteVariable('secondBestStrategy7Days', values=['Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de baixa com PUT', 'Trava de baixa com CALL', 'Asymmetric Wings', 'Calendário curto com PUT']),
            'thirdBestStrategy7Days': Orange.data.DiscreteVariable('thirdBestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Calendário curto com PUT']),
            'fourthBestStrategy7Days': Orange.data.DiscreteVariable('fourthBestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de alta com PUT', 'Calendário curto com PUT']),
            'DiaEmissao': Orange.data.DiscreteVariable('DiaEmissao', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira']),
            'bestStrategy30Days': Orange.data.DiscreteVariable('bestStrategy30Days', values=['Asymmetric Wings', 'Calendário curto com PUT', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de alta com CALL', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Iron Condor', 'Iron Fly', 'Trava de baixa com PUT']),
            'DiaFechamento': Orange.data.DiscreteVariable('DiaFechamento', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira']),
            'secondBestStrategy30Days': Orange.data.DiscreteVariable('secondBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Calendário curto com PUT','Trava de baixa com PUT']),
            'thirdBestStrategy30Days': Orange.data.DiscreteVariable('thirdBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Venda Coberta OTM', 'Trava de alta com PUT', 'Trava de baixa com PUT', 'Calendário curto com PUT', 'Venda Coberta ATM', 'Trava de baixa com CALL']),
            'fourthBestStrategy30Days': Orange.data.DiscreteVariable('fourthBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de alta com PUT', 'Trava de baixa com CALL'])
        }
    },    
    # Modelo 5:
    "ModeloTAP_ETH_13.pkcls": {
        "target": "pnlToken",
        "variables": {
            'bestStrategy7Days': Orange.data.DiscreteVariable('bestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Trava de alta com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de baixa com PUT', 'Trava de baixa com CALL', 'Trava de alta com CALL', 'Calendário curto com PUT', 'Iron Fly']),
            'secondBestStrategy7Days': Orange.data.DiscreteVariable('secondBestStrategy7Days', values=['Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de baixa com PUT', 'Trava de baixa com CALL', 'Asymmetric Wings', 'Calendário curto com PUT']),
            'thirdBestStrategy7Days': Orange.data.DiscreteVariable('thirdBestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Calendário curto com PUT']),
            'fourthBestStrategy7Days': Orange.data.DiscreteVariable('fourthBestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de alta com PUT', 'Calendário curto com PUT']),
            'DiaEmissao': Orange.data.DiscreteVariable('DiaEmissao', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira']),
            'bestStrategy30Days': Orange.data.DiscreteVariable('bestStrategy30Days', values=['Asymmetric Wings', 'Calendário curto com PUT', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de alta com CALL', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Iron Condor', 'Iron Fly', 'Trava de baixa com PUT']),
            'DiaFechamento': Orange.data.DiscreteVariable('DiaFechamento', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira']),
            'secondBestStrategy30Days': Orange.data.DiscreteVariable('secondBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Calendário curto com PUT']),
            'thirdBestStrategy30Days': Orange.data.DiscreteVariable('thirdBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Venda Coberta OTM', 'Trava de alta com PUT', 'Trava de baixa com PUT', 'Calendário curto com PUT', 'Venda Coberta ATM', 'Trava de baixa com CALL']),
            'fourthBestStrategy30Days': Orange.data.DiscreteVariable('fourthBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Calendário curto com PUT'])
        }
    },    
    # Modelo 6:
    "ModeloTBC_ETH_8_10.pkcls": {
        "target": "pnlToken",
        "variables": {
            'bestStrategy7Days': Orange.data.DiscreteVariable('bestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Trava de alta com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de baixa com PUT', 'Trava de baixa com CALL', 'Trava de alta com CALL', 'Calendário curto com PUT', 'Iron Fly']),
            'secondBestStrategy7Days': Orange.data.DiscreteVariable('secondBestStrategy7Days', values=['Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de baixa com PUT', 'Trava de baixa com CALL', 'Asymmetric Wings', 'Calendário curto com PUT']),
            'thirdBestStrategy7Days': Orange.data.DiscreteVariable('thirdBestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Calendário curto com PUT']),
            'fourthBestStrategy7Days': Orange.data.DiscreteVariable('fourthBestStrategy7Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de alta com PUT', 'Calendário curto com PUT']),
            'DiaEmissao': Orange.data.DiscreteVariable('DiaEmissao', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira']),
            'bestStrategy30Days': Orange.data.DiscreteVariable('bestStrategy30Days', values=['Asymmetric Wings', 'Calendário curto com PUT', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de alta com CALL', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Iron Condor', 'Iron Fly', 'Trava de baixa com PUT']),
            'DiaFechamento': Orange.data.DiscreteVariable('DiaFechamento', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira']),
            'secondBestStrategy30Days': Orange.data.DiscreteVariable('secondBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Calendário curto com PUT']),
            'thirdBestStrategy30Days': Orange.data.DiscreteVariable('thirdBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Venda Coberta OTM', 'Trava de alta com PUT', 'Trava de baixa com PUT', 'Calendário curto com PUT', 'Venda Coberta ATM', 'Trava de baixa com CALL']),
            'fourthBestStrategy30Days': Orange.data.DiscreteVariable('fourthBestStrategy30Days', values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Calendário curto com PUT'])
        }
    },    
    # Modelo 7:
    "ModeloTAP_BTC_3D.pkcls": {
        "target": "pnlToken",
        "variables": {
            'Duracao': Orange.data.DiscreteVariable('Duracao', values=['3']),
            'DiaEmissao': Orange.data.DiscreteVariable('DiaEmissao', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira']),
            'DiaFechamento': Orange.data.DiscreteVariable('DiaFechamento', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira'])
        }
    },    
    # Modelo 8:
    "ModeloTAP_BTC_7D.pkcls": {
        "target": "pnlToken",
        "variables": {
            'Duracao': Orange.data.DiscreteVariable('Duracao', values=['7']),
            'DiaEmissao': Orange.data.DiscreteVariable('DiaEmissao', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira']),
            'DiaFechamento': Orange.data.DiscreteVariable('DiaFechamento', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira'])
        }
    },    
    # Modelo 9:
    "ModeloTAP_ETH_3D.pkcls": {
        "target": "pnlToken",
        "variables": {
            'Duracao': Orange.data.DiscreteVariable('Duracao', values=['3']),
            'DiaEmissao': Orange.data.DiscreteVariable('DiaEmissao', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira']),
            'DiaFechamento': Orange.data.DiscreteVariable('DiaFechamento', values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira'])
        }
    },    

}

MODELOS_CARREGADOS: Dict[str, Any] = {}

# Define a estrutura dos dados de entrada esperados pela API
class PredictionPayload(BaseModel):
    model_name: str
    bestStrategy7Days: str
    secondBestStrategy7Days: str
    thirdBestStrategy7Days: str
    fourthBestStrategy7Days: str
    DiaEmissao: str
    bestStrategy30Days: str
    DiaFechamento: str
    secondBestStrategy30Days: str
    thirdBestStrategy30Days: str
    fourthBestStrategy30Days: str
    Duracao: str

@app.on_event("startup")
async def load_models():
    """Carrega todos os modelos e suas configurações na inicialização da API."""
    for model_file, config in MODELOS_CONFIG.items():
        if not os.path.exists(model_file):
            print(f"AVISO: O arquivo do modelo '{model_file}' não foi encontrado. Este modelo não estará disponível.")
            continue
        try:
            with open(model_file, 'rb') as file:
                MODELOS_CARREGADOS[model_file] = {
                    "model": pickle.load(file),
                    "target": Orange.data.DiscreteVariable(config["target"], values=['L', 'P']),
                    "domain": Orange.data.Domain([v for v in config["variables"].values()]),
                    "cols": [k for k in config["variables"].keys()]
                }
            print(f"Modelo '{model_file}' carregado com sucesso.")
        except Exception as e:
            print(f"ERRO: Não foi possível carregar o modelo '{model_file}'. Detalhes do erro: {e}.")
            # A API continuará a rodar, mas este modelo não estará disponível

# --- Endpoint da API ---
@app.post("/predict/")
async def predict(payload: PredictionPayload):
    """
    Realiza a previsão usando o modelo especificado no payload.
    
    - **Dados de entrada:** Um objeto JSON com o nome do modelo e as variáveis preditoras.
    - **Resposta:** Um objeto JSON com a previsão.
    """
    model_name = payload.model_name
    
    if model_name not in MODELOS_CARREGADOS:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_name}' não encontrado ou não carregado no servidor.")
    
    model_info = MODELOS_CARREGADOS[model_name]
    
    modelo_preditivo = model_info["model"]
    target_variable = model_info["target"]
    orange_domain_input = model_info["domain"]
    colunas_preditoras = model_info["cols"]

    # Converte o payload para um DataFrame pandas
    data_dict = payload.dict(exclude={'model_name'})
    df_para_predicao = pd.DataFrame([data_dict])

    # Garante que as colunas estão na ordem correta
    try:
        df_ordered = df_para_predicao[colunas_preditoras]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Erro: Coluna preditora faltando no payload para o modelo {model_name}: {e}")

    # Converte o DataFrame para Orange.data.Table
    try:
        orange_table_para_predicao = Orange.data.Table.from_list(
            domain=orange_domain_input,
            rows=df_ordered.values.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na conversão dos dados para o formato do modelo {model_name}: {e}")

    # Realiza a predição
    try:
        predicoes_raw = modelo_preditivo(orange_table_para_predicao)
        predicao = target_variable.values[int(predicoes_raw[0])]
        
        return {"predicao_result": predicao, "model_used": model_name}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao realizar a previsão com o modelo {model_name}. Detalhes: {e}")
