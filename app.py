from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import Orange
import pandas as pd
import pickle
import os

# --- Configuração da API ---
app = FastAPI(
    title="Modelo de Predição de Estratégias",
    description="Endpoint para prever o resultado de pnlToken a partir de estratégias financeiras.",
    version="1.0.0"
)

# --- Variáveis Globais para o Modelo e a Estrutura de Dados ---
# Elas serão carregadas na inicialização do servidor.
MODELO_PREDITIVO = None
TARGET_VARIABLE = None
ORANGE_DOMAIN_INPUT = None
COLUNAS_PREDITORAS = []

# --- Configuração das Variáveis e seus Tipos (CLASSES) ---
# Dicionário mapeando o nome das variáveis para suas classes (tipos) do Orange.
VARIAVEIS_E_TIPOS = {
    'bestStrategy7Days': Orange.data.DiscreteVariable(
        'bestStrategy7Days',
        values=['Asymmetric Wings', 'Iron Condor', 'Trava de alta com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM']
    ),
    'secondBestStrategy7Days': Orange.data.DiscreteVariable(
        'secondBestStrategy7Days',
        values=['Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM']
    ),
    'thirdBestStrategy7Days': Orange.data.DiscreteVariable(
        'thirdBestStrategy7Days',
        values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM']
    ),
    'fourthBestStrategy7Days': Orange.data.DiscreteVariable(
        'fourthBestStrategy7Days',
        values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de baixa com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM', 'Venda Coberta OTM']
    ),
    'DiaEmissao': Orange.data.DiscreteVariable(
        'DiaEmissao',
        values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira']
    ),
    'bestStrategy30Days': Orange.data.DiscreteVariable(
        'bestStrategy30Days',
        values=['Asymmetric Wings', 'Calendário curto com PUT', 'Trava de alta com PUT', 'Trava de baixa com CALL']
    ),
    'DiaFechamento': Orange.data.DiscreteVariable(
        'DiaFechamento',
        values=['domingo', 'quarta-feira', 'quinta-feira', 'sábado', 'segunda-feira', 'sexta-feira', 'terça-feira']
    ),
    'secondBestStrategy30Days': Orange.data.DiscreteVariable(
        'secondBestStrategy30Days',
        values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de alta com PUT', 'Trava de baixa com CALL']
    ),
    'thirdBestStrategy30Days': Orange.data.DiscreteVariable(
        'thirdBestStrategy30Days',
        values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Venda Coberta OTM']
    ),
    'fourthBestStrategy30Days': Orange.data.DiscreteVariable(
        'fourthBestStrategy30Days',
        values=['Asymmetric Wings', 'Iron Condor', 'Iron Fly', 'Trava de alta com CALL', 'Trava de baixa com PUT', 'Venda Coberta ATM']
    ),
    # Variável alvo (target variable)
    'pnlToken': Orange.data.DiscreteVariable('pnlToken', values=['L', 'P']),
}

# Define a estrutura dos dados de entrada esperados pela API
class PredictionPayload(BaseModel):
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

@app.on_event("startup")
async def load_model():
    """Carrega o modelo e define a estrutura de dados na inicialização da API."""
    global MODELO_PREDITIVO, TARGET_VARIABLE, ORANGE_DOMAIN_INPUT, COLUNAS_PREDITORAS
    
    # Nome do arquivo do modelo
    NOME_ARQUIVO_MODELO = 'ModeloTAP_ETH_2_3.pkcls'
    
    # Carrega o modelo
    if not os.path.exists(NOME_ARQUIVO_MODELO):
        raise RuntimeError(f"Erro: O arquivo do modelo '{NOME_ARQUIVO_MODELO}' não foi encontrado.")
    try:
        with open(NOME_ARQUIVO_MODELO, 'rb') as file:
            MODELO_PREDITIVO = pickle.load(file)
        print(f"Modelo '{NOME_ARQUIVO_MODELO}' carregado com sucesso.")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o modelo: {e}")

    # Prepara o domínio do Orange
    domain_attributes = []
    for var_name, var_obj in VARIAVEIS_E_TIPOS.items():
        if var_name == 'pnlToken':
            TARGET_VARIABLE = var_obj
        else:
            domain_attributes.append(var_obj)
    
    ORANGE_DOMAIN_INPUT = Orange.data.Domain(domain_attributes)
    COLUNAS_PREDITORAS = [attr.name for attr in domain_attributes]
    
    print("Estrutura de domínio do Orange preparada.")

# --- Endpoint da API ---
@app.post("/predict/")
async def predict(payload: PredictionPayload):
    """
    Realiza a previsão do pnlToken com base nos dados fornecidos.
    
    - **Dados de entrada:** Um objeto JSON com as variáveis preditoras.
    - **Resposta:** Um objeto JSON com a previsão.
    """
    if not MODELO_PREDITIVO or not TARGET_VARIABLE or not ORANGE_DOMAIN_INPUT or not COLUNAS_PREDITORAS:
        raise HTTPException(status_code=500, detail="O servidor não está pronto. O modelo não foi carregado corretamente.")

    # Converte o payload para um DataFrame pandas
    data_dict = payload.dict()
    df_para_predicao = pd.DataFrame([data_dict])

    # Garante que as colunas estão na ordem correta
    try:
        df_ordered = df_para_predicao[COLUNAS_PREDITORAS]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Erro: Coluna preditora faltando no payload: {e}")

    # Converte o DataFrame para Orange.data.Table
    try:
        orange_table_para_predicao = Orange.data.Table.from_list(
            domain=ORANGE_DOMAIN_INPUT,
            rows=df_ordered.values.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na conversão dos dados para o formato do modelo: {e}")

    # Realiza a predição
    try:
        predicoes_raw = MODELO_PREDITIVO(orange_table_para_predicao)
        predicao = TARGET_VARIABLE.values[int(predicoes_raw[0])]
        
        return {"predicao_pnlToken": predicao}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao realizar a previsão. Detalhes: {e}")
