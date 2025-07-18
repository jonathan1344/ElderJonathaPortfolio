import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
import traceback
import logging
import time
import warnings
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from skopt import BayesSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from scipy.sparse import issparse
import xlsxwriter

# --- Configuração de Logging ---
# Configura o sistema de log para direcionar a saída para um arquivo e para o console.
# Verifica se os handlers já existem para evitar duplicidade em caso de re-execução.
if not logging.getLogger().handlers:
    try:
        # Handler para escrever logs no arquivo
        file_handler = logging.FileHandler('inventory_pipeline.log', mode='w')
        file_handler.setLevel(logging.INFO)

        # Handler para escrever logs no console (apenas mensagens de nível INFO e superior)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Define o formato das mensagens de log
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Adiciona o formatador aos handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Obtém o logger raiz e define o nível mínimo de log
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Adiciona os handlers ao logger raiz
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        logging.info("Sistema de logging configurado (arquivo 'inventory_pipeline.log' e console).")

    except Exception as e:
        # Em caso de falha na configuração do logging, exibe um erro no console
        # e configura um logging básico apenas para o console.
        print(f"ERRO: Falha ao configurar o logging: {e}")
        print("O script continuará, mas o logging detalhado pode estar ausente.")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.warning("Logging de arquivo falhou, usando apenas console.")

# Ajusta o nível de log para algumas bibliotecas para reduzir a verbosidade.
logging.getLogger('xgboost').setLevel(logging.ERROR)
logging.getLogger('lightgbm').setLevel(logging.ERROR)
logging.getLogger('skopt').setLevel(logging.ERROR)

# --- Supressão de Avisos ---
# Configurações para suprimir avisos específicos de bibliotecas externas que
# podem poluir a saída do log ou console, mas que são esperados no contexto do script.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="feature_fraction is set=1.0, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=1.0")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# Definição Centralizada de Caminhos de Arquivo
# =============================================================================
# Define os caminhos para os arquivos de entrada e saída do pipeline.
# Utiliza caminhos relativos baseados no diretório onde o script está localizado.

# Determina o diretório base do script (a pasta PYTHON)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define o diretório pai (a pasta KPI performence - Rede)
PARENT_DIR = os.path.join(BASE_DIR, os.pardir)

PATHS = {
    # Caminho para o arquivo CSV de estoque atual.
    'estoque_atual_csv': os.path.join(PARENT_DIR, "Inventario", "Contetor Estante Atual", "Contentor+em+Estante.csv"),
    # Caminho para o arquivo Excel da base de custos.
    'base_custos_xlsx': os.path.join(PARENT_DIR, "Inventario", "Base Custos.xlsx"),
    # Caminhos para os arquivos CSV de histórico de inventário fechado (histórico e atual).
    'inventarios_fechados_historico_csv': os.path.join(PARENT_DIR, "Machine Learning", "Inventarios+Fechados+Contas", "Inventarios+Fechados+Contas+Historico.csv"),
    'inventarios_fechados_atual_csv': os.path.join(PARENT_DIR, "Machine Learning", "Inventarios+Fechados+Contas", "Inventarios+Fechados+Contas.csv"),
    # Caminho para a pasta contendo os arquivos CSV de histórico de movimentação.
    'movimentacao_folder': os.path.join(PARENT_DIR, "Machine Learning", "historico movimentação"),
    # Caminho para o arquivo de saída intermediário da Curva ABC.
    'curva_abc_xlsx': os.path.join(BASE_DIR, "Inventario", "CURVA_ABC.xlsx"),
    # Caminho para o arquivo de saída intermediário dos resultados mensais detalhados.
    'resultados_mensais_detalhados_xlsx': os.path.join(BASE_DIR, "Inventario", "Resultados_Mensais_Detalhados.xlsx"),
    # Caminho para o arquivo de saída final consolidado do pipeline.
    'previsao_inventario_final_xlsx': os.path.join(BASE_DIR, "Inventario", "Previsao_Inventario_Final.xlsx"),
}

# =============================================================================
# Funções Auxiliares
# =============================================================================

def standardize_codigo(df, column_name='Código'):
    """
    Padroniza a coluna de códigos dos itens.
    Converte os códigos para string e remove zeros à esquerda para garantir consistência.

    Args:
        df (pd.DataFrame): DataFrame contendo a coluna de códigos.
        column_name (str): Nome da coluna contendo os códigos dos itens.

    Returns:
        pd.DataFrame: DataFrame com a coluna de códigos padronizada.
    """
    if column_name in df.columns:
        df[column_name] = df[column_name].astype(str).str.strip().str.lstrip('0')
        df[column_name] = df[column_name].replace('', '0')
    return df

# =============================================================================
# Módulo 1: Geração da Curva ABC
# =============================================================================

def generate_curva_abc(caminho_estoque, caminho_custos, caminho_saida):
    """
    Carrega dados de estoque e custos, calcula o valor de estoque por item,
    classifica os itens em Curva ABC e exporta o resultado para um arquivo Excel.

    Args:
        caminho_estoque (str): Caminho para o arquivo CSV de estoque atual.
        caminho_custos (str): Caminho para o arquivo Excel da base de custos.
        caminho_saida (str): Caminho para o arquivo Excel de saída da Curva ABC.

    Returns:
        tuple: Uma tupla contendo um booleano indicando sucesso (True) ou falha (False)
               e o DataFrame resultante da Curva ABC (ou um DataFrame vazio em caso de falha).
    """
    logging.info("Iniciando a geração da Curva ABC.")

    # Define os nomes das colunas chave nos arquivos de entrada.
    coluna_produto_estoque = "Produto"
    coluna_descr_produto = "Descr. Produto"
    coluna_material_custos = "Material"
    coluna_custo = "Custo Médio"

    # --- Carregar e Preparar Dados ---
    # Carrega os dados de estoque e custos dos arquivos especificados.
    try:
        estoque_df = pd.read_csv(caminho_estoque, delimiter=';', encoding='latin1', low_memory=False)
        logging.info(f"Arquivo de estoque carregado: {caminho_estoque}")
    except FileNotFoundError:
        logging.critical(f"Erro: Arquivo de estoque não encontrado em {caminho_estoque}. Geração da Curva ABC abortada.")
        return False, pd.DataFrame()
    except Exception as e:
        logging.critical(f"Erro ao carregar arquivo de estoque: {e}. Geração da Curva ABC abortada.")
        logging.error(traceback.format_exc())
        return False, pd.DataFrame()

    try:
        custos_df = pd.read_excel(caminho_custos)
        logging.info(f"Arquivo de custos carregado: {caminho_custos}")
    except FileNotFoundError:
        logging.critical(f"Erro: Arquivo de custos não encontrado em {caminho_custos}. Geração da Curva ABC abortada.")
        return False, pd.DataFrame()
    except Exception as e:
        logging.critical(f"Erro ao carregar arquivo de custos: {e}. Geração da Curva ABC abortada.")
        logging.error(traceback.format_exc())
        return False, pd.DataFrame()

    # --- Limpeza e Padronização das Colunas de Mesclagem ---
    # Realiza a limpeza e padronização das colunas de código e custo para garantir
    # que a mesclagem entre os DataFrames de estoque e custo seja bem-sucedida.
    logging.info("Padronizando colunas de código e custos para mesclagem.")
    try:
        estoque_df[coluna_produto_estoque] = estoque_df[coluna_produto_estoque].astype(str).str.strip()
        custos_df[coluna_material_custos] = custos_df[coluna_material_custos].astype(str).str.strip()

        # Renomeia colunas no DataFrame de custos para facilitar a fusão.
        custos_df = custos_df.rename(columns={
            coluna_material_custos: coluna_produto_estoque,
            coluna_custo: coluna_custo,
            "Ano": "Ano",
            "Período": "Mes"
        })

        # Converte colunas de ano e mês para números para ordenação correta.
        custos_df["Ano"] = pd.to_numeric(custos_df["Ano"], errors='coerce')
        custos_df["Mes"] = pd.to_numeric(custos_df["Mes"], errors='coerce')

        # Seleciona apenas o custo mais recente para cada produto com base no ano e mês.
        custos_recente = custos_df.sort_values(by=["Ano", "Mes"], ascending=[False, False])\
                                .drop_duplicates(subset=[coluna_produto_estoque], keep='first')
        logging.info("Custo mais recente selecionado para cada produto.")

        # Mescla os dados de estoque com os custos mais recentes.
        estoque_df = estoque_df.merge(
            custos_recente[[coluna_produto_estoque, coluna_custo]],
            on=coluna_produto_estoque,
            how='left'
        )
        logging.info("Dados de estoque mesclados com custos recentes.")

        # Calcula o valor total de estoque para cada item.
        estoque_df["Valor Estoque"] = estoque_df["Estoqueada"] * estoque_df[coluna_custo]
        logging.info("Valor de estoque calculado para cada item.")

    except Exception as e:
        logging.critical(f"Erro durante a limpeza e padronização para mesclagem: {e}. Geração da Curva ABC abortada.")
        logging.error(traceback.format_exc())
        return False, pd.DataFrame()


    # --- Agrupar e Calcular Métricas para a Curva ABC ---
    # Agrupa os dados por produto e calcula as métricas necessárias para a Curva ABC.
    logging.info("Agrupando dados e calculando métricas para a Curva ABC.")
    try:
        resultado_df = estoque_df.groupby([coluna_produto_estoque, coluna_descr_produto]).agg(
            Total_Estoqueada=("Estoqueada", "sum"),
            Quantidade_Locações=("Locação", "nunique"),
            Custo_Unitário=(coluna_custo, "first"),
            Valor_Estoque=("Valor Estoque", "sum")
        ).reset_index()

        # Preenche valores nulos nas colunas de valor e custo com zero.
        resultado_df['Valor_Estoque'] = resultado_df['Valor_Estoque'].fillna(0)
        resultado_df['Custo_Unitário'] = resultado_df['Custo_Unitário'].fillna(0)

        # Ordena os itens pelo valor de estoque em ordem decrescente.
        resultado_df = resultado_df.sort_values(by="Valor_Estoque", ascending=False)

        # Calcula o valor total geral de todos os itens no estoque.
        valor_total_geral = resultado_df['Valor_Estoque'].sum()
        logging.info(f"Valor total geral de estoque calculado: {valor_total_geral:,.2f}")

        # Calcula a participação percentual individual de cada item no valor total.
        resultado_df['% individual'] = resultado_df['Valor_Estoque'] / valor_total_geral if valor_total_geral > 0 else 0

        # Calcula a participação percentual acumulada.
        resultado_df['% acumulada'] = resultado_df['% individual'].cumsum()
        logging.info("Porcentagens individual e acumulada calculadas.")

        # Classifica os itens nas categorias A, B ou C com base na participação acumulada.
        resultado_df['Curva ABC'] = 'C'
        resultado_df.loc[resultado_df['% acumulada'] <= 0.95, 'Curva ABC'] = 'B'
        resultado_df.loc[resultado_df['% acumulada'] <= 0.80, 'Curva ABC'] = 'A'
        logging.info("Classificação Curva ABC (A, B, C) realizada.")

    except Exception as e:
        logging.critical(f"Erro durante o cálculo das métricas da Curva ABC: {e}. Geração da Curva ABC abortada.")
        logging.error(traceback.format_exc())
        return False, pd.DataFrame()


    # --- Formatar e Salvar Resultado ---
    # Formata o DataFrame resultante e o salva em um arquivo Excel com formatação específica.
    logging.info(f"Formatando e salvando o arquivo da Curva ABC em: {caminho_saida}")
    try:
        # Cria o diretório de saída se ele não existir.
        output_dir = os.path.dirname(caminho_saida)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Diretório de saída criado: {output_dir}")

        # Renomeia colunas para o formato de saída desejado no arquivo Excel.
        resultado_df_output = resultado_df.rename(columns={
            coluna_produto_estoque: "Desenho",
            coluna_descr_produto: "Descrição",
            "Quantidade_Locações": "Qtde. Locação",
            "Total_Estoqueada": "Soma de Estoqueada",
            "Custo_Unitário": "Custo Médio",
            "Valor_Estoque": "Soma de Valor Total"
        })

        # Define a ordem das colunas no arquivo de saída.
        colunas_saida = [
            "Desenho", "Descrição", "Qtde. Locação", "Soma de Estoqueada",
            "Custo Médio", "Soma de Valor Total", "% individual", "% acumulada", "Curva ABC"
        ]
        resultado_df_output = resultado_df_output[colunas_saida]

        # Salva o DataFrame em um arquivo Excel utilizando xlsxwriter para formatação.
        with pd.ExcelWriter(caminho_saida, engine='xlsxwriter') as writer:
            resultado_df_output.to_excel(writer, index=False, sheet_name='CURVA_ABC')

            # Aplica formatação específica às colunas da planilha.
            workbook = writer.book
            worksheet = writer.sheets['CURVA_ABC']

            # Mapeia os nomes das colunas para seus índices para aplicar a formatação.
            col_mapping_for_format = {col: idx for idx, col in enumerate(colunas_saida)}

            text_format = workbook.add_format({'num_format': '@'})
            if 'Desenho' in col_mapping_for_format: worksheet.set_column(col_mapping_for_format['Desenho'], col_mapping_for_format['Desenho'], None, text_format)

            num_format = workbook.add_format({'num_format': '#,##0'})
            if 'Qtde. Locação' in col_mapping_for_format: worksheet.set_column(col_mapping_for_format['Qtde. Locação'], col_mapping_for_format['Qtde. Locação'], None, num_format)
            if 'Soma de Estoqueada' in col_mapping_for_format: worksheet.set_column(col_mapping_for_format['Soma de Estoqueada'], col_mapping_for_format['Soma de Estoqueada'], None, num_format)

            money_format = workbook.add_format({'num_format': 'R$ #,##0.00'})
            if 'Custo Médio' in col_mapping_for_format: worksheet.set_column(col_mapping_for_format['Custo Médio'], col_mapping_for_format['Custo Médio'], None, money_format)
            if 'Soma de Valor Total' in col_mapping_for_format: worksheet.set_column(col_mapping_for_format['Soma de Valor Total'], col_mapping_for_format['Soma de Valor Total'], None, money_format)

            percent_format = workbook.add_format({'num_format': '0.00%'})
            if '% individual' in col_mapping_for_format: worksheet.set_column(col_mapping_for_format['% individual'], col_mapping_for_format['% individual'], None, percent_format)
            if '% acumulada' in col_mapping_for_format: worksheet.set_column(col_mapping_for_format['% acumulada'], col_mapping_for_format['% acumulada'], None, percent_format)

        logging.info(f"Arquivo da Curva ABC salvo com sucesso em: {caminho_saida}")
        # Retorna True indicando sucesso e o DataFrame original (sem renomeamento para exportação)
        return True, resultado_df

    except Exception as e:
        logging.critical(f"Erro ao salvar arquivo Excel da Curva ABC: {e}. Geração da Curva ABC abortada.")
        logging.error(traceback.format_exc())
        return False, pd.DataFrame()

# =============================================================================
# Módulo 2: Processamento de Inventários Fechados
# =============================================================================

def load_and_combine_inventory_history(file_paths):
    """
    Carrega múltiplos arquivos CSV de histórico de inventário, concatena-os
    e retorna um único DataFrame combinado.

    Args:
        file_paths (list): Lista de caminhos para os arquivos CSV de histórico de inventário.

    Returns:
        pd.DataFrame: DataFrame combinado de todos os arquivos de histórico.
    """
    logging.info("Iniciando carregamento e combinação de arquivos de histórico de inventário.")
    li = []
    # Itera sobre cada caminho de arquivo fornecido.
    for file_path in file_paths:
        # Verifica se o arquivo existe antes de tentar carregar.
        if not os.path.exists(file_path):
            logging.warning(f"Arquivo de histórico de inventário não encontrado: {file_path}. Pulando este arquivo.")
            continue
        try:
            # Carrega o arquivo CSV.
            df = pd.read_csv(file_path, delimiter=";", encoding="latin1", low_memory=False)
            li.append(df)
            logging.info(f"Arquivo de histórico de inventário carregado: {file_path}")
        except Exception as e:
            logging.error(f"Erro ao carregar arquivo de histórico de inventário {file_path}: {e}")
            logging.error(traceback.format_exc())
            continue

    # Verifica se algum DataFrame foi carregado com sucesso.
    if not li:
        logging.critical("Nenhum arquivo de histórico de inventário carregado com sucesso. Processamento abortado.")
        return pd.DataFrame()

    # Concatena todos os DataFrames carregados em um único DataFrame.
    combined_df = pd.concat(li, axis=0, ignore_index=True)
    logging.info(f"Todos os arquivos de histórico de inventário concatenados. Dimensões iniciais: {combined_df.shape}")

    return combined_df


def process_inventarios_fechados(df_inventarios_combinados, arquivo_destino):
    """
    Processa o DataFrame combinado de inventários fechados, realiza limpeza,
    transformações, cálculos de métricas e agrupações, exportando o resultado
    tratado para Excel.

    Args:
        df_inventarios_combinados (pd.DataFrame): DataFrame contendo o histórico de inventários combinados.
        arquivo_destino (str): Caminho para o arquivo Excel de saída dos resultados processados.

    Returns:
        tuple: Uma tupla contendo um booleano indicando sucesso (True) ou falha (False)
               e o DataFrame resultante dos resultados processados (ou um DataFrame vazio em caso de falha).
    """
    logging.info("Iniciando o processamento de Inventários Fechados (a partir de dados combinados).")

    # Verifica se o DataFrame de entrada está vazio.
    if df_inventarios_combinados.empty:
        logging.critical("DataFrame combinado de inventários está vazio. Processamento de Inventários Fechados abortado.")
        return False, pd.DataFrame()

    # Cria uma cópia do DataFrame para evitar modificar o original.
    df = df_inventarios_combinados.copy()

    # =============================================================================
    # Verificação das colunas necessárias
    # =============================================================================
    # Verifica a presença de colunas essenciais para o processamento.
    colunas_necessarias = [
        "NOK", "Status", "Código", "Locação", "UdE", "Compartimento",
        "Status Inventario", "Atualização", "Física", "Quantidade Fisica", "Custo", "Lógica"
    ]
    for col in colunas_necessarias:
        if col not in df.columns:
            logging.critical(f"Erro: Coluna essencial '{col}' não encontrada no DataFrame combinado de Inventários Fechados. Processamento abortado.")
            return False, pd.DataFrame()
    logging.info("Colunas essenciais verificadas no DataFrame combinado.")

    # =============================================================================
    # Filtragem inicial
    # =============================================================================
    # Aplica filtros iniciais para remover dados irrelevantes ou incompletos.
    logging.info("Aplicando filtros iniciais nos dados combinados.")
    try:
        # Desconsidera linhas onde "Código", "Locação" ou "UdE" estão vazios.
        df = df[(df["Código"].notna()) & (df["Locação"].notna()) & (df["UdE"].notna())].copy()

        # Filtra linhas onde NOK está vazio ou nulo e Status é "FECHADO".
        df = df[(df["NOK"].isna() | (df["NOK"] == "")) & (df["Status"] == "FECHADO")].copy()
        logging.info(f"Dados filtrados. Dimensões: {df.shape}")

    except Exception as e:
        logging.critical(f"Erro durante a filtragem inicial dos dados combinados: {e}. Processamento abortado.")
        logging.error(traceback.format_exc())
        return False, pd.DataFrame()


    # =============================================================================
    # Conversão de tipos de colunas
    # =============================================================================
    # Converte colunas para os tipos de dados apropriados (data, numérico, float).
    logging.info("Convertendo tipos de colunas.")
    try:
        # Converte a coluna de data "Atualização" para o tipo datetime.
        df["Atualização"] = pd.to_datetime(df["Atualização"], format="%d/%m/%Y %H:%M", errors="coerce")
        logging.info("Coluna 'Atualização' convertida para datetime.")

        # Verifica se a coluna "Atualização" foi convertida corretamente.
        if df["Atualização"].isna().all():
            logging.critical("Erro: A coluna 'Atualização' está vazia ou não foi convertida corretamente para data. Processamento abortado.")
            return False, pd.DataFrame()

        # Converte colunas numéricas para o tipo numérico.
        colunas_numericas = ["UdE", "Compartimento", "Contagem", "Quantidade Fisica", "Física", "Lógica"]
        for col in colunas_numericas:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        logging.info("Colunas numéricas convertidas.")

        # Converte a coluna "Custo" para float, tratando separadores decimais.
        df["Custo"] = df["Custo"].astype(str).str.replace(".", "").str.replace(",", ".").astype(float)
        logging.info("Coluna 'Custo' convertida para float.")

        # Verifica se a coluna "Custo" foi convertida corretamente.
        if df["Custo"].isna().all():
            logging.critical('Erro: A coluna "Custo" está vazia ou não foi convertida corretamente para float. Processamento abortado.')
            return False, pd.DataFrame()

    except Exception as e:
        logging.critical(f"Erro durante a conversão de tipos de colunas: {e}. Processamento abortado.")
        logging.error(traceback.format_exc())
        return False, pd.DataFrame()


    # =============================================================================
    # Transformações de colunas
    # =============================================================================
    # Realiza transformações nas colunas existentes para criar novas features ou padronizar dados.
    logging.info("Aplicando transformações em colunas.")
    try:
        # Padroniza a coluna "Código".
        df["Código"] = df["Código"].apply(lambda x: ("0000000000000" + str(x))[-13:])
        logging.info("Coluna 'Código' padronizada.")

        # Cria colunas separadas para Data, Ano e Mês a partir de "Atualização".
        df["Data INV"] = df["Atualização"].dt.date
        df["Ano"] = df["Atualização"].dt.year
        df["Mês"] = df["Atualização"].dt.month
        logging.info("Colunas de data (Data INV, Ano, Mês) criadas.")

        # Extrai a "Estante" da coluna "Locação".
        df["Estante"] = df["Locação"].apply(lambda x: x.split(".")[0] if isinstance(x, str) else x)
        logging.info("Coluna 'Estante' criada.")

    except Exception as e:
        logging.critical(f"Erro durante as transformações de colunas: {e}. Processamento abortado.")
        logging.error(traceback.format_exc())
        return False, pd.DataFrame()


    # =============================================================================
    # Criação da chave única ("CHAVE")
    # =============================================================================
    # Cria uma chave única para cada registro combinando informações relevantes.
    logging.info("Criando chave única para cada registro.")
    try:
        df["CHAVE"] = df.apply(
            lambda row: f"{row['Data INV']}_{row['Código']}_{row['Locação']}_{int(row['UdE'])}_{int(row['Compartimento'])}_{row['Status Inventario']}"
            if pd.notna(row['Data INV']) and pd.notna(row['Código']) and pd.notna(row['Locação']) and pd.notna(row['UdE']) and pd.notna(row['Compartimento']) and pd.notna(row['Status Inventario'])
            else None,
            axis=1
        )
        # Remove linhas onde a chave não pôde ser criada devido a valores nulos.
        df.dropna(subset=["CHAVE"], inplace=True)
        logging.info("Coluna 'CHAVE' criada e linhas com chave inválida removidas.")
    except Exception as e:
        logging.critical(f"Erro ao criar a chave única: {e}. Processamento abortado.")
        logging.error(traceback.format_exc())
        return False, pd.DataFrame()


    # =============================================================================
    # Ordenação e manutenção da última contagem por chave
    # =============================================================================
    # Ordena os dados e mantém apenas a última contagem para cada chave única.
    logging.info("Ordenando e mantendo apenas a última contagem por chave.")
    try:
        df = df.sort_values(by=["CHAVE", "Contagem", "Atualização"], ascending=[True, False, False])
        df = df.drop_duplicates(subset="CHAVE", keep="first").copy()
        logging.info(f"Dados ordenados e duplicatas por CHAVE removidas. Dimensões: {df.shape}")
    except Exception as e:
        logging.critical(f"Erro durante a ordenação e remoção de duplicatas: {e}. Processamento abortado.")
        logging.error(traceback.format_exc())
        return False, pd.DataFrame()


    # =============================================================================
    # Coluna condicional "Status Controle"
    # =============================================================================
    # Cria a coluna "Status Controle" com base no "Status Inventario".
    logging.info("Criando coluna 'Status Controle'.")
    try:
        df["Status Controle"] = df.apply(
            lambda row: "Custo Evitado" if row["Status Inventario"] == "ANULADO"
            else ("Valor Consultivado" if row["Status Inventario"] == "FECHADO" else "-"),
            axis=1
        )
        logging.info("Coluna 'Status Controle' criada.")
    except Exception as e:
        logging.critical(f"Erro ao criar a coluna 'Status Controle': {e}. Processamento abortado.")
        logging.error(traceback.format_exc())
        return False, pd.DataFrame()


    # =============================================================================
    # Tratamento da coluna Quantidade Fisica
    # =============================================================================
    # Ajusta a coluna "Quantidade Fisica" utilizando o valor de "Física" se disponível.
    logging.info("Tratando coluna 'Quantidade Fisica'.")
    try:
        df["Quantidade Fisica"] = df["Física"].fillna(df["Quantidade Fisica"])
        df = df.drop(columns=["Física"])
        logging.info("Coluna 'Quantidade Fisica' ajustada e 'Física' removida.")
    except Exception as e:
        logging.critical(f"Erro ao tratar a coluna 'Quantidade Fisica': {e}. Processamento abortado.")
        logging.error(traceback.format_exc())
        return False, pd.DataFrame()


    # =============================================================================
    # Cálculo das novas métricas
    # =============================================================================
    # Calcula métricas financeiras e de diferença entre quantidades.
    logging.info("Calculando novas métricas (Valor Logica, Valor Fisico, Vlr Delta, Delta ABS).")
    try:
        # Verifica a existência da coluna "Lógica" antes de calcular "Valor Logica".
        if "Lógica" in df.columns:
            df["Valor Logica"] = df["Custo"] * df["Lógica"]
        else:
            logging.critical("Erro: A coluna 'Lógica' não foi encontrada no DataFrame. Não é possível calcular 'Valor Logica'. Processamento abortado.")
            return False, pd.DataFrame()

        # Calcula o valor físico e as diferenças em valor.
        df["Valor Fisico"] = df["Custo"] * df["Quantidade Fisica"]
        df["Vlr Delta"] = df["Valor Fisico"] - df["Valor Logica"]
        df["Delta ABS"] = df["Vlr Delta"].abs()
        logging.info("Métricas calculadas.")

    except Exception as e:
        logging.critical(f"Erro durante o cálculo das novas métricas: {e}. Processamento abortado.")
        logging.error(traceback.format_exc())
        return False, pd.DataFrame()


    # =============================================================================
    # Formatação das colunas monetárias
    # =============================================================================
    # Arredonda as colunas monetárias para duas casas decimais.
    logging.info("Formatação das colunas monetárias.")
    try:
        colunas_monetarias = ["Custo", "Valor Logica", "Valor Fisico", "Vlr Delta", "Delta ABS"]
        for col in colunas_monetarias:
            if col in df.columns:
                df[col] = df[col].round(2)
        logging.info("Colunas monetárias formatadas para 2 casas decimais.")
    except Exception as e:
        logging.warning(f"Erro ao formatar colunas monetárias: {e}")


    # =============================================================================
    # Remoção de colunas desnecessárias
    # =============================================================================
    # Remove colunas que não são mais necessárias para as etapas subsequentes.
    logging.info("Removendo colunas desnecessárias.")
    try:
        colunas_remover = [
            "Descrição", "Impressão", "Locação", "UdE", "Compartimento", "Cd. Original",
            "Status Contável", "Contagem", "Final", "Status", "Usuario", "Atualização",
            "Status Inventario", "Estante", "CHAVE", "Notas", "Usuário", "Inclusão", "Tipo Vão", "Tipo UdE", "Nr. UDE", "Data Criação", "Data Inicio", "Data Lançamento", "Data Fim", "Data"
        ]
        df = df.drop(columns=[col for col in colunas_remover if col in df.columns])
        logging.info("Colunas desnecessárias removidas.")
    except Exception as e:
        logging.warning(f"Erro ao remover colunas desnecessárias: {e}")


    # =============================================================================
    # Filtrar linhas onde Status Controle é "ANULADO"
    # =============================================================================
    # Remove linhas com status "ANULADO".
    logging.info("Filtrando linhas com 'Status Controle' = 'ANULADO'.")
    try:
        df = df[df["Status Controle"] != "ANULADO"].copy()
        logging.info(f"Linhas com 'Status Controle' = 'ANULADO' removidas. Dimensões: {df.shape}")
    except Exception as e:
        logging.warning(f"Erro ao filtrar linhas com 'Status Controle' = 'ANULADO': {e}")


    # =============================================================================
    # Agrupar por Código, Data INV, Ano e Mês e somar as colunas numéricas
    # =============================================================================
    # Agrupa os dados por item e período para obter métricas agregadas.
    logging.info("Agrupando dados por item, data e período para somar métricas.")
    try:
        colunas_somar = ["Lógica", "Quantidade Fisica", "Valor Logica", "Valor Fisico", "Vlr Delta", "Delta ABS"]
        colunas_somar_existentes = [col for col in colunas_somar if col in df.columns]
        if not colunas_somar_existentes:
             logging.warning("Nenhuma coluna numérica para somar após a remoção. Agrupamento pode não ter o efeito esperado.")
             df_agrupado = df.groupby(["Código", "Data INV", "Ano", "Mês"], as_index=False).first()
        else:
             df_agrupado = df.groupby(["Código", "Data INV", "Ano", "Mês"], as_index=False)[colunas_somar_existentes].sum()

        logging.info(f"Dados agrupados. Dimensões: {df_agrupado.shape}")

    except Exception as e:
        logging.critical(f"Erro durante o agrupamento dos dados: {e}. Processamento abortado.")
        logging.error(traceback.format_exc())
        return False, pd.DataFrame()


    # =============================================================================
    # Exportação para Excel (Opcional - Mantido para compatibilidade interna do pipeline)
    # =============================================================================
    # Exporta o DataFrame processado para um arquivo Excel.
    logging.info(f"Exportando dados tratados para Excel em: {arquivo_destino}")
    try:
        # Cria o diretório de saída se ele não existir.
        output_dir = os.path.dirname(arquivo_destino)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Diretório de saída criado: {output_dir}")

        df_agrupado.to_excel(arquivo_destino, index=False)
        logging.info(f"Dados tratados exportados com sucesso para {arquivo_destino}")
        # Retorna True indicando sucesso e o DataFrame resultante.
        return True, df_agrupado
    except Exception as e:
        logging.critical(f"Erro ao exportar os dados tratados para Excel: {e}. Processamento abortado.")
        logging.error(traceback.format_exc())
        return False, pd.DataFrame()


# =============================================================================
# Módulo 3: Previsão e Planejamento de Inventário
# =============================================================================

def carregar_e_processar_movimentacao(movimentacao_folder_path):
    """
    Carrega e concatena múltiplos arquivos CSV de histórico de movimentação de uma pasta.
    Realiza a limpeza básica dos dados e agrega features de movimentação por item/mês.

    Args:
        movimentacao_folder_path (str): Caminho para a pasta contendo os arquivos CSV de movimentação.

    Returns:
        pd.DataFrame: DataFrame contendo as features de movimentação agregadas por item/mês.
    """
    logging.info(f"Iniciando carregamento e processamento dos arquivos de movimentação da pasta: {movimentacao_folder_path}")
    # Busca todos os arquivos CSV na pasta especificada.
    all_files = glob.glob(os.path.join(movimentacao_folder_path, "*.csv"))

    # Verifica se foram encontrados arquivos.
    if not all_files:
        logging.warning(f"Nenhum arquivo CSV encontrado na pasta de movimentação: {movimentacao_folder_path}")
        return pd.DataFrame()

    li = []
    # Itera sobre cada arquivo encontrado.
    for filename in all_files:
        try:
            # Carrega o arquivo CSV de movimentação.
            df = pd.read_csv(filename, encoding='latin-1', sep=',', low_memory=False)
            li.append(df)
            logging.info(f"Arquivo de movimentação carregado: {filename}")
        except Exception as e:
            logging.error(f"Erro ao carregar arquivo de movimentação {filename}: {e}")
            logging.error(traceback.format_exc())
            continue

    # Verifica se algum DataFrame foi carregado.
    if not li:
        logging.warning("Nenhum DataFrame carregado com sucesso dos arquivos de movimentação.")
        return pd.DataFrame()

    # Concatena todos os DataFrames de movimentação.
    df_mov = pd.concat(li, axis=0, ignore_index=True)
    logging.info(f"Todos os arquivos de movimentação concatenados. Dimensões iniciais: {df_mov.shape}")

    # Define e verifica a presença de colunas essenciais nos dados de movimentação.
    required_mov_cols = ['Ano', 'Mes', 'CD_ARTICOLO', 'Tipo_Movimento', 'Qtd_Movimentos', 'Total_Qtde']
    available_mov_cols = [col for col in required_mov_cols if col in df_mov.columns]
    if len(available_mov_cols) < len(required_mov_cols):
        missing_mov_cols = [col for col in required_mov_cols if col not in df_mov.columns]
        logging.error(f"Colunas essenciais ausentes nos dados de movimentação: {missing_mov_cols}")
        logging.error(f"Colunas encontradas no CSV lido: {df_mov.columns.tolist()}")
        if 'CD_ARTICOLO' not in available_mov_cols:
             logging.error("A coluna 'CD_ARTICOLO' (código do item) não foi encontrada. Processamento de movimentação abortado.")
             return pd.DataFrame()
        required_mov_cols = available_mov_cols

    # Seleciona e renomeia as colunas necessárias.
    df_mov = df_mov[required_mov_cols].copy()
    df_mov.rename(columns={'CD_ARTICOLO': 'Código', 'Mes': 'Mês'}, inplace=True)

    # Padroniza os códigos dos itens.
    df_mov = standardize_codigo(df_mov, 'Código')
    logging.info("Códigos dos itens padronizados nos dados de movimentação.")

    # Remove linhas duplicadas.
    initial_rows = df_mov.shape[0]
    df_mov.drop_duplicates(inplace=True)
    logging.info(f"Removidas {initial_rows - df_mov.shape[0]} linhas duplicadas. Dimensões após remoção: {df_mov.shape}")

    # Converte colunas numéricas e preenche NaNs com zero.
    for col in ['Ano', 'Mês', 'Qtd_Movimentos', 'Total_Qtde']:
        if col in df_mov.columns:
            df_mov[col] = pd.to_numeric(df_mov[col], errors='coerce').fillna(0)

    # Remove linhas com valores nulos em colunas essenciais após conversão.
    df_mov.dropna(subset=['Código', 'Tipo_Movimento'], inplace=True)

    logging.info(f"Dados de movimentação limpos e preparados. Dimensões finais: {df_mov.shape}")

    # --- Agregação de Features de Movimentação ---
    # Agrega as features de movimentação por item e mês.
    logging.info("Agregando features de movimentação por item e mês.")
    try:
        df_mov_features = pd.DataFrame(columns=['Código', 'Ano', 'Mês']) # Inicializa com colunas base

        if 'Tipo_Movimento' in df_mov.columns:
            # Agrega quantidade total e contagem de movimentos por tipo de movimento.
            df_mov_agg_type = df_mov.groupby(['Código', 'Ano', 'Mês', 'Tipo_Movimento']).agg(
                Total_Qtde_Mov=('Total_Qtde', 'sum'),
                Qtd_Mov_Count=('Qtd_Movimentos', 'sum')
            ).reset_index()

            if not df_mov_agg_type.empty:
                # Cria tabelas pivô para transformar tipos de movimento em colunas.
                df_mov_pivot_total_qtde = df_mov_agg_type.pivot_table(
                    index=['Código', 'Ano', 'Mês'],
                    columns='Tipo_Movimento',
                    values='Total_Qtde_Mov',
                    fill_value=0
                )
                df_mov_pivot_qtd_mov = df_mov_agg_type.pivot_table(
                    index=['Código', 'Ano', 'Mês'],
                    columns='Tipo_Movimento',
                    values='Qtd_Mov_Count',
                    fill_value=0
                )

                # Renomeia as colunas pivoteadas.
                df_mov_pivot_total_qtde.columns = [f'{col}_Total_Qtde_Mensal' for col in df_mov_pivot_total_qtde.columns]
                df_mov_pivot_qtd_mov.columns = [f'{col}_Qtd_Movimentos_Mensal' for col in df_mov_pivot_qtd_mov.columns]

                # Mescla as features de quantidade total e contagem de movimentos.
                df_mov_features = df_mov_pivot_total_qtde.reset_index().merge(
                    df_mov_pivot_qtd_mov.reset_index(),
                    on=['Código', 'Ano', 'Mês'],
                    how='outer'
                ).fillna(0)
            else:
                 logging.warning("DataFrame agregado por tipo de movimento está vazio. Features por tipo de movimento não geradas.")

        else:
            logging.warning("Coluna 'Tipo_Movimento' não encontrada. Features por tipo de movimento não serão geradas.")


        # Agrega a quantidade total geral e contagem total geral de movimentos.
        df_mov_total_agg = df_mov.groupby(['Código', 'Ano', 'Mês']).agg(
            Total_Geral_Qtde_Mensal=('Total_Qtde', 'sum'),
            Total_Geral_Qtd_Movimentos_Mensal=('Qtd_Movimentos', 'sum')
        ).reset_index()

        # Calcula o número de tipos de movimento únicos por item/mês.
        if 'Tipo_Movimento' in df_mov.columns:
             df_num_tipos_mov = df_mov.groupby(['Código', 'Ano', 'Mês']).agg(
                 Num_Tipos_Mov_Mensal=('Tipo_Movimento', 'nunique')
             ).reset_index()
             df_mov_total_agg = df_mov_total_agg.merge(df_num_tipos_mov, on=['Código', 'Ano', 'Mês'], how='left').fillna(0)
        else:
             df_mov_total_agg['Num_Tipos_Mov_Mensal'] = 0

        # Mescla as features totais e de número de tipos de movimento com as features por tipo de movimento.
        if not df_mov_features.empty:
             df_mov_features = df_mov_features.merge(
                 df_mov_total_agg,
                 on=['Código', 'Ano', 'Mês'],
                 how='outer'
             ).fillna(0)
        else:
             # Se df_mov_features estava vazio, usa df_mov_total_agg como base.
             df_mov_features = df_mov_total_agg.copy().fillna(0)


        logging.info(f"Features de movimentação agregadas concluídas. Dimensões: {df_mov_features.shape}")

        return df_mov_features

    except Exception as e:
        logging.error(f"Erro durante a agregação das features de movimentação: {e}")
        logging.error(traceback.format_exc())
        return pd.DataFrame()


def feature_engineering(df_inv, df_mov_features, items_with_history_codes):
    """
    Realiza a engenharia de features no DataFrame de histórico de inventário.
    Calcula features temporais e baseadas na Curva ABC, e mescla com features de movimentação.
    Processa apenas itens identificados com histórico relevante para modelagem ML.

    Args:
        df_inv (pd.DataFrame): DataFrame de histórico de inventário processado (com informações da Curva ABC).
        df_mov_features (pd.DataFrame): DataFrame contendo as features de movimentação agregadas.
        items_with_history_codes (set): Conjunto de códigos de itens com histórico relevante.

    Returns:
        pd.DataFrame: DataFrame com as features engenheiradas para modelagem ML.
    """
    logging.info("Iniciando a engenharia de features para itens com histórico.")

    # Define as colunas base esperadas no DataFrame de entrada.
    required_cols_base = ['Código', 'Ano', 'Mês', 'Lógica', 'Quantidade Fisica', 'Data INV', 'Curva_ABC', 'Valor Logica', 'Valor Fisico', 'Vlr Delta', 'Delta ABS']

    # Define uma lista de nomes de colunas de movimentação esperadas para garantir consistência.
    placeholder_mov_cols = [
        'Entrada na Quarentena_Total_Qtde_Mensal', 'Entrada no Estoque_Total_Qtde_Mensal',
        'MovimentaÃ§Ã£o Interna_Total_Qtde_Mensal', 'MovimentaÃ§Ã£o com Quarentena_Total_Qtde_Mensal',
        'SaÃ\xadda da Quarentena_Total_Qtde_Mensal', 'SaÃ\xadda por Venda_Total_Qtde_Mensal',
        'Entrada na Quarentena_Qtd_Movimentos_Mensal', 'Entrada no Estoque_Qtd_Movimentos_Mensal',
        'MovimentaÃ§Ã£o Interna_Qtd_Movimentos_Mensal', 'MovimentaÃ§Ã£o com Quarentena_Qtd_Movimentos_Mensal',
        'SaÃ\xadda da Quarentena_Qtd_Movimentos_Mensal', 'SaÃ\xadda por Venda_Qtd_Movimentos_Mensal',
        'Total_Geral_Qtde_Mensal', 'Total_Geral_Qtd_Movimentos_Mensal', 'Num_Tipos_Mov_Mensal'
    ]
    # Define as colunas de saída esperadas após a engenharia de features.
    expected_output_cols = required_cols_base + [
        'Tendencia_Anual', 'Media_Movel_3M', 'Media_Movel_6M', 'Media_ABC', 'Desvio_ABC'
    ] + placeholder_mov_cols

    # Cria um DataFrame vazio com as colunas de saída esperadas para retornar em caso de falha.
    empty_df_with_expected_cols = pd.DataFrame(columns=expected_output_cols)

    # Verifica se as colunas essenciais estão presentes no DataFrame de entrada.
    missing_essential_cols_input = [col for col in required_cols_base if col not in df_inv.columns]
    if missing_essential_cols_input:
         logging.critical(f"Colunas essenciais ausentes no DataFrame de inventário de entrada para engenharia de features: {missing_essential_cols_input}. Processamento abortado.")
         return empty_df_with_expected_cols

    try:
        # Filtra o DataFrame de inventário para incluir apenas itens com histórico relevante.
        df_inv_filtered_for_fe = df_inv[df_inv['Código'].isin(items_with_history_codes)].copy()

        # Verifica se o DataFrame filtrado não está vazio.
        if df_inv_filtered_for_fe.empty:
             logging.warning("DataFrame de inventário filtrado para engenharia de features está vazio. Nenhum item com histórico encontrado.")
             return empty_df_with_expected_cols

        df_copy = df_inv_filtered_for_fe.copy()

        # Converte Ano e Mês para um formato de data para ordenação temporal.
        df_copy['Data'] = pd.to_datetime(df_copy['Ano'].astype(str) + '-' + df_copy['Mês'].astype(str), errors='coerce')
        # Ordena os dados por item e data.
        df_copy = df_copy.sort_values(['Código', 'Data']).dropna(subset=['Data'])

        # Calcula a tendência anual da quantidade física.
        df_copy['Tendencia_Anual'] = (df_copy.groupby('Código')['Quantidade Fisica']
                                     .pct_change(periods=12)
                                     .replace([np.inf, -np.inf], np.nan)
                                     .fillna(0) * 100)

        # Calcula médias móveis da quantidade física para diferentes janelas de tempo.
        for window in [3, 6]:
            df_copy[f'Media_Movel_{window}M'] = (df_copy.groupby('Código')['Quantidade Fisica']
                                                  .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean()))

        # Calcula a média e o desvio padrão da quantidade física por categoria da Curva ABC.
        if 'Curva_ABC' in df_copy.columns:
            df_copy['Curva_ABC'] = df_copy['Curva_ABC'].fillna('Unknown')
            df_copy['Media_ABC'] = df_copy.groupby('Curva_ABC')['Quantidade Fisica'].transform('mean')
            df_copy['Desvio_ABC'] = df_copy.groupby('Curva_ABC')['Quantidade Fisica'].transform('std')
            # Preenche NaNs nas médias e desvios padrão por categoria com a média e desvio padrão globais.
            global_mean_hist = df_copy['Quantidade Fisica'].mean()
            global_std_hist = df_copy['Quantidade Fisica'].std()
            df_copy['Media_ABC'] = df_copy['Media_ABC'].fillna(global_mean_hist)
            df_copy['Desvio_ABC'] = df_copy['Desvio_ABC'].fillna(global_std_hist)
        else:
            logging.warning("Coluna 'Curva_ABC' não encontrada no DataFrame. Média e Desvio Padrão globais dos itens COM histórico serão utilizados.")
            # Se a coluna Curva_ABC não existe, usa a média e desvio padrão globais para todos os itens.
            global_mean_hist = df_copy['Quantidade Fisica'].mean()
            global_std_hist = df_copy['Quantidade Fisica'].std()
            df_copy['Media_ABC'] = global_mean_hist
            df_copy['Desvio_ABC'] = global_std_hist

        # Remove a coluna de data temporária.
        df_copy = df_copy.drop(columns=['Data'])

        # Filtra as features de movimentação para incluir apenas itens com histórico relevante.
        df_mov_features_filtered = df_mov_features[df_mov_features['Código'].isin(items_with_history_codes)].copy()

        # Mescla as features de inventário com as features de movimentação.
        if not df_mov_features_filtered.empty:
            logging.info("Mesclando features de movimentação com as features de inventário.")
            df_combined = pd.merge(
                df_copy,
                df_mov_features_filtered,
                on=['Código', 'Ano', 'Mês'],
                how='left'
            )
            # Preenche valores nulos nas colunas de movimentação com zero após a mesclagem.
            mov_cols_merged = [col for col in df_mov_features_filtered.columns if col not in ['Código', 'Ano', 'Mês']]
            for col in mov_cols_merged:
                if col in df_combined.columns:
                    df_combined[col] = df_combined[col].fillna(0)
        else:
            logging.warning("DataFrame de features de movimentação filtrado para itens com histórico está vazio. Features de movimentação serão preenchidas com zero.")
            df_combined = df_copy
            # Adiciona colunas de movimentação com valores zero se o DataFrame de movimentação estiver vazio.
            for col in placeholder_mov_cols:
                if col not in df_combined.columns:
                    df_combined[col] = 0.0

        # Preenche valores nulos em features numéricas e categóricas com valores padrão.
        features_to_fillna_zero = [
            'Tendencia_Anual', 'Media_Movel_3M', 'Media_Movel_6M', 'Media_ABC', 'Desvio_ABC',
            'Valor Logica', 'Valor Fisico', 'Vlr Delta', 'Delta ABS'
        ] + placeholder_mov_cols

        for col in features_to_fillna_zero:
            if col in df_combined.columns:
                df_combined[col] = df_combined[col].fillna(0)

        if 'Curva_ABC' in df_combined.columns:
             df_combined['Curva_ABC'] = df_combined['Curva_ABC'].fillna('Unknown')

        # Remove linhas com valores nulos em colunas essenciais após a engenharia de features.
        essential_cols_for_dropna = ['Código', 'Ano', 'Mês', 'Quantidade Fisica']
        df_final = df_combined.dropna(subset=essential_cols_for_dropna).copy()

        logging.info(f"Engenharia de features concluída. Dimensões do DataFrame final: {df_final.shape}")
        return df_final

    except ValueError as e:
        logging.critical(f"Erro durante a engenharia de features: {str(e)}. Processamento abortado.")
        logging.error(traceback.format_exc())
        return empty_df_with_expected_cols
    except Exception as e:
        logging.critical(f"Erro inesperado durante a engenharia de features: {str(e)}. Processamento abortado.")
        logging.error(traceback.format_exc())
        return empty_df_with_expected_cols


class ModelTrainer:
    """
    Classe responsável por preparar os dados para treinamento, treinar e otimizar
    diferentes modelos de Machine Learning, e avaliar seu desempenho.
    """
    def __init__(self):
        # Define os modelos de regressão a serem utilizados e seus espaços de hiperparâmetros para otimização.
        self.models = {
            'XGBoost': (XGBRegressor, {'n_estimators': (100, 500), 'max_depth': (3, 10), 'learning_rate': (0.01, 0.3, 'log-uniform')}),
            'LightGBM': (LGBMRegressor, {
                'n_estimators': (100, 1000),
                'max_depth': (5, 15),
                'num_leaves': (20, 100),
                'min_child_samples': (10, 100),
                'learning_rate': (0.01, 0.3),
                'feature_fraction': (0.7, 1.0),
                'reg_alpha': (0.0, 1.0),
                'reg_lambda': (0.0, 1.0),
                'min_split_gain': (0.0, 0.1)
            }),
            'Linear Regression': (LinearRegression, {'fit_intercept': [True, False]}),
            'Ridge': (Ridge, {'alpha': (0.1, 10, 'log-uniform'), 'max_iter': (1000, 50000)}),
            'Lasso': (Lasso, {'alpha': (0.1, 10, 'log-uniform'), 'max_iter': (1000, 50000)}),
            'Decision Tree': (DecisionTreeRegressor, {'max_depth': (5, 30)})
        }
        self.preprocessor = None
        self.original_features = None
        self.processed_feature_names = None
        self.test_metrics = {}


    def prepare_data(self, df, target_col='Quantidade Fisica'):
        """
        Prepara os dados para treinamento e teste.
        Divide o DataFrame em conjuntos de treino e teste com base no tempo,
        e aplica o pré-processamento (escalonamento e one-hot encoding).

        Args:
            df (pd.DataFrame): DataFrame contendo as features engenheiradas e a variável alvo.
            target_col (str): Nome da coluna que representa a variável alvo (quantidade física).

        Returns:
            tuple: Uma tupla contendo os conjuntos de treino e teste pré-processados (X_train_proc,
                   X_test_proc), e as séries da variável alvo (y_train, y_test). Retorna DataFrames
                   e Séries vazios em caso de falha ou dados insuficientes.
        """
        logging.info("Iniciando preparação dos dados para treinamento e teste.")
        # Reinicializa atributos para garantir um estado limpo antes da preparação.
        self.preprocessor = None
        self.original_features = None
        self.processed_feature_names = None
        self.test_metrics = {}

        # Verifica se a coluna alvo está presente no DataFrame.
        if target_col not in df.columns:
            logging.critical(f"Coluna alvo '{target_col}' não encontrada no DataFrame de entrada para preparação dos dados. Preparação abortada.")
            expected_cols = [col for col in df.columns if col != target_col]
            return pd.DataFrame(columns=expected_cols), pd.DataFrame(columns=expected_cols), pd.Series(dtype='float64'), pd.Series(dtype='float64')

        # Remove linhas com valores nulos na coluna alvo.
        df_cleaned = df.dropna(subset=[target_col]).copy()

        # Verifica se o DataFrame não ficou vazio após remover NaNs.
        if df_cleaned.shape[0] == 0:
            logging.warning("DataFrame vazio após remover NaNs na coluna alvo. Preparação dos dados abortada.")
            cols_for_empty_return = [col for col in df.columns if col != target_col]
            self.original_features = cols_for_empty_return
            return pd.DataFrame(columns=cols_for_empty_return), pd.DataFrame(columns=cols_for_empty_return), pd.Series(dtype='float64'), pd.Series(dtype='float64')

        # Define colunas a serem removidas antes do treinamento.
        cols_to_always_remove = ['Código', 'Mês', target_col, 'Data INV', 'Lógica_ABC', 'Custo_Unitário', 'Valor_Estoque', 'Ano_Previsto', 'Mês_Previsto']

        # Seleciona as features disponíveis para treinamento.
        available_features_for_training = [col for col in df_cleaned.columns if col not in cols_to_always_remove]

        self.original_features = available_features_for_training

        # Verifica se há features disponíveis para treinamento.
        if not self.original_features:
             logging.warning("Nenhuma feature disponível para treinamento após remoção de colunas irrelevantes.")
             cols_for_empty_return = [col for col in df.columns if col not in ['Código', 'Mês', target_col, 'Data INV', 'Lógica_ABC', 'Custo_Unitário', 'Valor_Estoque', 'Ano_Previsto', 'Mês_Previsto']]
             self.original_features = cols_for_empty_return
             return pd.DataFrame(columns=cols_for_empty_return), pd.DataFrame(columns=cols_for_empty_return), pd.Series(dtype='float64'), pd.Series(dtype='float64')

        logging.info(f"Features selecionadas para treinamento ({len(self.original_features)}): {self.original_features}")

        # Separa features (X) e variável alvo (y).
        X = df_cleaned[self.original_features].copy()
        y = df_cleaned[target_col].copy()

        # --- Tratamento de Valores Problemáticos ---
        # Substitui valores infinitos por NaN e preenche NaNs em features.
        logging.info("Verificando e tratando valores problemáticos (NaN, Inf) nas features do conjunto de TREINO.")
        X = X.replace([np.inf, -np.inf], np.nan)

        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if X[col].dtype == 'object' or X[col].dtype == 'category':
                    X[col] = X[col].fillna('Unknown')
                else:
                    X[col] = X[col].fillna(0)

        # Verifica a presença de valores infinitos após a substituição inicial.
        numeric_cols_before_split = X.select_dtypes(include=np.number).columns
        if not numeric_cols_before_split.empty:
            num_inf_values = np.isinf(X[numeric_cols_before_split]).sum().sum()
            if num_inf_values > 0:
                logging.warning(f"Foram detectados {num_inf_values} valores infinitos em features numéricas após a substituição inicial.")

        logging.info(f"Total de valores nulos nas features ANTES da divisão treino/teste (após fillna): {X.isnull().sum().sum()}")

        # --- Verificação de Features Constantes ---
        # Identifica features com desvio padrão zero no conjunto de treino.
        logging.info("Verificando features com desvio padrão zero (constantes) no conjunto de TREINO.")
        constant_features_train = []
        if not numeric_cols_before_split.empty:
            try:
                std_dev = X[numeric_cols_before_split].std()
                constant_features_train = std_dev[std_dev == 0].index.tolist()

                if constant_features_train:
                    logging.warning(f"As seguintes features apresentaram desvio padrão zero no TREINO e podem não ser informativas: {constant_features_train}.")
            except Exception as e:
                logging.warning(f"Erro ao calcular desvio padrão para identificar features constantes no TREINO: {e}")
        else:
            logging.warning("Nenhuma feature numérica no TREINO para verificar desvio padrão.")

        # --- Divisão Temporal Treino/Teste ---
        # Divide os dados em conjuntos de treino e teste com base no último ano disponível.
        anos_disponiveis = sorted(df_cleaned['Ano'].unique())
        if len(anos_disponiveis) < 2:
            logging.warning(f"Menos de 2 anos de dados disponíveis. Divisão temporal treino/teste não aplicada.")
            X_train = X.copy()
            y_train = y.copy()
            X_test = pd.DataFrame(columns=self.original_features)
            y_test = pd.Series(dtype='float64')
        else:
            last_year = anos_disponiveis[-1]
            X_train = X[df_cleaned['Ano'] < last_year].copy()
            y_train = y[df_cleaned['Ano'] < last_year].copy()
            X_test = X[df_cleaned['Ano'] == last_year].copy()
            y_test = y[df_cleaned['Ano'] == last_year].copy()
            logging.info(f"Divisão temporal realizada: Treino (dados anteriores a {last_year}), Teste (dados de {last_year}).")

        logging.info(f"Dimensões dos dados de treino: {X_train.shape}, Dimensões dos dados de teste: {X_test.shape}")

        # Verifica se o conjunto de treino não está vazio.
        if X_train.empty:
            logging.warning("Conjunto de treino vazio. Pré-processamento e treinamento serão pulados.")
            cols_for_empty_return = [col for col in self.original_features]
            return pd.DataFrame(columns=cols_for_empty_return), pd.DataFrame(columns=cols_for_empty_return), pd.Series(dtype='float64'), pd.Series(dtype='float64')

        # --- Análise da Variável Alvo no Treino ---
        logging.info("Análise da distribuição da variável alvo ('Quantidade Fisica') no conjunto de Treino:")
        if len(np.unique(y_train)) > 0:
            logging.info(f"Número de valores únicos na variável alvo: {len(np.unique(y_train))}")
            logging.info(f"Estatísticas descritivas da variável alvo:\n{pd.Series(y_train).describe()}")
            logging.info(f"Contagem de valores zero na variável alvo no treino: {(y_train == 0).sum()}")
        else:
            logging.warning("Conjunto de treino vazio ou variável alvo sem dados para análise.")

        # --- Pré-processamento ---
        # Define o pré-processador para escalar features numéricas e aplicar one-hot encoding em categóricas.
        for col in self.original_features:
            if col in X_train.columns and X_train[col].isnull().sum() > 0:
                if X_train[col].dtype == 'object' or X_train[col].dtype == 'category':
                    X_train[col] = X_train[col].fillna('Unknown')
                else:
                    X_train[col] = X_train[col].fillna(0)
            if not X_test.empty and col in X_test.columns and X_test[col].isnull().sum() > 0:
                if X_test[col].dtype == 'object' or X_test[col].dtype == 'category':
                    X_test[col] = X_test[col].fillna('Unknown')
                else:
                    X_test[col] = X_test[col].fillna(0)


        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
        ct_features = [col for col in self.original_features if col in X_train.columns]

        numerical_features_ordered = [col for col in ct_features if col in numerical_features]
        categorical_features_ordered = [col for col in ct_features if col in categorical_features]

        preprocessor = ColumnTransformer(transformers=[
            ('num', RobustScaler(), numerical_features_ordered),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features_ordered)
        ], remainder='drop', verbose_feature_names_out=True)

        logging.info("Aplicando pré-processamento (fit e transform) nos dados de treino.")
        try:
            # Aplica o pré-processamento nos conjuntos de treino e teste.
            X_train_proc = preprocessor.fit_transform(X_train[ct_features])
            if not X_test.empty:
                 X_test_proc = preprocessor.transform(X_test[ct_features])
            else:
                # Cria um array ou matriz esparsa vazia para o teste se o conjunto de teste estiver vazio.
                num_processed_features = preprocessor.get_feature_names_out(ct_features).shape[0]
                generates_sparse = any(hasattr(t[1], 'sparse_output') and t[1].sparse_output for t in preprocessor.transformers)
                if generates_sparse:
                    from scipy.sparse import csr_matrix
                    X_test_proc = csr_matrix((0, num_processed_features))
                else:
                    X_test_proc = np.empty((0, num_processed_features))

            self.preprocessor = preprocessor

            logging.info("Pré-processamento aplicado com sucesso.")
        except Exception as e:
            logging.critical(f"Erro durante o fit/transform do preprocessor: {e}. Preparação abortada.")
            logging.error(traceback.format_exc())
            cols_for_empty_return = [col for col in self.original_features]
            return pd.DataFrame(columns=cols_for_empty_return), pd.DataFrame(columns=cols_for_empty_return), pd.Series(dtype='float64'), pd.Series(dtype='float64')

        # Tenta obter os nomes das features após o pré-processamento.
        try:
            self.processed_feature_names = list(self.preprocessor.get_feature_names_out())
        except Exception as e:
            logging.warning(f"Não foi possível obter os nomes das features processadas: {e}")
            if hasattr(X_train_proc, 'shape'):
                 self.processed_feature_names = [f'proc_feature_{i}' for i in range(X_train_proc.shape[1])]
            else:
                 self.processed_feature_names = []

        # Cria DataFrames temporários com nomes de colunas para análise de correlação (apenas para logging).
        X_train_proc_df = None
        if (isinstance(X_train_proc, np.ndarray) or issparse(X_train_proc)) and self.processed_feature_names:
            try:
                X_train_proc_dense = X_train_proc.toarray() if issparse(X_train_proc) else X_train_proc
                if self.processed_feature_names and len(self.processed_feature_names) == X_train_proc_dense.shape[1]:
                     X_train_proc_df = pd.DataFrame(X_train_proc_dense, columns=self.processed_feature_names, index=X_train.index)
                     logging.debug("DataFrame de treino processado criado para logs/análise.")
                else:
                     logging.warning(f"Número de nomes ({len(self.processed_feature_names)}) não corresponde ao número de colunas ({X_train_proc_dense.shape[1]}). DataFrame de treino processado NÃO criado com nomes.")
            except Exception as e:
                 logging.warning(f"Erro ao criar X_train_proc_df: {e}")

        X_test_proc_df = None
        if (isinstance(X_test_proc, np.ndarray) or issparse(X_test_proc)) and self.processed_feature_names:
            try:
                X_test_proc_dense = X_test_proc.toarray() if issparse(X_test_proc) else X_test_proc
                if self.processed_feature_names and len(self.processed_feature_names) == X_test_proc_dense.shape[1]:
                     if not X_test.empty:
                         X_test_proc_df = pd.DataFrame(X_test_proc_dense, columns=self.processed_feature_names, index=X_test.index)
                         logging.debug("DataFrame de teste processado criado para logs/analise.")
                     else:
                          X_test_proc_df = pd.DataFrame(columns=self.processed_feature_names)
                          logging.debug("DataFrame de teste processado vazio criado com nomes para logs/análise.")
                else:
                     logging.warning(f"Número de nomes ({len(self.processed_feature_names)}) não corresponde ao número de colunas de teste ({X_test_proc_dense.shape[1]}). DataFrame de teste processado NÃO criado com nomes.")
            except Exception as e:
                 logging.warning(f"Erro ao criar X_test_proc_df: {e}")

        # --- Análise de Correlação (para Logging) ---
        logging.info("Calculando correlação das Features Processadas com a variável alvo ('Quantidade Fisica') no conjunto de Treino.")
        if isinstance(X_train_proc_df, pd.DataFrame) and len(np.unique(y_train)) > 1 and not X_train_proc_df.empty:
            try:
                y_train_series = y_train if isinstance(y_train, pd.Series) else pd.Series(y_train, index=X_train_proc_df.index)
                correlations = X_train_proc_df.corrwith(y_train_series, method='pearson').dropna()

                if not correlations.empty:
                    correlations = correlations[correlations.index != target_col]
                    logging.info(f"Correlações das Features (excluindo NaNs, top 30):\n{correlations.sort_values(ascending=False).head(30)}")
                    logging.info(f"Correlações das Features (excluindo NaNs, bottom 30):\n{correlations.sort_values(ascending=True).head(30)}")
                else:
                    logging.warning("Nenhuma feature com variância suficiente para calcular correlação (após pré-processamento).")

            except Exception as e:
                logging.warning(f"Não foi possível calcular correlação das features processadas: {e}")
                logging.error(traceback.format_exc())

        elif not isinstance(X_train_proc_df, pd.DataFrame):
            logging.warning("X_train_proc_df não é DataFrame. Correlação não calculada.")
        elif len(np.unique(y_train)) <= 1:
            logging.warning("Variável alvo tem apenas um valor único no treino. Correlação não calculada.")
        else:
            logging.warning("X_train_proc_df é vazio. Correlação não calculada.")

        logging.info("Preparação dos dados concluída.")

        return X_train_proc, X_test_proc, y_train, y_test


    def train_models(self, X_train_proc, y_train, cv_splits=5):
        """
        Treina e otimiza os modelos de Machine Learning utilizando BayesSearchCV.
        Aplica validação cruzada temporal para avaliar o desempenho durante o treinamento.

        Args:
            X_train_proc: Features de treino pré-processadas.
            y_train (pd.Series or np.ndarray): Variável alvo de treino.
            cv_splits (int): Número de splits para a validação cruzada temporal.

        Returns:
            dict: Um dicionário contendo os modelos treinados, melhores parâmetros e scores de CV.
        """
        logging.info("Iniciando treinamento e otimização dos modelos com BayesSearchCV.")
        results = {}

        # Verifica se o pré-processador e as features originais foram definidos.
        if self.preprocessor is None or self.original_features is None:
             logging.critical("Pré-processador ou features originais não definidos no Trainer. Treinamento abortado.")
             return {}, None

        # Verifica se há dados de treino suficientes.
        if X_train_proc is None or y_train is None or X_train_proc.shape[0] == 0:
            logging.warning("Dados de treino insuficientes. Treinamento dos modelos pulado.")
            return {}, None

        y_train_for_opt = y_train.values if isinstance(y_train, pd.Series) else y_train

        # Itera sobre cada modelo definido.
        for name, (estimator_class, params_space) in self.models.items():
            logging.info(f"--- Treinando e Otimizando Modelo: {name} ---")
            try:
                # Instancia o estimador.
                if hasattr(estimator_class, 'random_state'):
                    estimator = estimator_class(random_state=42)
                else:
                    estimator = estimator_class()

                X_train_for_opt = X_train_proc
                # Converte dados esparsos para densos se necessário para o estimador.
                if issparse(X_train_for_opt):
                     try:
                         X_train_for_opt = X_train_for_opt.toarray()
                         logging.debug(f"Dados de entrada convertidos para formato denso para treinamento com {name}.")
                     except Exception as conv_e:
                         logging.error(f"Erro ao converter dados de entrada para formato denso para treinamento com {name}: {conv_e}")
                         results[name] = {'model': None, 'params': None, 'score': np.nan}
                         logging.error(f"Treinamento do modelo {name} pulado devido a erro de conversão sparse para dense.")
                         continue

                # --- Otimização de Hiperparâmetros com BayesSearchCV ---
                if not params_space:
                    # Se não há espaço de parâmetros, treina o modelo com parâmetros padrão.
                    logging.warning(f"Espaço de busca de hiperparâmetros vazio para o modelo {name}. Treinando com parâmetros padrão.")
                    model = estimator
                    start_time_model = time.time()
                    model.fit(X_train_for_opt, y_train_for_opt)
                    end_time_model = time.time()
                    results[name] = {
                        'model': model,
                        'params': model.get_params(),
                        'score': np.nan # Score de CV não aplicável aqui
                    }
                    logging.info(f"Treinamento padrão para o modelo {name} concluído em {end_time_model - start_time_model:.2f}s.")
                else:
                    # Configura a validação cruzada temporal.
                    cv = TimeSeriesSplit(n_splits=min(cv_splits, max(2, len(y_train_for_opt) // 2)))

                    # Define o número de iterações para o BayesSearchCV.
                    n_iterations = min(50, max(2, len(y_train_for_opt) // max(1, cv.get_n_splits(X_train_for_opt))))

                    if n_iterations < 50:
                         logging.warning(f"Número de iterações para BayesSearchCV reduzido para {n_iterations} devido ao tamanho do dataset/CV.")

                    # Configura e executa o BayesSearchCV.
                    opt = BayesSearchCV(
                        estimator,
                        params_space,
                        n_iter=n_iterations,
                        cv=cv,
                        scoring='neg_mean_absolute_error', # Utiliza MAE negativo como métrica de otimização
                        random_state=42,
                        n_jobs=-1 # Utiliza todos os núcleos disponíveis
                    )

                    start_opt_time = time.time()
                    opt.fit(X_train_for_opt, y_train_for_opt)
                    end_opt_time = time.time()

                    # O score retornado pelo BayesSearchCV é negativo para MAE, então inverte o sinal.
                    cv_score = -opt.best_score_

                    # Armazena o melhor modelo, melhores parâmetros e score de CV.
                    results[name] = {
                        'model': opt.best_estimator_,
                        'params': opt.best_params_,
                        'score': cv_score
                    }
                    logging.info(f"Otimização para o modelo {name} concluída em {end_opt_time - start_opt_time:.2f}s. Melhor MAE (CV): {cv_score:.4f}")
                    logging.info(f"Melhores parâmetros encontrados: {opt.best_params_}")

                # --- Logging de Importância de Features ou Coeficientes ---
                # Loga a importância das features (para modelos baseados em árvore) ou coeficientes (para modelos lineares).
                if results[name].get('model') is not None and self.processed_feature_names:
                     trained_model = results[name]['model']
                     try:
                         if hasattr(trained_model, 'feature_importances_') and trained_model.feature_importances_ is not None and len(trained_model.feature_importances_) == len(self.processed_feature_names):
                             importance = pd.DataFrame({
                                 'feature': self.processed_feature_names,
                                 'importance': trained_model.feature_importances_
                             }).sort_values('importance', ascending=False)
                             logging.info(f"Importância das Features para o modelo {name} (Top 20):\n{importance.head(20)}")
                         elif hasattr(trained_model, 'coef_') and trained_model.coef_ is not None and len(trained_model.coef_) == len(self.processed_feature_names):
                             coefs = pd.DataFrame({
                                 'feature': self.processed_feature_names,
                                 'coef': trained_model.coef_
                             }).sort_values('coef', key=abs, ascending=False)
                             logging.info(f"Coeficientes do modelo {name} (Top 20 por Magnitude):\n{coefs.head(20)}")
                         else:
                             logging.warning(f"O modelo {name} não possui atributos de importância de feature ou coeficientes válidos para logar.")
                     except Exception as e:
                          logging.warning(f"Não foi possível obter/logar importância de feature/coeficientes para o modelo {name}: {e}")

            except Exception as e:
                logging.critical(f"Erro ao treinar/otimizar o modelo {name}: {str(e)}. Treinamento abortado para este modelo.")
                logging.error(traceback.format_exc())
                results[name] = {'model': None, 'params': None, 'score': np.nan}

        # Retorna os resultados do treinamento para todos os modelos.
        return results

    def evaluate_models(self, models_results, X_test_proc, y_test):
        """
        Avalia o desempenho preditivo dos modelos treinados no conjunto de teste.
        Calcula métricas como MAE, RMSE e R2 para cada modelo.

        Args:
            models_results (dict): Dicionário contendo os resultados do treinamento dos modelos.
            X_test_proc: Features de teste pré-processadas.
            y_test (pd.Series or np.ndarray): Variável alvo de teste.

        Returns:
            tuple: Uma tupla contendo um dicionário com as métricas de avaliação no teste
                   e o nome do modelo com o melhor desempenho (menor MAE) no teste.
        """
        logging.info("Iniciando avaliação dos modelos no conjunto de teste.")
        metrics = {}
        best_test_score = float('inf')
        best_model_name_on_test = None

        # Verifica se há dados de teste suficientes para avaliação.
        if X_test_proc is None or y_test is None or X_test_proc.shape[0] == 0 or y_test.shape[0] == 0:
            logging.warning("Conjunto de teste vazio. Avaliação dos modelos pulada.")
            for name in self.models.keys():
                 metrics[name] = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
            self.test_metrics = metrics
            return metrics, None

        y_test_for_eval = y_test.values if isinstance(y_test, pd.Series) else y_test

        # Itera sobre cada modelo para avaliação.
        for name in self.models.keys():
            model_info = models_results.get(name, {})

            # Verifica se o modelo foi treinado com sucesso.
            if model_info.get('model') is None:
                metrics[name] = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
                logging.warning(f"Modelo {name} não treinado com sucesso ou não encontrado nos resultados do treino. Avaliação pulada.")
                continue

            try:
                logging.info(f"Avaliando modelo: {name}")
                model = model_info['model']

                X_test_for_pred = X_test_proc
                # Converte dados esparsos para densos se necessário para a previsão.
                if issparse(X_test_for_pred):
                     try:
                         X_test_for_pred = X_test_for_pred.toarray()
                         logging.debug(f"Dados de entrada convertidos para formato denso para avaliação com {name}.")
                     except Exception as conv_e:
                         logging.error(f"Erro ao converter dados de entrada para formato denso para avaliação com {name}: {conv_e}")
                         metrics[name] = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
                         logging.error(f"Avaliação do modelo {name} pulada devido a erro de conversão sparse para dense.")
                         continue

                # Gera previsões no conjunto de teste.
                y_pred = model.predict(X_test_for_pred)

                # Calcula métricas de avaliação.
                mae = mean_absolute_error(y_test_for_eval, y_pred)
                mse = mean_squared_error(y_test_for_eval, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test_for_eval, y_pred)

                # Armazena as métricas.
                metrics[name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                }
                logging.info(f"Métricas do modelo {name} no Teste: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

                # Atualiza o melhor modelo com base no menor MAE.
                if mae < best_test_score:
                    best_test_score = mae
                    best_model_name_on_test = name

            except Exception as e:
                logging.error(f"Erro ao avaliar modelo {name} no teste: {e}. Avaliação pulada para este modelo.")
                logging.error(traceback.format_exc())
                metrics[name] = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}

        logging.info("Avaliação dos modelos concluída.")
        self.test_metrics = metrics
        logging.info(f"\nModelo com o melhor desempenho (menor MAE) no conjunto de teste: {best_model_name_on_test}")
        # Retorna as métricas de avaliação e o nome do melhor modelo no teste.
        return metrics, best_model_name_on_test


class InventoryForecaster:
    """
    Classe responsável por gerar as previsões de quantidade para todos os itens elegíveis
    e criar um plano de alocação mensal para inventário cíclico, considerando a capacidade
    histórica e a prioridade dos itens (Curva ABC).
    """
    def __init__(self, all_models_results, best_model_name_on_test, preprocessor, original_features, df_inv_original):
        """
        Inicializa a classe InventoryForecaster.

        Args:
            all_models_results (dict): Dicionário com os resultados do treinamento de todos os modelos.
            best_model_name_on_test (str): Nome do modelo com melhor desempenho no conjunto de teste.
            preprocessor: O objeto ColumnTransformer utilizado para pré-processamento.
            original_features (list): Lista dos nomes das features originais utilizadas no treinamento.
            df_inv_original (pd.DataFrame): DataFrame original com o histórico de inventários processados.
        """
        self.all_models_results = all_models_results
        self.best_model_name = best_model_name_on_test
        self.preprocessor = preprocessor
        self.original_features = original_features
        self.df_inv_original = df_inv_original

        # Filtra os modelos treinados com sucesso para uso na previsão.
        self.models_for_prediction = {
             name: model_info.get('model')
             for name, model_info in self.all_models_results.items()
             if model_info.get('model') is not None
        }

        # Define o melhor modelo para previsão, com fallback se o modelo original não estiver disponível.
        if not self.models_for_prediction:
            logging.warning("Nenhum modelo treinado com sucesso disponível para geração de previsões ML.")
            self.best_model = None
            self.best_model_name = None
        else:
             self.best_model = self.models_for_prediction.get(self.best_model_name)
             if self.best_model is None:
                 available_models = list(self.models_for_prediction.keys())
                 if available_models:
                     logging.warning(f"O modelo identificado como 'melhor' ({self.best_model_name}) não foi treinado com sucesso. Utilizando o primeiro modelo disponível para referência: {available_models[0]}.")
                     self.best_model_name = available_models[0]
                     self.best_model = self.models_for_prediction[self.best_model_name]
                 else:
                      logging.warning("Nenhum modelo treinado com sucesso disponível para ser definido como 'melhor modelo'.")
                      self.best_model_name = None
                      self.best_model = None

        logging.info(f"Modelo de referência para exportação consolidada: {self.best_model_name}")


    def _load_counting_history(self):
        """
        Carrega o histórico de contagem mensal da quantidade Lógica a partir do DataFrame
        de histórico de inventário original para estimar a capacidade de alocação mensal.

        Returns:
            dict: Um dicionário onde as chaves são os números dos meses (1-12) e os valores
                  são as médias históricas da quantidade Lógica contada naquele mês.
        """
        logging.info("Carregando histórico de contagem mensal da quantidade Lógica para planejamento da alocação.")
        try:
            # Verifica se os dados de inventário originais são válidos e contêm as colunas necessárias.
            if self.df_inv_original is None or self.df_inv_original.empty or 'Data INV' not in self.df_inv_original.columns or 'Lógica' not in self.df_inv_original.columns:
                logging.warning("Dados de inventário insuficientes ou inválidos para carregar histórico de contagem. Utilizando valores padrão (zero) para capacidade.")
                return {m: 0 for m in range(1, 13)}

            df_inv_copy = self.df_inv_original.copy()
            # Converte a coluna 'Data INV' para datetime e trata erros.
            df_inv_copy['Data INV'] = pd.to_datetime(df_inv_copy['Data INV'], errors='coerce')
            # Converte a coluna 'Lógica' para numérico e preenche NaNs com zero.
            df_inv_copy['Lógica'] = pd.to_numeric(df_inv_copy['Lógica'], errors='coerce').fillna(0)
            # Remove linhas onde a conversão de data falhou ou a quantidade Lógica é nula.
            df_inv_copy.dropna(subset=['Data INV', 'Lógica'], inplace=True)

            # Verifica se o DataFrame não ficou vazio após a limpeza.
            if df_inv_copy.empty:
                logging.warning("Nenhuma data válida ou quantidade Lógica encontrada no histórico de contagem. Utilizando valores padrão (zero) para capacidade.")
                return {m: 0 for m in range(1, 13)}

            # Extrai o mês e ano da data.
            df_inv_copy['Mês'] = df_inv_copy['Data INV'].dt.month
            df_inv_copy['Ano'] = df_inv_copy['Data INV'].dt.year

            # Calcula a soma mensal da quantidade Lógica para cada ano/mês.
            monthly_logic_sums = df_inv_copy.groupby(['Ano', 'Mês'])['Lógica'].sum().reset_index()

            # Verifica se foram calculadas somas mensais.
            if monthly_logic_sums.empty:
                logging.warning("Nenhuma soma mensal de quantidade Lógica calculada. Utilizando valores padrão (zero) para capacidade.")
                return {m: 0 for m in range(1, 13)}

            # Calcula a média da soma mensal da quantidade Lógica para cada mês ao longo dos anos.
            monthly_avg_logic = monthly_logic_sums.groupby('Mês')['Lógica'].mean()
            # Cria um dicionário com a média para cada mês (1 a 12), preenchendo com zero se não houver dados para um mês.
            monthly_avg_logic_dict = {mes: monthly_avg_logic.get(mes, 0) for mes in range(1, 13)}
            logging.info("Média histórica mensal da quantidade Lógica calculada para estimativa de capacidade.")
            return monthly_avg_logic_dict

        except Exception as e:
            logging.error(f"Erro ao carregar histórico de contagem para alocação: {e}")
            logging.error(traceback.format_exc())
            logging.warning("Erro durante o carregamento do histórico de contagem. Utilizando valores padrão para capacidade mensal.")
            # Retorna valores padrão em caso de erro.
            return {
                1: 532106, 2: 630700, 3: 1156641, 4: 777805,
                5: 1030484, 6: 996625, 7: 1536120, 8: 2485435,
                9: 980246, 10: 637624, 11: 589063, 12: 702769
            }


    def generate_predictions_and_allocate(self, df_abc, df_inv_historico_features, items_without_history_df, current_year, current_month):
        """
        Gera previsões de quantidade para todos os itens elegíveis (ML para itens com histórico,
        Lógica_ABC para itens sem histórico) e desenvolve um plano de alocação mensal para
        inventário cíclico, considerando a capacidade histórica e a prioridade dos itens (Curva ABC).

        Args:
            df_abc (pd.DataFrame): DataFrame contendo os resultados da Curva ABC.
            df_inv_historico_features (pd.DataFrame): DataFrame com as features históricas engenheiradas para itens com histórico.
            items_without_history_df (pd.DataFrame): DataFrame contendo informações dos itens sem histórico.
            current_year (int): Ano atual de execução do script.
            current_month (int): Mês atual de execução do script.

        Returns:
            tuple: Uma tupla contendo:
                   - df_allocated_predictions_final (pd.DataFrame): DataFrame final com previsões e plano de alocação.
                   - df_monthly_consolidated (pd.DataFrame): DataFrame vazio (consolidação mensal desabilitada).
                   - plano_inventario_resumo (dict): Dicionário com o resumo da alocação mensal.
        """
        logging.info("Iniciando a geração de previsões e o planejamento da alocação mensal.")

        # Filtra itens elegíveis para inventário cíclico (Lógica_ABC > 0) com base na Curva ABC.
        df_elegiveis_total = df_abc[df_abc['Lógica_ABC'] > 0].copy()
        logging.info(f"Total de itens elegíveis para inventário cíclico: {len(df_elegiveis_total)}")

        # --- Separação de Itens com e sem Histórico ---
        # Identifica itens elegíveis que possuem histórico relevante para modelagem ML e aqueles que não possuem.
        logging.info(f"Separando itens elegíveis com e sem histórico para previsão.")
        items_without_history_codes = set(items_without_history_df['Código'].unique())

        df_elegiveis_with_history = df_elegiveis_total[~df_elegiveis_total['Código'].isin(items_without_history_codes)].copy()
        logging.info(f"Itens elegíveis COM histórico para previsão ML: {len(df_elegiveis_with_history)}")

        df_elegiveis_without_history = df_elegiveis_total[df_elegiveis_total['Código'].isin(items_without_history_codes)].copy()
        logging.info(f"Itens elegíveis SEM histórico para previsão Lógica_ABC: {len(df_elegiveis_without_history)}")

        # Verifica se a soma dos itens com e sem histórico corresponde ao total de elegíveis.
        if len(df_elegiveis_with_history) + len(df_elegiveis_without_history) != len(df_elegiveis_total):
             logging.warning("A contagem de itens elegíveis com/sem histórico não corresponde ao total de elegíveis.")

        # --- Geração de Previsões ---
        df_predictions_with_history = pd.DataFrame()
        # Gera previsões utilizando modelos ML para itens com histórico.
        if not df_elegiveis_with_history.empty and self.preprocessor is not None and self.models_for_prediction and not df_inv_historico_features.empty:
            logging.info("Preparando dados para previsão ML (itens com histórico).")

            # Seleciona as features mais recentes para cada item com histórico.
            last_calculated_features = df_inv_historico_features.sort_values(['Ano', 'Mês']).groupby('Código').tail(1).copy()

            # Seleciona as colunas de features originais presentes nas features calculadas.
            cols_to_select_from_hist = ['Código'] + [col for col in self.original_features if col in last_calculated_features.columns]

            last_calculated_features_subset = last_calculated_features[cols_to_select_from_hist].copy()

            df_previsao_base_with_history = last_calculated_features_subset.copy()

            # Define a lista de features esperadas para a previsão ML.
            X_predict_ml_features_list = [col for col in self.original_features if col in df_previsao_base_with_history.columns]

            # Verifica e adiciona colunas faltantes com valores padrão para garantir a compatibilidade com o pré-processador.
            missing_cols_in_previsao_ml = [col for col in self.original_features if col not in df_previsao_base_with_history.columns]
            if missing_cols_in_previsao_ml:
                 logging.critical(f"Colunas de features esperadas para previsão ML ausentes: {missing_cols_in_previsao_ml}. Geração de previsão ML abortada.")
                 logging.warning(f"Adicionando colunas faltantes {missing_cols_in_previsao_ml} a df_previsao_base_with_history com valores padrão (0/'Unknown').")
                 for col in missing_cols_in_previsao_ml:
                      if col in df_inv_historico_features.columns:
                           if df_inv_historico_features[col].dtype == 'object' or df_inv_historico_features[col].dtype == 'category':
                               df_previsao_base_with_history[col] = 'Unknown'
                           else:
                               df_previsao_base_with_history[col] = 0.0
                      else:
                           df_previsao_base_with_history[col] = 0.0

                 X_predict_ml_features_list = [col for col in self.original_features if col in df_previsao_base_with_history.columns]

            # Seleciona as features para a previsão.
            X_predict_ml_df = df_previsao_base_with_history[X_predict_ml_features_list].copy()

            # Trata valores problemáticos nas features de previsão antes do pré-processamento.
            logging.info("Verificando e tratando valores problemáticos (NaN, Inf) nas features de previsão ML ANTES do pré-processamento.")
            X_predict_ml_df = X_predict_ml_df.replace([np.inf, -np.inf], np.nan)
            for col in X_predict_ml_df.columns:
                 if X_predict_ml_df[col].isnull().sum() > 0:
                      logging.warning(f"Valores nulos detectados na feature '{col}' no conjunto de previsão ML antes do pré-processamento. Preenchendo com 0 ou 'Unknown'.")
                      if X_predict_ml_df[col].dtype == 'object' or X_predict_ml_df[col].dtype == 'category':
                           X_predict_ml_df[col] = X_predict_ml_df[col].fillna('Unknown')
                      else:
                           if col in df_inv_historico_features.select_dtypes(include=np.number).columns:
                                median_val = df_inv_historico_features[col].median()
                                X_predict_ml_df[col] = X_predict_ml_df[col].fillna(median_val if pd.notna(median_val) else 0)
                           else:
                                X_predict_ml_df[col] = X_predict_ml_df[col].fillna(0)

            logging.info(f"Gerando previsões ML para {len(X_predict_ml_df)} itens COM histórico utilizando {len(self.models_for_prediction)} modelos treinados.")

            all_item_model_preds_ml_list = []

            # Itera sobre cada modelo treinado para gerar previsões.
            for nome_modelo, modelo in self.models_for_prediction.items():
                logging.info(f"Gerando previsão com o modelo {nome_modelo} (itens com histórico).")
                try:
                     # Aplica o pré-processamento nos dados de previsão.
                     X_predict_ml_pp = self.preprocessor.transform(X_predict_ml_df)

                     X_predict_ml_for_model = X_predict_ml_pp
                     # Converte dados esparsos para densos se necessário para a previsão.
                     if issparse(X_predict_ml_for_model):
                          try:
                              X_predict_ml_for_model = X_predict_ml_for_model.toarray()
                              logging.debug(f"Dados de entrada convertidos para formato denso para previsão ML com {nome_modelo}.")
                          except Exception as conv_e:
                              logging.error(f"Erro ao converter dados de entrada para formato denso para previsão ML com {nome_modelo}: {conv_e}")
                              logging.error(f"Geração de previsão ML com o modelo {nome_modelo} pulada devido a erro de conversão sparse para dense.")
                              continue

                     # Gera as previsões e garante que não sejam negativas.
                     predictions_ml = np.maximum(0, modelo.predict(X_predict_ml_for_model))

                     # Cria um DataFrame com as previsões para o modelo atual.
                     df_preds_this_model_ml = pd.DataFrame({
                         'Código': df_previsao_base_with_history['Código'].values,
                         'Model_Name': nome_modelo,
                         'Previsao_Quantidade': predictions_ml
                     })
                     all_item_model_preds_ml_list.append(df_preds_this_model_ml)

                except Exception as pred_e:
                     logging.error(f"Erro durante a geração de previsão ML com o modelo {nome_modelo}: {pred_e}")
                     logging.error(traceback.format_exc())
                     # Adiciona previsões como NaN em caso de erro.
                     df_preds_failed_model_ml = pd.DataFrame({
                          'Código': df_previsao_base_with_history['Código'].values,
                          'Model_Name': nome_modelo,
                          'Previsao_Quantidade': np.full(len(df_previsao_base_with_history), np.nan)
                     })
                     all_item_model_preds_ml_list.append(df_preds_failed_model_ml)

            # Concatena as previsões de todos os modelos para itens com histórico.
            if all_item_model_preds_ml_list:
                 df_predictions_with_history = pd.concat(all_item_model_preds_ml_list, ignore_index=True)
                 logging.info(f"Previsões ML por item/modelo (itens com histórico) geradas. Dimensões: {df_predictions_with_history.shape}")
            else:
                 logging.warning("Nenhuma previsão ML por item/modelo gerada para itens com histórico.")
                 df_predictions_with_history = pd.DataFrame()

        df_predictions_without_history = pd.DataFrame()
        # Gera previsões utilizando a Lógica_ABC para itens sem histórico.
        if not df_elegiveis_without_history.empty:
             logging.info("Gerando previsões (Lógica_ABC) para itens SEM histórico.")
             preds_list_without_history = []
             # Utiliza os nomes dos modelos treinados para manter a estrutura, mesmo que a previsão seja baseada apenas na Lógica_ABC.
             model_names_trained = list(self.models_for_prediction.keys())
             if not model_names_trained:
                  model_names_trained = ['Modelo_Padrao_Logica_ABC']
                  logging.warning("Nenhum modelo treinado com sucesso. Utilizando 'Modelo_Padrao_Logica_ABC' para previsões de itens sem histórico.")

             # Cria um DataFrame de previsões para cada nome de modelo, usando a Lógica_ABC como previsão.
             for nome_modelo in model_names_trained:
                  df_preds_this_model_no_hist = df_elegiveis_without_history[['Código', 'Lógica_ABC']].copy()
                  df_preds_this_model_no_hist['Model_Name'] = nome_modelo
                  df_preds_this_model_no_hist.rename(columns={'Lógica_ABC': 'Previsao_Quantidade'}, inplace=True)
                  preds_list_without_history.append(df_preds_this_model_no_hist)

             # Concatena as previsões para itens sem histórico.
             if preds_list_without_history:
                  df_predictions_without_history = pd.concat(preds_list_without_history, ignore_index=True)
                  logging.info(f"Previsões (Lógica_ABC) por item/modelo (itens sem histórico) geradas. Dimensões: {df_predictions_without_history.shape}")
             else:
                  logging.warning("Nenhuma previsão (Lógica_ABC) por item/modelo gerada para itens sem histórico.")
                  df_predictions_without_history = pd.DataFrame()

        # Combina as previsões de itens com e sem histórico.
        df_all_predictions_raw = pd.concat([df_predictions_with_history, df_predictions_without_history], ignore_index=True)
        logging.info(f"Previsões combinadas (ML + Lógica_ABC). Dimensões: {df_all_predictions_raw.shape}")

        # Mescla as previsões com informações base dos itens (Curva ABC, Lógica_ABC, Custo, Valor Estoque).
        base_item_info_cols_from_elegiveis = ['Código', 'Curva_ABC', 'Lógica_ABC', 'Custo_Unitário', 'Valor_Estoque']
        df_all_predictions_with_abc = df_all_predictions_raw.merge(
            df_elegiveis_total[base_item_info_cols_from_elegiveis],
            on='Código',
            how='left'
        )
        # Preenche NaNs nas colunas de informação base com valores padrão.
        for col in base_item_info_cols_from_elegiveis:
             if col in df_all_predictions_with_abc.columns:
                  if df_all_predictions_with_abc[col].isnull().sum() > 0:
                       if df_all_predictions_with_abc[col].dtype == 'object' or df_all_predictions_with_abc[col].dtype == 'category':
                            df_all_predictions_with_abc[col] = df_all_predictions_with_abc[col].fillna('Unknown')
                       else:
                            df_all_predictions_with_abc[col] = pd.to_numeric(df_all_predictions_with_abc[col], errors='coerce').fillna(0)

        # --- Geração do Plano de Alocação Mensal ---
        logging.info("Iniciando a geração do plano de alocação mensal.")
        # Carrega a média histórica mensal da quantidade Lógica para estimar a capacidade.
        media_contagem_mensal_logica = self._load_counting_history()

        # Define o período de planejamento (próximos 12 meses a partir do mês atual).
        next_year = current_year + 1
        meses_restantes_ano_atual = [(current_year, mes) for mes in range(current_month, 13)]
        meses_proximo_ano = [(next_year, mes) for mes in range(1, 13)]
        periodo_planejamento = meses_restantes_ano_atual + meses_proximo_ano
        logging.info(f"Período de planejamento da alocação: {periodo_planejamento}")

        # Identifica itens inventariados no ano corrente para excluí-los do plano de alocação.
        logging.info(f"Identificando itens inventariados no ano corrente ({current_year}) para exclusão do plano de alocação.")
        items_counted_this_year = set()
        try:
            if self.df_inv_original is not None and not self.df_inv_original.empty and 'Data INV' in self.df_inv_original.columns:
                df_inv_current_year = self.df_inv_original.dropna(subset=['Data INV']).copy()
                # Garante que 'Data INV' seja datetime antes de acessar .dt
                df_inv_current_year['Data INV'] = pd.to_datetime(df_inv_current_year['Data INV'], errors='coerce')
                df_inv_current_year.dropna(subset=['Data INV'], inplace=True) # Remove linhas onde a conversão falhou
                df_inv_current_year = df_inv_current_year[df_inv_current_year['Data INV'].dt.year == current_year]
                items_counted_this_year = set(df_inv_current_year['Código'].unique().tolist())
                logging.info(f"{len(items_counted_this_year)} itens identificados como inventariados no ano corrente (excluídos da alocação).")
            else:
                logging.warning("Dados históricos originais insuficientes para identificar itens inventariados no ano corrente. Nenhum item será excluído do plano de alocação por este critério.")
        except Exception as e:
            logging.error(f"Erro ao identificar itens inventariados no ano corrente para exclusão da alocação: {e}")
            logging.error(traceback.format_exc())

        # Filtra os itens elegíveis para incluir apenas aqueles que não foram inventariados no ano corrente.
        df_elegiveis_para_alocacao = df_elegiveis_total[~df_elegiveis_total['Código'].isin(items_counted_this_year)].copy()
        logging.info(f"Total de itens elegíveis para alocação (após exclusão de inventariados no ano corrente): {len(df_elegiveis_para_alocacao)}")

        # Verifica se há itens elegíveis para alocação.
        if df_elegiveis_para_alocacao.empty:
             logging.warning("Nenhum item elegível para alocação após exclusão de inventariados no ano corrente. O plano de inventário ficará vazio.")
             return pd.DataFrame(), pd.DataFrame(), {}

        # Prepara o DataFrame para ordenação por prioridade.
        df_elegiveis_para_alocacao_ordenacao = df_elegiveis_para_alocacao[['Código', 'Curva_ABC', 'Valor_Estoque', 'Lógica_ABC']].copy()
        # Mapeia as categorias da Curva ABC para valores numéricos para ordenação.
        curva_map = {'A': 3, 'B': 2, 'C': 1, 'Unknown': 0}
        df_elegiveis_para_alocacao_ordenacao['Prioridade_Curva'] = df_elegiveis_para_alocacao_ordenacao['Curva_ABC'].map(curva_map).fillna(0)

        # Converte a quantidade Lógica_ABC para numérico e depois para inteiro.
        df_elegiveis_para_alocacao_ordenacao['Lógica_ABC'] = pd.to_numeric(df_elegiveis_para_alocacao_ordenacao['Lógica_ABC'], errors='coerce').fillna(0)
        df_elegiveis_para_alocacao_ordenacao['Lógica_ABC_Int'] = df_elegiveis_para_alocacao_ordenacao['Lógica_ABC'].round().astype(int)

        # Ordena os itens por prioridade (Curva ABC e Valor Estoque).
        df_elegiveis_para_alocacao_ordenacao = df_elegiveis_para_alocacao_ordenacao.sort_values(
            ['Prioridade_Curva', 'Valor_Estoque'],
            ascending=[False, False]
        ).reset_index(drop=True)
        logging.info(f"Itens elegíveis ordenados por prioridade para alocação: {len(df_elegiveis_para_alocacao_ordenacao)} itens.")

        itens_alocados = {}
        plano_inventario_resumo = {}
        # Define um buffer para a capacidade mensal de contagem.
        BUFFER_CAPACIDADE = 0.9
        items_already_allocated_this_run = set()

        # --- Processo de Alocação Mensal ---
        # Aloca os itens nos meses de planejamento com base na capacidade e prioridade.
        logging.info("Iniciando o processo de alocação item a item por mês.")
        allocated_count = 0
        # Itera sobre cada mês no período de planejamento.
        for ano_mes in periodo_planejamento:
            ano_plan, mes_plan = ano_mes
            # Calcula a capacidade alvo de peças Lógica para o mês.
            target_pecas_logica_mes = int(media_contagem_mensal_logica.get(mes_plan, 0) * BUFFER_CAPACIDADE)

            pecas_logica_planejadas_mes = 0
            itens_planejados_mes_codigos = []

            # Itera sobre os itens elegíveis ordenados por prioridade.
            for index, item in df_elegiveis_para_alocacao_ordenacao.iterrows():
                codigo = item['Código']
                qtd_logica_item = item['Lógica_ABC_Int']

                # Pula itens que já foram alocados.
                if codigo in items_already_allocated_this_run:
                    continue

                # Aloca o item se a capacidade do mês não for excedida.
                if pecas_logica_planejadas_mes + qtd_logica_item <= target_pecas_logica_mes:
                    itens_alocados[codigo] = (ano_plan, mes_plan)
                    items_already_allocated_this_run.add(codigo)
                    pecas_logica_planejadas_mes += qtd_logica_item
                    itens_planejados_mes_codigos.append(codigo)
                    allocated_count += 1

            # Registra o resumo da alocação para o mês atual.
            mes_chave_resumo = f'{ano_plan}-{mes_plan:02d}'
            plano_inventario_resumo[mes_chave_resumo] = {
                'Ano': ano_plan,
                'Mês': mes_plan,
                'Itens_Alocados': len(itens_planejados_mes_codigos),
                'Pecas_Logica_Alocada': pecas_logica_planejadas_mes,
                'Capacidade_Pecas_Logica': target_pecas_logica_mes,
            }
            logging.info(f"  {mes_chave_resumo}: {len(itens_planejados_mes_codigos)} itens alocados | {pecas_logica_planejadas_mes:,.0f} peças Lógica (Capacidade: {target_pecas_logica_mes:,.0f}) | Itens restantes para alocar: {len(df_elegiveis_para_alocacao_ordenacao) - len(items_already_allocated_this_run)}")

            # Interrompe a alocação se todos os itens elegíveis forem alocados.
            if len(items_already_allocated_this_run) == len(df_elegiveis_para_alocacao_ordenacao):
                logging.info("Todos os itens elegíveis para alocação foram alocados. Processo de alocação finalizado.")
                break

        logging.info(f"\nTotal de itens alocados no período planejado: {allocated_count}")

        # --- Combinação de Previsões e Alocação ---
        logging.info("Combinando previsões por modelo com os resultados do plano de alocação.")
        # Cria um DataFrame a partir do dicionário de itens alocados.
        alocacao_df = pd.DataFrame.from_dict(itens_alocados, orient='index', columns=['Ano_Previsto', 'Mês_Previsto'])
        alocacao_df.index.name = 'Código'
        alocacao_df.reset_index(inplace=True)

        # Mescla as previsões com as informações de alocação.
        df_allocated_predictions_final = pd.merge(
            df_all_predictions_with_abc,
            alocacao_df,
            on='Código',
            how='inner' # Mantém apenas os itens que foram alocados
        )

        # --- Cálculo de Métricas Previstas e Formatação ---
        # Calcula métricas de valor e imposto com base nas previsões e informações de custo/valor do ABC.
        if not df_allocated_predictions_final.empty:
            # Converte colunas de ano e mês previstos para inteiro.
            df_allocated_predictions_final['Ano_Previsto'] = df_allocated_predictions_final['Ano_Previsto'].astype(int)
            df_allocated_predictions_final['Mês_Previsto'] = df_allocated_predictions_final['Mês_Previsto'].astype(int)
            # Preenche NaNs na Curva_ABC com 'Unknown'.
            if 'Curva_ABC' in df_allocated_predictions_final.columns:
                df_allocated_predictions_final['Curva_ABC'] = df_allocated_predictions_final['Curva_ABC'].fillna('Unknown')

            # Converte colunas numéricas para o tipo apropriado e preenche NaNs com zero.
            for col in ['Lógica_ABC', 'Custo_Unitario', 'Valor_Estoque', 'Previsao_Quantidade']:
                if col in df_allocated_predictions_final.columns:
                    df_allocated_predictions_final[col] = pd.to_numeric(df_allocated_predictions_final[col], errors='coerce').fillna(0)

            # Arredonda a coluna Lógica_ABC para inteiro.
            if 'Lógica_ABC' in df_allocated_predictions_final.columns:
                 df_allocated_predictions_final['Lógica_ABC'] = df_allocated_predictions_final['Lógica_ABC'].round().astype(int)

            # Calcula o valor físico previsto, delta previsto e imposto previsto.
            if 'Custo_Unitario' in df_allocated_predictions_final.columns and 'Previsao_Quantidade' in df_allocated_predictions_final.columns:
                 df_allocated_predictions_final['Valor Fisico Previsto'] = df_allocated_predictions_final['Previsao_Quantidade'] * df_allocated_predictions_final['Custo_Unitario']
                 # O Vlr Delta Previsto é a diferença entre o Valor Físico Previsto e o Valor de Estoque (Valor Lógica) do ABC.
                 df_allocated_predictions_final['Vlr Delta Previsto'] = df_allocated_predictions_final['Valor Fisico Previsto'] - df_allocated_predictions_final['Valor_Estoque']
                 df_allocated_predictions_final['Delta ABS Previsto'] = df_allocated_predictions_final['Vlr Delta Previsto'].abs()
                 # Calcula o imposto previsto (exemplo: 18% sobre o Delta ABS Previsto).
                 df_allocated_predictions_final['TOTAL IMPOSTO Previsto'] = df_allocated_predictions_final['Delta ABS Previsto'] * 0.18

                 # Formata as colunas monetárias previstas para duas casas decimais.
                 cols_prev_monetarias = ['Valor Fisico Previsto', 'Vlr Delta Previsto', 'Delta ABS Previsto', 'TOTAL IMPOSTO Previsto']
                 for col in cols_prev_monetarias:
                      if col in df_allocated_predictions_final.columns:
                           df_allocated_predictions_final[col] = df_allocated_predictions_final[col].round(2)

        logging.info(f"DataFrame final com previsões alocadas criado. Dimensões: {df_allocated_predictions_final.shape}")

        # DataFrame de consolidação mensal é mantido vazio conforme configuração.
        df_monthly_consolidated = pd.DataFrame()
        logging.info("Geração da aba 'Previsao_Mensal' consolidada pulada conforme configuração.")

        # Retorna o DataFrame final alocado, o DataFrame de consolidação mensal (vazio) e o resumo do plano de inventário.
        return df_allocated_predictions_final, df_monthly_consolidated, plano_inventario_resumo


def exportar_resultados_completos(df_resultados_finais_alocados, resultados_modelos_metrics, plano_inventario_resumo_dict, df_previsao_mensal_consolidada_df, df_categorizacao_itens, output_path, df_curva_abc_gerada, df_resultados_processados_gerados):
    """
    Exporta os resultados gerados pelo pipeline completo para um arquivo Excel.
    Inclui as abas de Métricas dos Modelos, Previsões Detalhadas por Modelo e Categorização de Itens.
    As abas Curva ABC, Resultados Mensais Detalhados e Resumo do Plano de Inventário são excluídas.

    Args:
        df_resultados_finais_alocados (pd.DataFrame): DataFrame final com previsões e plano de alocação.
        resultados_modelos_metrics (dict): Dicionário com as métricas de avaliação dos modelos.
        plano_inventario_resumo_dict (dict): Dicionário com o resumo da alocação mensal.
        df_previsao_mensal_consolidada_df (pd.DataFrame): DataFrame de previsão mensal consolidada (vazio).
        df_categorizacao_itens (pd.DataFrame): DataFrame com a categorização dos itens.
        output_path (str): Caminho completo para o arquivo Excel de saída.
        df_curva_abc_gerada (pd.DataFrame): DataFrame da Curva ABC gerada (não será exportado).
        df_resultados_processados_gerados (pd.DataFrame): DataFrame de Resultados Mensais Processados (não será exportado).

    Returns:
        bool: True se a exportação foi bem-sucedida, False caso contrário.
    """
    logging.info("Iniciando a exportação dos resultados para o arquivo Excel.")
    try:
        # Verifica se o caminho de saída foi especificado.
        if not output_path:
            logging.critical("Caminho de saída não especificado. Exportação de resultados pulada.")
            return False

        # Cria o diretório de saída se ele não existir.
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Diretório de saída criado: {output_dir}")

        try:
            # Utiliza pd.ExcelWriter com o engine 'xlsxwriter' para permitir formatação.
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:

                # --- Exportar Métricas de Avaliação dos Modelos ---
                logging.info("Exportando Métricas de Avaliação dos Modelos.")
                # Converte o dicionário de métricas para um DataFrame.
                df_metrics = pd.DataFrame(resultados_modelos_metrics).T
                # Garante que as colunas de métricas existam, preenchendo com NaN se necessário.
                for col in ['MAE', 'RMSE', 'R2']:
                    if col not in df_metrics.columns:
                         df_metrics[col] = np.nan

                # Prepara o DataFrame de métricas para escrita, lidando com o caso de estar vazio.
                if df_metrics.empty:
                     df_metrics_to_write = pd.DataFrame(columns=['Modelo', 'MAE', 'RMSE', 'R2'])
                     logging.warning("DataFrame de métricas vazio. Aba 'Metricas_Modelos' será exportada apenas com cabeçalhos.")
                else:
                     df_metrics_to_write = df_metrics.reset_index().rename(columns={'index': 'Modelo'})
                     logging.info("Métricas de avaliação exportadas com sucesso.")

                # Escreve a aba de Métricas dos Modelos.
                df_metrics_to_write.to_excel(writer, sheet_name='Metricas_Modelos', index=False)


                # --- Exportar previsões detalhadas por modelo ---
                # Verifica se o DataFrame final alocado é válido e contém o nome do modelo.
                if isinstance(df_resultados_finais_alocados, pd.DataFrame) and not df_resultados_finais_alocados.empty and 'Model_Name' in df_resultados_finais_alocados.columns:
                    logging.info("Exportando previsões detalhadas por modelo em abas separadas.")
                    # Define as colunas esperadas no DataFrame final alocado para exportação.
                    expected_cols_in_df_resultados_finais = [
                        'Código', 'Model_Name', 'Ano_Previsto', 'Mês_Previsto', 'Curva_ABC',
                        'Lógica_ABC', 'Custo_Unitario', 'Valor_Estoque', 'Previsao_Quantidade',
                        'Valor Fisico Previsto', 'Vlr Delta Previsto', 'Delta ABS Previsto', 'TOTAL IMPOSTO Previsto'
                    ]

                    # Filtra o DataFrame para incluir apenas as colunas esperadas.
                    df_resultados_finais_filtered_cols = df_resultados_finais_alocados[[col for col in expected_cols_in_df_resultados_finais if col in df_resultados_finais_alocados.columns]].copy()

                    # Obtém a lista de nomes de modelos presentes nos resultados.
                    trained_model_names = df_resultados_finais_filtered_cols['Model_Name'].unique()

                    # Itera sobre cada nome de modelo para criar uma aba específica.
                    for model_name in trained_model_names:
                        logging.info(f"Exportando previsões para o modelo: {model_name}")
                        # Filtra os dados para o modelo atual.
                        df_model_data = df_resultados_finais_filtered_cols[df_resultados_finais_filtered_cols['Model_Name'] == model_name].copy()

                        # Define as colunas a serem exportadas para a aba do modelo (excluindo o nome do modelo).
                        cols_to_export = [col for col in df_model_data.columns if col != 'Model_Name']

                        # Verifica se há dados e colunas para exportar.
                        if 'Previsao_Quantidade' in cols_to_export and cols_to_export:
                            # Cria um nome de aba válido e único para o modelo.
                            sheet_name = f'Prev_{str(model_name)[:26]}'.replace(" ", "_").replace("-","_").replace("(","").replace(")","")
                            sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in ('_'))
                            sheet_name = sheet_name[:31] # Limita o nome da aba a 31 caracteres

                            if not sheet_name: # Fallback para nome de aba se a sanitização resultar em vazio
                                sheet_name = f'Prev_Model_{abs(hash(model_name)) % 1000}'

                            # Ordena os dados por período previsto e código do item.
                            df_model_data = df_model_data.sort_values(by=['Ano_Previsto', 'Mês_Previsto', 'Código'])

                            # Escreve a aba para o modelo atual.
                            df_model_data[cols_to_export].to_excel(writer, sheet_name=sheet_name, index=False)
                            logging.info(f"  Aba '{sheet_name}' exportada com {len(df_model_data)} itens alocados.")
                        else:
                            logging.warning(f"Nenhuma coluna relevante encontrada para exportar para o modelo '{model_name}'. Aba '{sheet_name}' não criada/exportada.")

                elif isinstance(df_resultados_finais_alocados, pd.DataFrame):
                    logging.warning("DataFrame de resultados finais vazio ou sem a coluna 'Model_Name'. Nenhuma aba detalhada por modelo será criada.")
                else:
                    logging.warning("Formato inválido para o DataFrame de resultados finais. Nenhuma aba detalhada por modelo será criada.")


                # --- Exporta a aba de Categorização de Itens ---
                # Verifica se o DataFrame de categorização é válido e não está vazio.
                if isinstance(df_categorizacao_itens, pd.DataFrame) and not df_categorizacao_itens.empty:
                    logging.info("Exportando a aba 'Categorizacao_Itens'.")
                    # Define as colunas esperadas na aba de categorização.
                    categorizacao_cols = ['Código', 'Inventario', 'Movimentacao', 'Curva_ABC', 'Lógica_ABC']
                    # Filtra o DataFrame para incluir apenas as colunas esperadas.
                    cols_to_export_cat = [col for col in df_categorizacao_itens.columns if col in categorizacao_cols]

                    # Verifica se há colunas para exportar.
                    if cols_to_export_cat:
                         # Escreve a aba de Categorização de Itens.
                         df_categorizacao_itens[cols_to_export_cat].to_excel(writer, sheet_name='Categorizacao_Itens', index=False)
                         logging.info("Aba 'Categorizacao_Itens' exportada com sucesso.")
                    else:
                         logging.warning("DataFrame de categorização não contém as colunas esperadas. Aba 'Categorizacao_Itens' não exportada.")

                elif isinstance(df_categorizacao_itens, pd.DataFrame):
                    logging.warning("DataFrame de categorização de itens vazio. Aba 'Categorizacao_Itens' não exportada.")
                else:
                    logging.warning("Formato inválido para o DataFrame de categorização de itens. Aba 'Categorizacao_Itens' não exportada.")

                # --- Abas Excluídas ---
                # As abas 'CURVA_ABC', 'Resultados_Mensais_Detalhados' e 'Resumo_Plano_Inventario'
                # foram removidas da exportação conforme solicitação do usuário.
                logging.info("Abas 'CURVA_ABC', 'Resultados_Mensais_Detalhados' e 'Resumo_Plano_Inventario' excluídas da exportação final.")


            logging.info(f"Processo de escrita no arquivo Excel concluído: {output_path}")
            return True

        except Exception as e:
             logging.critical(f"Erro durante a escrita do arquivo Excel {output_path}: {str(e)}. Exportação abortada.")
             logging.error(traceback.format_exc())
             # Em caso de erro na escrita, avisa que o arquivo pode estar incompleto.
             if os.path.exists(output_path):
                  logging.warning(f"O arquivo Excel de saída ({output_path}) pode estar incompleto devido a um erro de escrita. Recomenda-se verificar ou excluí-lo.")
             return False

    except Exception as e:
        logging.critical(f"Erro inesperado antes do processo de escrita do arquivo de resultados: {str(e)}. Exportação abortada.")
        logging.error(traceback.format_exc())
        return False

# Função para verificar a existência dos arquivos de entrada essenciais
def verify_files(paths_dict):
    """
    Verifica se os arquivos de entrada essenciais existem nos caminhos especificados.
    Levanta FileNotFoundError se algum arquivo ou pasta estiver faltando.

    Args:
        paths_dict (dict): Dicionário contendo os caminhos para os arquivos e pastas de entrada.

    Raises:
        FileNotFoundError: Se algum arquivo ou pasta essencial não for encontrado.
    """
    logging.info("Verificando a existência dos arquivos de entrada essenciais.")
    # Lista dos arquivos CSV e Excel essenciais.
    required_files = [
        paths_dict['estoque_atual_csv'],
        paths_dict['base_custos_xlsx'],
        paths_dict['inventarios_fechados_historico_csv'],
        paths_dict['inventarios_fechados_atual_csv']
    ]
    # Lista das pastas essenciais.
    required_folders = [
        paths_dict['movimentacao_folder']
    ]

    # Verifica quais arquivos e pastas estão faltando.
    missing_files = [f for f in required_files if not Path(f).exists()]
    missing_folders = [f for f in required_folders if not Path(f).is_dir()]

    # Se houver arquivos ou pastas faltando, registra um erro crítico e levanta uma exceção.
    if missing_files or missing_folders:
        error_message = "Arquivos ou pastas essenciais faltando:"
        if missing_files:
            error_message += f"\n  Arquivos: {', '.join(missing_files)}"
        if missing_folders:
            error_message += f"\n  Pastas: {', '.join(missing_folders)}"

        logging.critical(error_message)
        # Exibe um print adicional para garantir que o erro seja visível no console.
        print(f"\n!!! ERRO: {error_message}\n")
        raise FileNotFoundError(error_message)

    logging.info("Todos os arquivos e pastas essenciais de entrada encontrados.")


# =============================================================================
# Função Principal do Pipeline
# =============================================================================

def main_pipeline():
    """
    Função principal que orquestra a execução sequencial dos módulos do pipeline:
    1. Geração da Curva ABC.
    2. Processamento de Inventários Fechados.
    3. Previsão e Planejamento de Inventário.
    """
    # Registra o início da execução do pipeline.
    logging.info("Iniciando o pipeline completo de previsão e planejamento de inventário.")

    # Obtém a data e hora atuais para fins de planejamento e log.
    start_time = time.time()
    now = datetime.now()
    current_year = now.year
    current_month = now.month
    logging.info(f"Data de execução: Ano {current_year}, Mês {current_month}")

    # Inicializa variáveis para armazenar resultados intermediários e finais dos módulos.
    df_curva_abc_gerada = pd.DataFrame()
    df_resultados_processados_gerados = pd.DataFrame()
    df_inventarios_combinados = pd.DataFrame()
    df_inv_original = pd.DataFrame()
    df_abc = pd.DataFrame()
    df_mov_features = pd.DataFrame()
    df_inv_historico_features = pd.DataFrame()
    items_without_history_df = pd.DataFrame()
    df_categorizacao_itens = pd.DataFrame()
    trainer = None
    X_train_proc = None
    X_test_proc = None
    y_train = pd.Series(dtype='float64')
    y_test = pd.Series(dtype='float64')
    models_results = {}
    metrics_on_test = {}
    best_model_name_on_test = None
    df_allocated_predictions_final = pd.DataFrame()
    df_monthly_consolidated = pd.DataFrame()
    plano_inventario_resumo = {}

    # Variável para rastrear o último arquivo/pasta processado em caso de erro fatal.
    last_processed_step = "Inicialização do Pipeline"

    try:
        # --- Módulo de Validação de Arquivos de Entrada ---
        last_processed_step = "Validação de Arquivos de Entrada"
        logging.info("\n--- [Módulo de Validação] Verificando arquivos de entrada ---")
        verify_files(PATHS)
        logging.info("Validação de arquivos de entrada concluída com sucesso.")


        # --- Módulo 1: Geração da Curva ABC ---
        last_processed_step = "Módulo 1: Geração da Curva ABC"
        logging.info("\n--- [Módulo 1/3] Executando: Geração da Curva ABC ---")
        curva_abc_success, df_curva_abc_gerada = generate_curva_abc(
            PATHS['estoque_atual_csv'],
            PATHS['base_custos_xlsx'],
            PATHS['curva_abc_xlsx']
        )

        # Verifica se a geração da Curva ABC foi bem-sucedida.
        if not curva_abc_success:
            logging.critical("Módulo 1/3 falhou: Geração da Curva ABC não foi concluída com sucesso. Pipeline abortado.")
            # Tenta exportar os resultados parciais em caso de falha.
            exportar_resultados_completos(
                df_allocated_predictions_final, metrics_on_test, plano_inventario_resumo,
                df_monthly_consolidated, df_categorizacao_itens, PATHS['previsao_inventario_final_xlsx'],
                df_curva_abc_gerada, df_resultados_processados_gerados
            )
            return # Aborta a execução do pipeline

        # Renomeia colunas do DataFrame da Curva ABC para consistência interna do pipeline.
        logging.info("Renomeando colunas do DataFrame da Curva ABC para consistência interna.")
        df_abc = df_curva_abc_gerada.rename(columns={
            'Produto': 'Código',
            'Descr. Produto': 'Descrição',
            'Quantidade_Locações': 'Qtde_Locacao',
            'Total_Estoqueada': 'Lógica_ABC',
            'Custo_Unitário': 'Custo_Unitário',
            'Curva ABC': 'Curva_ABC',
            'Valor_Estoque': 'Valor_Estoque'
        }).copy()

        # Converte colunas numéricas no DataFrame da Curva ABC.
        cols_to_numeric_abc_loaded = ['Lógica_ABC', 'Custo_Unitário', 'Valor_Estoque', 'Qtde_Locacao']
        for col in cols_to_numeric_abc_loaded:
            if col in df_abc.columns:
                df_abc[col] = pd.to_numeric(df_abc[col], errors='coerce').fillna(0)
        df_abc = standardize_codigo(df_abc, 'Código')
        logging.info("Curva ABC gerada formatada e padronizada para uso no pipeline.")


        # --- Módulo 2: Processamento de Inventários Fechados ---
        last_processed_step = "Módulo 2: Processamento de Inventários Fechados (Carregamento e Combinação)"
        logging.info("\n--- [Módulo 2/3] Executando: Processamento de Inventários Fechados ---")

        # Carrega e combina os arquivos de histórico de inventário.
        inventory_history_files = [
            PATHS['inventarios_fechados_historico_csv'],
            PATHS['inventarios_fechados_atual_csv']
        ]
        df_inventarios_combinados = load_and_combine_inventory_history(inventory_history_files)

        # Verifica se o carregamento e combinação foram bem-sucedidos.
        if df_inventarios_combinados.empty:
             logging.critical("Módulo 2/3 falhou: Nenhum dado de histórico de inventário carregado e combinado. Processamento abortado.")
             exportar_resultados_completos(
                 df_allocated_predictions_final, metrics_on_test, plano_inventario_resumo,
                 df_monthly_consolidated, df_categorizacao_itens, PATHS['previsao_inventario_final_xlsx'],
                 df_curva_abc_gerada, df_resultados_processados_gerados
             )
             return # Aborta a execução

        # Processa o DataFrame combinado de inventários fechados.
        last_processed_step = "Módulo 2: Processamento de Inventários Fechados (Limpeza e Agregação)"
        resultados_processados_success, df_resultados_processados_gerados = process_inventarios_fechados(
            df_inventarios_combinados,
            PATHS['resultados_mensais_detalhados_xlsx']
        )

        # Verifica se o processamento foi bem-sucedido.
        if not resultados_processados_success:
            logging.critical("Módulo 2/3 falhou: Processamento de Inventários Fechados não foi concluído com sucesso. Pipeline abortado.")
            exportar_resultados_completos(
                df_allocated_predictions_final, metrics_on_test, plano_inventario_resumo,
                df_monthly_consolidated, df_categorizacao_itens, PATHS['previsao_inventario_final_xlsx'],
                df_curva_abc_gerada, df_resultados_processados_gerados
            )
            return # Aborta a execução

        # Define o DataFrame de inventário original processado para uso nas próximas etapas.
        df_inv_original = df_resultados_processados_gerados.copy()
        df_inv_original = standardize_codigo(df_inv_original, 'Código')
        logging.info("Resultados Mensais Processados gerados padronizados para uso no pipeline.")


        # --- Módulo 3: Previsão e Planejamento de Inventário ---
        last_processed_step = "Módulo 3: Previsão e Planejamento (Carregamento e Processamento Movimentação)"
        logging.info("\n--- [Módulo 3/3] Executando: Previsão e Planejamento de Inventário ---")

        # Carrega e processa o histórico de movimentação.
        logging.info("Carregando e processando dados de movimentação.")
        df_mov_features = carregar_e_processar_movimentacao(PATHS['movimentacao_folder'])
        # Avisa se nenhum dado de movimentação foi carregado.
        if df_mov_features.empty:
             logging.warning("Nenhum dado de movimentação carregado ou processado com sucesso. Features de movimentação estarão vazias.")

        # Identifica itens com e sem histórico relevante para modelagem.
        last_processed_step = "Módulo 3: Previsão e Planejamento (Identificação Itens com/sem Histórico)"
        logging.info("Identificando itens com e sem histórico para modelagem.")
        df_elegiveis_total_codes = set(df_abc['Código'].unique())
        items_in_inv_history_codes = set(df_inv_original['Código'].unique())
        items_in_mov_history_codes = set(df_mov_features['Código'].unique())

        items_with_history_codes = df_elegiveis_total_codes.intersection(items_in_inv_history_codes.union(items_in_mov_history_codes))
        items_without_history_codes = df_elegiveis_total_codes.difference(items_in_inv_history_codes.union(items_in_mov_history_codes))

        # Cria um DataFrame com informações dos itens sem histórico.
        items_without_history_df = df_abc[df_abc['Código'].isin(items_without_history_codes)].copy()
        # Cria um DataFrame para categorização de itens com base no histórico.
        df_categorizacao_itens = df_abc[df_abc['Lógica_ABC'] > 0][['Código', 'Curva_ABC', 'Lógica_ABC']].copy()
        df_categorizacao_itens['Inventario'] = np.where(
            df_categorizacao_itens['Código'].isin(items_in_inv_history_codes),
            'Já Inventariado',
            'Nunca Inventariado'
        )
        df_categorizacao_itens['Movimentacao'] = np.where(
            df_categorizacao_itens['Código'].isin(items_in_mov_history_codes),
            'Movimentação Recente',
            'Sem Movimentação'
        )
        logging.info("Identificação e categorização de itens concluída.")


        # Mescla Curva ABC com Resultados Mensais Processados para a engenharia de features.
        last_processed_step = "Módulo 3: Previsão e Planejamento (Mesclagem para Feature Engineering)"
        logging.info("Mesclando Curva ABC com Resultados Mensais Processados para engenharia de features.")
        df_inv_with_abc_for_fe = pd.merge(
            df_inv_original,
            df_abc[['Código', 'Curva_ABC', 'Lógica_ABC', 'Custo_Unitário', 'Valor_Estoque']],
            on='Código',
            how='left'
        )
        # Preenche NaNs nas colunas mescladas.
        df_inv_with_abc_for_fe['Curva_ABC'] = df_inv_with_abc_for_fe['Curva_ABC'].fillna('Unknown')
        for col in ['Lógica_ABC', 'Custo_Unitário', 'Valor_Estoque']:
             if col in df_inv_with_abc_for_fe.columns:
                  df_inv_with_abc_for_fe[col] = pd.to_numeric(df_inv_with_abc_for_fe[col], errors='coerce').fillna(0)
        logging.info("Mesclagem para engenharia de features concluída.")


        # Executa a engenharia de features.
        last_processed_step = "Módulo 3: Previsão e Planejamento (Engenharia de Features)"
        logging.info("Executando engenharia de features.")
        df_inv_historico_features = feature_engineering(
            df_inv_with_abc_for_fe,
            df_mov_features,
            items_with_history_codes
        )
        # Avisa se o DataFrame de features históricas estiver vazio.
        if df_inv_historico_features is None or df_inv_historico_features.empty:
            logging.warning("DataFrame de features históricas (para itens COM histórico) vazio. Modelagem ML será pulada.")


        # Prepara os dados para treinamento e teste ML.
        last_processed_step = "Módulo 3: Previsão e Planejamento (Preparação de Dados para ML)"
        logging.info("Preparando dados para treinamento e teste ML.")
        trainer = ModelTrainer()
        X_train_proc, X_test_proc, y_train, y_test = trainer.prepare_data(df_inv_historico_features)

        # --- Treinamento e Avaliação de Modelos ML ---
        last_processed_step = "Módulo 3: Previsão e Planejamento (Treinamento e Avaliação de Modelos ML)"
        # Verifica se há dados de treino suficientes para treinar modelos ML.
        if X_train_proc is None or y_train is None or X_train_proc.shape[0] == 0:
            logging.warning("Dados de treino insuficientes ou problema na preparação. Treinamento e avaliação de modelos ML pulados.")
            models_results = {}
            metrics_on_test = {}
            best_model_name_on_test = None
        else:
             # Verifica se o pré-processador e features originais estão definidos antes do treinamento.
             if trainer.preprocessor is None or trainer.original_features is None:
                  logging.warning("Erro interno: Pré-processador ou features originais não definidos. Treinamento e avaliação podem não executar corretamente.")
                  models_results = {}
                  metrics_on_test = {}
                  best_model_name_on_test = None
             else:
                 logging.info("Executando treinamento de modelos ML.")
                 models_results = trainer.train_models(X_train_proc, y_train)

                 # Verifica se algum modelo foi treinado com sucesso antes de avaliar.
                 if not models_results or all(model_info.get('model') is None for model_info in models_results.values()):
                     logging.warning("Nenhum modelo treinado com sucesso. Avaliação pulada.")
                     metrics_on_test = {}
                     best_model_name_on_test = None
                 else:
                     logging.info("Executando avaliação de modelos ML.")
                     metrics_on_test, best_model_name_on_test = trainer.evaluate_models(models_results, X_test_proc, y_test)


        # --- Geração de Previsões e Plano de Alocação ---
        last_processed_step = "Módulo 3: Previsão e Planejamento (Geração de Previsões e Plano de Alocação)"
        logging.info("Gerando previsões e plano de alocação.")
        # Instancia a classe InventoryForecaster e gera as previsões e o plano de alocação.
        forecaster = InventoryForecaster(models_results, best_model_name_on_test, trainer.preprocessor, trainer.original_features, df_inv_original)

        df_allocated_predictions_final, df_monthly_consolidated, plano_inventario_resumo = forecaster.generate_predictions_and_allocate(
            df_abc,
            df_inv_historico_features,
            items_without_history_df,
            current_year,
            current_month
        )

        logging.info("Processo de previsão e planejamento concluído.")


        # --- Exportação Final de Resultados ---
        last_processed_step = "Exportação Final de Resultados"
        logging.info("\n--- Exportando resultados do pipeline para Excel ---")
        # Exporta os resultados finais para o arquivo Excel.
        if not exportar_resultados_completos(
            df_allocated_predictions_final, metrics_on_test, plano_inventario_resumo,
            df_monthly_consolidated, df_categorizacao_itens, PATHS['previsao_inventario_final_xlsx'],
            df_curva_abc_gerada, df_resultados_processados_gerados
        ):
            logging.critical("Falha crítica durante a exportação final de resultados. Pipeline concluído com erros.")
            return # Aborta a execução

    # --- Tratamento de Exceções ---
    except FileNotFoundError as fnf_error:
        # Captura e trata erros de arquivo não encontrado.
        logging.critical(f"Erro de arquivo não encontrado: {fnf_error}", exc_info=True)
        print(f"\n!!! ERRO: Arquivo ou pasta essencial não encontrado durante a etapa: {last_processed_step} !!!")
        print(f"Detalhes: {fnf_error}")
        print(f"Verifique se os caminhos dos arquivos em PATHS estão corretos e se os arquivos/pastas existem.")
        print(f"Caminho base do script: {BASE_DIR}")
        print(f"Caminhos configurados: {PATHS}")
        print(f"Consulte o arquivo de log para informações completas: inventory_pipeline.log")
        return # Aborta a execução

    except Exception as e:
        # Captura e trata quaisquer outras exceções não tratadas.
        logging.critical(f"Erro fatal não tratado durante a execução do pipeline.", exc_info=True)
        print(f"\n!!! ERRO FATAL INESPERADO !!!")
        print(f"Ocorreu uma falha crítica durante a execução do pipeline na etapa: {last_processed_step}.")
        print(f"Detalhes do erro: {str(e)}")
        print(f"Consulte o arquivo de log para informações completas: inventory_pipeline.log")
        # Tenta exportar os resultados parciais como fallback em caso de erro fatal.
        exportar_resultados_completos(
            df_allocated_predictions_final, metrics_on_test, plano_inventario_resumo,
            df_monthly_consolidated, df_categorizacao_itens, PATHS['previsao_inventario_final_xlsx'],
            df_curva_abc_gerada, df_resultados_processados_gerados
        )
        return # Aborta a execução


    # --- Conclusão do Pipeline (em caso de sucesso) ---
    # Registra o tempo total de execução e informa sobre a conclusão bem-sucedida.
    tempo_exec = time.time() - start_time
    logging.info(f"Pipeline completo finalizado com sucesso em {tempo_exec:.2f} segundos.")
    print(f"\nPipeline completo de previsão e planejamento de inventário concluído com sucesso.")
    print(f"Tempo total de execução: {tempo_exec:.2f} segundos.")
    print(f"Os resultados foram exportados para: {PATHS['previsao_inventario_final_xlsx']}")
    print(f"Verifique o arquivo de log para detalhes: inventory_pipeline.log")


if __name__ == "__main__":
    # Ponto de entrada do script quando executado diretamente.
    print("--- Script de Previsão e Planejamento de Inventário Iniciado ---")
    print("--- Script Finalizado ---")
    main_pipeline()
    