import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import sys
# Manter import de MAE para cálculo, remover RMSE e R2 pois não serão usados
from sklearn.metrics import mean_absolute_error # Removendo mean_squared_error, r2_score
# Remover xlsxwriter import pois não será necessário se a exportação for simplificada
# import xlsxwriter # Removido

# --- Definição dos Caminhos (Completos) ---
# Usando apenas os dois arquivos especificados pelo usuário

PATHS = {
    # Caminho COMPLETO para o arquivo CSV de inventário real do ano atual
    'inventarios_fechados_atual_csv': r'C:\Users\Usuário\Documents\Elder\BI\KPI performence - Rede\Machine Learning\Inventarios+Fechados+Contas\Inventarios+Fechados+Contas.csv', # << AJUSTE ESTE CAMINHO COMPLETO
    # Caminho COMPLETO para o arquivo Excel com as previsões gerado pelo teste_machine
    'previsao_inventario_final_xlsx': r"C:\Users\Usuário\Documents\Elder\BI\KPI performence - Rede\PYTHON\Inventario - teste_machine\Previsao_Inventario_Final_teste_machine.xlsx", # << AJUSTE ESTE CAMINHO COMPLETO
    # Caminho COMPLETO para o arquivo Excel de saída da validação
    'validacao_output_xlsx': r"C:\Users\Usuário\Documents\Elder\BI\KPI performence - Rede\PYTHON\Inventario\Validacao_Inventario__machines.xlsx",
}

def clean_code(df, col='Código'):
    if df is not None and not df.empty and col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lstrip('0').replace('', '0')
    return df

def get_latest_date_from_actual_file(file_path):
    print(f"Buscando data mais recente no arquivo de inventário real: {file_path}")
    all_dates = []

    if os.path.exists(file_path):
        try:
            # Usar low_memory=False para tentar lidar com mixed types
            df = pd.read_csv(file_path, delimiter=";", encoding="latin1", low_memory=False, usecols=['Atualização'])
            df['Atualização_str'] = df['Atualização'].astype(str)
            df['Atualização_dt'] = pd.to_datetime(df['Atualização_str'], format="%d/%m/%Y %H:%M", errors="coerce")
            all_dates.extend(df['Atualização_dt'].dropna().tolist())
            print(f"Datas carregadas de {file_path}")
        except Exception as e:
            print(f"Aviso: Não foi possível carregar datas do arquivo {file_path}: {e}. Verifique o formato da coluna 'Atualização'.")
    else:
        print(f"Erro: Arquivo de inventário real não encontrado em {file_path}.")

    if not all_dates:
        print("Nenhuma data válida encontrada no arquivo de inventário real.")
        now = datetime.now()
        fallback_year = now.year
        fallback_month = now.month - 1
        if fallback_month == 0:
             fallback_month = 12
             fallback_year -= 1
        print(f"Usando fallback de data (mês anterior ao atual do sistema): {fallback_year}-{fallback_month:02d}.")
        return fallback_year, fallback_month

    latest_date = max(all_dates)
    print(f"Data mais recente encontrada: {latest_date.strftime('%d/%m/%Y %H:%M')}")
    return latest_date.year, latest_date.month


def load_actual_ytd(file_path, year, month):
    print(f"\nCarregando dados reais de inventário para o ano {year} até o Mês {month:02d} do arquivo: {file_path}")
    df_actual = pd.DataFrame()

    if os.path.exists(file_path):
        try:
            # Usar low_memory=False para tentar lidar com mixed types
            df = pd.read_csv(file_path, delimiter=";", encoding="latin1", low_memory=False)

            required_cols = ['Código', 'Atualização', 'Quantidade Fisica']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                print(f"Colunas essenciais ausentes no arquivo de inventário real: {missing}. Processamento de dados reais abortado.")
                return pd.DataFrame()

            df = df.dropna(subset=required_cols).copy()
            df = clean_code(df, 'Código')

            df['dt'] = pd.to_datetime(df['Atualização'], format="%d/%m/%Y %H:%M", errors="coerce")
            df = df.dropna(subset=['dt']).copy()
            df['Ano'] = df['dt'].dt.year
            df['Mês'] = df['dt'].dt.month

            ytd = df[(df['Ano'] == year) & (df['Mês'] <= month)].copy()
            ytd.loc[:, 'Quantidade Fisica'] = pd.to_numeric(ytd['Quantidade Fisica'], errors="coerce").fillna(0)

            df_actual_ytd = ytd.groupby('Código')['Quantidade Fisica'].sum().reset_index(name='Real_YTD')

            if not df_actual_ytd.empty:
                print(f"Dados reais YTD (Ano {year} até Mês {month:02d}) carregados para {len(df_actual_ytd)} códigos.")
            else:
                print(f"Nenhum dado real YTD encontrado para o Ano {year} até o Mês {month:02d}.")

            return df_actual_ytd

        except Exception as e:
            print(f"Erro ao carregar ou processar {file_path}: {e}")
            return pd.DataFrame()
    else:
        print(f"Erro: Arquivo de inventário real não encontrado em {file_path}.")
        return pd.DataFrame()


def load_predictions(file_path, year, month):
    print(f"\nCarregando previsões dos modelos para o Ano {year} até o Mês {month:02d} do arquivo: {file_path}")
    predicted_data_by_model = {}
    if not os.path.exists(file_path):
        print(f"Erro: Arquivo Excel de previsão não encontrado em {file_path}. Não é possível carregar previsões.")
        return predicted_data_by_model

    try:
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names

        model_sheets = [sheet for sheet in sheet_names if sheet.startswith('Prev_')]

        if not model_sheets:
            print(f"Nenhuma aba de modelo (iniciando com 'Prev_') encontrada no arquivo {file_path}.")
            return predicted_data_by_model

        for sheet_name in model_sheets:
            print(f"Processando aba de previsão: {sheet_name}")
            try:
                df = xls.parse(sheet_name)

                df = clean_code(df, 'Código')

                required_pred_cols = ['Código', 'Ano_Previsto', 'Mês_Previsto', 'Previsao_Quantidade']
                if not all(col in df.columns for col in required_pred_cols):
                    missing = [col for col in required_pred_cols if col not in df.columns]
                    print(f"Aba '{sheet_name}' não contém colunas essenciais: {missing}. Pulando.")
                    continue

                df['Ano_Previsto'] = pd.to_numeric(df['Ano_Previsto'], errors='coerce').fillna(-1).astype(int)
                df['Mês_Previsto'] = pd.to_numeric(df['Mês_Previsto'], errors='coerce').fillna(-1).astype(int)
                df['Previsao_Quantidade'] = pd.to_numeric(df['Previsao_Quantidade'], errors='coerce').fillna(0)

                df_pred_ytd = df[(df['Ano_Previsto'] == year) & (df['Mês_Previsto'] <= month)].copy()

                agg = df_pred_ytd.groupby('Código')['Previsao_Quantidade'].sum().reset_index()
                model_name = sheet_name.replace('Prev_', '')
                predicted_data_by_model[model_name] = agg.rename(columns={'Previsao_Quantidade': f'Prev_YTD_{model_name}'})

                print(f"  Previsões YTD para o modelo '{model_name}' carregadas ({len(predicted_data_by_model[model_name])} códigos com previsões no período).")

            except Exception as e:
                print(f"Erro ao processar a aba '{sheet_name}': {e}")

    except FileNotFoundError:
        print(f"Erro: Arquivo Excel de previsão não encontrado em {file_path}")
    except Exception as e:
        print(f"Erro geral ao ler o arquivo Excel {file_path}: {e}")

    return predicted_data_by_model


def calculate_metrics(actual, predictions):
    # Inicia o DataFrame de resultados detalhados com os dados reais
    results = actual.copy()

    metrics = {}

    # Itera sobre as previsões de cada modelo
    for model, pred_df in predictions.items():
        print(f"Calculando métricas para o modelo: {model}")
        # Juntar os dados reais com as previsões deste modelo
        merged = results.merge(pred_df, on='Código', how='left')

        real_col = 'Real_YTD'
        forecast_col = f'Prev_YTD_{model}'

        merged[forecast_col] = merged[forecast_col].fillna(0)

        merged[real_col] = pd.to_numeric(merged[real_col], errors='coerce').fillna(0)
        merged[forecast_col] = pd.to_numeric(merged[forecast_col], errors='coerce').fillna(0)

        # Calcula a diferença (Erro: Previsto - Real) e o Erro Absoluto
        merged[f'Erro_{model}'] = merged[forecast_col] - merged[real_col]
        merged[f'Erro_Abs_{model}'] = merged[f'Erro_{model}'].abs()

        # --- Calcular Métricas Agregadas ---
        # MAE
        mae = mean_absolute_error(merged[real_col], merged[forecast_col])

        # Totais
        total_real = merged[real_col].sum()
        total_previsto = merged[forecast_col].sum()
        diferenca_total = total_previsto - total_real

        # WMAPE
        sum_abs_error = merged[f'Erro_Abs_{model}'].sum()
        wmape = (sum_abs_error / total_real) * 100 if total_real != 0 else np.nan

        # MAPE (Calculado apenas para Real > 0)
        merged[f'APE_{model}'] = np.nan
        valid_ape_indices = merged[real_col] > 0
        if valid_ape_indices.any():
            merged.loc[valid_ape_indices, f'APE_{model}'] = (merged.loc[valid_ape_indices, f'Erro_Abs_{model}'] / merged.loc[valid_ape_indices, real_col]) * 100
        mape = merged[f'APE_{model}'].mean()

        # Armazena as métricas no dicionário (apenas as solicitadas)
        metrics[model] = {
            'MAE': mae,
            'WMAPE': wmape,
            'MAPE': mape,
            'Total_Real': total_real,
            'Total_Previsto': total_previsto,
            'Diferenca_Total': diferenca_total
        }

        # Atualiza o DataFrame de resultados detalhados com as novas colunas deste modelo
        cols_to_add_this_model = [
            'Código',
            forecast_col,
            f'Erro_{model}',
            f'Erro_Abs_{model}',
            f'APE_{model}' # Inclui a coluna APE por item
        ]
        cols_to_add_this_model_existing = [col for col in merged.columns if col in cols_to_add_this_model]

        results = results.merge(
            merged[cols_to_add_this_model_existing],
            on='Código',
            how='left'
        )

    return results, metrics


def export_results(detailed_results, metrics_summary, year, month, output_path):
    print(f"\nExportando resultados para o arquivo Excel: {output_path}")
    try:
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:

            # --- Exportar a tabela detalhada para uma aba ---
            detailed_cols_order = ['Código', 'Real_YTD']

            # Adiciona colunas de cada modelo na ordem (Prev, Erro, Erro_Abs, APE)
            model_names_in_results = [col.replace('Prev_YTD_', '') for col in detailed_results.columns if col.startswith('Prev_YTD_')]
            for model_name in model_names_in_results:
                detailed_cols_order.extend([
                    f'Prev_YTD_{model_name}',
                    f'Erro_{model_name}',
                    f'Erro_Abs_{model_name}',
                    f'APE_{model_name}'
                ])
            detailed_cols_order_existing = [col for col in detailed_cols_order if col in detailed_results.columns]
            detailed_results_ordered = detailed_results[detailed_cols_order_existing].copy()

            detailed_results_ordered = detailed_results_ordered.sort_values(by='Real_YTD', ascending=False)

            detailed_sheet_name = f'Detalhado_YTD_Mes{month}'
            detailed_results_ordered.to_excel(writer, sheet_name=detailed_sheet_name, index=False)
            print(f"Aba '{detailed_sheet_name}' (Detalhado YTD) exportada com sucesso.")

            # --- Exportar o resumo das métricas para outra aba ---
            metrics_sheet_name = 'Resumo_Metricas_YTD'
            df_metrics_summary = pd.DataFrame(metrics_summary).T

            # Garante a ordem das colunas na aba de métricas
            metrics_cols_order_for_export = [
                'MAE', 'WMAPE', 'MAPE', 'Total_Real', 'Total_Previsto', 'Diferenca_Total'
            ]
            metrics_cols_order_for_export_existing = [col for col in metrics_cols_order_for_export if col in df_metrics_summary.columns]

            df_metrics_summary[metrics_cols_order_for_export_existing].to_excel(writer, sheet_name=metrics_sheet_name, index=True)
            print(f"Aba '{metrics_sheet_name}' (Resumo Métricas YTD) exportada com sucesso.")

        print(f"\nProcesso de escrita no arquivo Excel concluído: {output_path}")
    except Exception as e:
        print(f"Erro ao exportar resultados completos da validação para Excel: {e}")


if __name__ == "__main__":
    print("--- Script de Validação de Previsão de Inventário (Simplificado) ---")

    # target_year é fixo como 2025 para validar as previsões de 2025
    target_year = 2025
    # last_month é determinado pela data mais recente nos dados reais do arquivo especificado
    latest_year_in_data, last_month = get_latest_date_from_actual_file(PATHS['inventarios_fechados_atual_csv'])

    # Carrega os dados reais YTD para o target_year (2025) até o last_month encontrado
    actual = load_actual_ytd(PATHS['inventarios_fechados_atual_csv'], target_year, last_month)

    # Carrega as previsões YTD para o target_year (2025) até o last_month encontrado
    predictions = load_predictions(PATHS['previsao_inventario_final_xlsx'], target_year, last_month)

    # Verifica se há dados suficientes para realizar a validação
    if actual is None or actual.empty or not predictions:
        print("Dados insuficientes para validação. Verifique os arquivos de entrada e o período.")
        if actual is not None and actual.empty:
            print(" -> Dados reais (actual) estão vazios para o período.")
        if not predictions:
            print(" -> Nenhuma previsão encontrada para o período nos arquivos de previsão.")
    else:
        # Calcula as métricas e obtém os resultados detalhados
        full_results, metrics = calculate_metrics(actual, predictions)

        # Apresenta as métricas no console
        print("\nMétricas de Validação (Resumo):")
        if metrics:
            df_metrics_summary_display = pd.DataFrame(metrics).T
            # Garante a ordem das colunas na exibição do console
            summary_cols_order = [
                'MAE', 'WMAPE', 'MAPE', 'Total_Real', 'Total_Previsto', 'Diferenca_Total'
            ]
            summary_cols_order_existing = [col for col in summary_cols_order if col in df_metrics_summary_display.columns]

            print(df_metrics_summary_display[summary_cols_order_existing].round(2).to_string())

            if 'MAE' in df_metrics_summary_display.columns:
                df_metrics_summary_cleaned = df_metrics_summary_display.dropna(subset=['MAE'])
                if not df_metrics_summary_cleaned.empty:
                    best_model = df_metrics_summary_cleaned['MAE'].idxmin()
                    min_mae = df_metrics_summary_cleaned['MAE'].min()
                    print(f"\nModelo com menor MAE: {best_model} (MAE: {min_mae:.2f})")
                else:
                    print("\nNão foi possível determinar o melhor modelo (todos os MAEs são NaN).")
            else:
                print("\nMétrica 'MAE' não calculada.")

        else:
            print("Nenhum resumo de métricas calculado (verifique erros anteriores).")

        # Exporta os resultados detalhados e o resumo das métricas para o arquivo Excel
        export_results(full_results, metrics, target_year, last_month, PATHS['validacao_output_xlsx'])

    print("\n--- Script de Validação Finalizado ---")
