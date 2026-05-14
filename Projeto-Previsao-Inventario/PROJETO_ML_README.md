# 🔮 Projeto: Previsão de Inventário com Machine Learning

**Empresa:** Stellantis Mopar | **Período:** Jul/2023 - Nov/2025 | **Impacto:** R$50k/mês economizados

---

## 📋 RESUMO EXECUTIVO

Sistema automatizado de **previsão de variância de estoque** usando Machine Learning. Identifica erros de inventário antes que ocorram, permitindo pré-contagens automáticas e elimina R$50 mil/mês em perdas.

**Tecnologias:** Python (Pandas, Scikit-Learn, XGBoost, LightGBM), SQL, Power BI

---

## 🎯 PROBLEMA

### **Contexto:**
- 82 mil Part Numbers diferentes
- 252 mil locações de armazenamento
- Compras **100% reativas** ("o que pedimos ontem")

### **Desafios:**
1. **Picos de demanda** → Falta de estoque (stockouts)
2. **Vales de demanda** → Excesso de estoque (overstock)
3. **Sem previsão** → Decisões baseadas em achismo, não dados
4. **Lead time Fornecedores** → 45, 60, 90 dias (sem visibility)

### **Impacto Negativo:**
- Custos altos com compras emergenciais
- Capital bloqueado em estoque desnecessário
- Perda de venda por falta
- Falta de informação para planejamento

---

## 💡 SOLUÇÃO

### **Arquitetura do Sistema**

```
┌─────────────────────────────────────────┐
│  DADOS HISTÓRICOS (24 meses)            │
│  - Consumo por PN                       │
│  - Sazonalidade (meses específicos)     │
│  - Lead time dos fornecedores           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  EXTRAÇÃO & TRANSFORMAÇÃO               │
│  - SQL: BigQuery / Azure                │
│  - Python: Pandas cleaning/feature eng  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  MODELOS TESTADOS                       │
│  - XGBoost (melhor acuracidade)        │
│  - Decision Tree                        │
│  - Linear Regression                    │
│  - Random Forest (instável)             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  OUTPUT: PREVISÃO DE ERRO               │
│  - Variância esperada (±%)              │
│  - Confiança (95%+)                     │
│  - Ação: Se erro > R$5k → Pré-contagem │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Power BI DAILY DASHBOARD               │
│  - Alertas automáticos                  │
│  - Histórico de acuracidade             │
│  - Actionable insights                  │
└─────────────────────────────────────────┘
```

---

## 🔧 IMPLEMENTAÇÃO

### **Fase 1: Análise de Dados (Dive Deep)**

```python
# Extrair 24 meses de histórico
SELECT pn, data, consumo, lead_time
FROM inventario_historico
WHERE data >= DATE_SUB(CURDATE(), INTERVAL 24 MONTH)

# Análise Pandas
df_consumo = pd.read_sql(query, connection)
df_consumo['sazonalidade'] = df_consumo['data'].dt.month
df_consumo['lead_time_dias'] = df_consumo['lead_time'].dt.days
```

**Insights Encontrados:**
- Sazonalidade: Certos meses (ex: fevereiro, junho) têm 40% mais consumo
- Lead time varia de 45-90 dias por fornecedor
- Padrão: Consumo + Sazonalidade + Lead time = Erro Estoque

---

### **Fase 2: Feature Engineering**

```python
# Features criadas
features = {
    'consumo_media_3m': rolling_avg_3_meses,
    'consumo_media_6m': rolling_avg_6_meses,
    'sazonalidade': mes_do_ano,
    'lead_time_dias': dias_entrega_fornecedor,
    'volatilidade': std_dev_consumo,
    'pn_categoria': abc_curve_classification
}

# Target: Erro de Estoque Previsto
y = df['erro_real_estoque']
```

---

### **Fase 3: Modelagem ML**

```python
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

# Teste múltiplos modelos
modelos = {
    'xgboost': XGBRegressor(),
    'decision_tree': DecisionTreeRegressor(),
    'random_forest': RandomForestRegressor()
}

# Validação cruzada
for nome, modelo in modelos.items():
    scores = cross_val_score(modelo, X, y, cv=5, scoring='r2')
    print(f"{nome}: {scores.mean():.3f}")

# VENCEDOR: XGBoost com R² = 0.87
```

---

### **Fase 4: Regra de Decisão**

```python
# Previsão + Confiança
predicao = modelo.predict(features)
confianca = modelo.predict_proba(features)

# Ação automática
if confianca > 0.95 and abs(predicao) > 5000:
    trigger_pre_contagem(pn_id)
    enviar_alerta_warehouse(pn_id, predicao)
else:
    log_historico(pn_id, predicao)
```

---

## 📊 RESULTADOS

### **Performance do Modelo**

| Métrica | Valor |
|---------|-------|
| **R² Score** | 0.87 |
| **MAE (Mean Absolute Error)** | ±R$3.2k |
| **Acuracidade Previsão** | 92% |
| **Tempo Processamento** | <2 segundos |

### **Impacto Operacional**

| Métrica | Antes | Depois | Impacto |
|---------|-------|--------|---------|
| **Identificação de erros** | Manual/reativo | Automático/proativo | 24/7 coverage |
| **Lead time resposta** | 5-7 dias | <24 horas | Ação instantânea |
| **Economia mensal** | — | R$50k (pilot 25%) | Escalável 100% |
| **Acuracidade modelo** | — | 92% | Confiável |

### **Pilot: 25% Part Numbers Críticos (2 meses)**

✅ **R$50 mil em delta líquido reduzido**
✅ **Acuracidade 92% em 5 classes de erro**
✅ **Zero falsos positivos críticos**
✅ **Pronto para scale 100%**

---

## 🚀 COMO USAR

### **1. Instalar Dependências**

```bash
pip install -r requirements.txt
```

**Arquivo `requirements.txt`:**
```
pandas==1.5.3
numpy==1.23.0
scikit-learn==1.2.0
xgboost==1.7.0
lightgbm==3.3.0
sqlalchemy==2.0.0
pyodbc==4.0.32  # Para SQL Server
python-dotenv==1.0.0
```

### **2. Preparar Dados**

Coloque arquivo CSV com histórico:
```
data,pn,consumo,lead_time,estoque_real,estoque_sistema
2024-01-01,PN-001,150,45,500,480
2024-01-01,PN-002,80,60,200,220
...
```

### **3. Executar Modelo**

```bash
python teste_machine.py
```

**Output esperado:**
```
Carregando dados...
Treinando XGBoost...
Validação cruzada: 0.87
Previsões salvas em: predictions.csv
```

### **4. Ver Resultados**

```python
import pandas as pd
resultados = pd.read_csv('predictions.csv')
resultados[['pn', 'erro_previsto', 'confianca', 'acao']].head(10)
```

---

## 📈 PRÓXIMOS PASSOS

### **Curto Prazo (Rollout)**
- [ ] Testar com 50% dos Part Numbers
- [ ] Validar alertas com warehouse
- [ ] Ajustar threshold de R$5k conforme feedback

### **Médio Prazo (Automação Completa)**
- [ ] Integrar com SAP MM (feed automático)
- [ ] Dashboard Power BI conectado ao modelo
- [ ] Alertas no Slack/Email para operadores

### **Longo Prazo (Evolução)**
- [ ] Incluir variáveis externas (sazonalidade de mercado)
- [ ] Previsão de demanda (não só erro)
- [ ] Otimização de quantidade de compra

---

## 📝 NOTAS TÉCNICAS

### **Por que XGBoost?**
- **Vantagens:** Melhor R² (0.87), rápido, lida bem com dados mistos (numéricos + categóricos)
- **Desvantagens:** Menos interpretável (black-box)
- **Alternativa:** Decision Tree mais simples (R² 0.74) para interpretação

### **Por que não Random Forest?**
- Testei, mas ficou instável com lead times longos (>60 dias)
- XGBoost se ajusta melhor à sazonalidade

### **Confidencialidade**
- Dados foram randomizados ±5%
- Part Numbers originais alterados
- Valores permanecem proporcionais para demonstrar método

---

## 👨‍💻 AUTOR

**Elder Jonathan Vieira Pimentel**
- Engenheiro Mecatrônico (UFSJ 2024)
- MBA Data Science & Analytics (USP/ESALQ)
- Black Belt Six Sigma
- Especialista em Otimização de Processos & ML

📧 jhon_elder@hotmail.com | 📱 +55 31 98421-1947

---

**Última atualização:** Maio 2026
