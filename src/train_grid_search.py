# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engine import (
   discretisation,
   encoding
)
from sklearn import (
   model_selection,
   tree,
   naive_bayes,
   linear_model,
   ensemble,
   pipeline,
   
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# %%
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.5f}'.format)
pd.set_option('display.max_colwidth', None)

def df_summary(df):
   if isinstance(df, pd.Series):
      df = df.to_frame()

   total_linhas = len(df)
   qtd_duplicadas = df.duplicated().sum() 

   resumo = pd.DataFrame({
       'coluna': df.columns,
       'dtype': df.dtypes.values,
       'qtdUnique': df.nunique(dropna=True).values,
       'qtdNãoNulos': df.notnull().sum().values,
       'qtdNulos': df.isnull().sum().values,
       'pctNaoNulos': (df.notnull().sum() / total_linhas * 100).round(2).values,
       'pctNulos': (df.isnull().sum() / total_linhas * 100).round(2).values,
   })
   resumo = resumo.sort_values(by='qtdNulos', ascending=False).reset_index(drop=True)  
   print(f"Dimensões do DataFrame: {df.shape}")
   print(f"Quantidade de duplicatas: {qtd_duplicadas}")

   return resumo

# %% SAMPLE
db_path = '../data/abt.csv'
df = pd.read_csv(db_path)

oot_filter = df['dtRef'] == df['dtRef'].max()
df_oot = df[oot_filter].copy()
df_oot['flagChurn'].value_counts(normalize=True)

base_filter = df['dtRef'] != df['dtRef'].max()
df_base = df[base_filter].copy()
df_base['flagChurn'].value_counts(normalize=True)

target = 'flagChurn'
features = df_base.columns[2:-1]
cat_features = df_base[features].select_dtypes(include=['object']).columns.tolist()
num_features = df_base[features].select_dtypes(include=['int', 'float']).columns.tolist()

y = df_base[target]
X = df_base[features]

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, 
    y,
    random_state=42,
    stratify=y,
    test_size=0.2)

# %%
df_train = X_train.copy()
df_train[target] = y_train.copy()

arvore = tree.DecisionTreeClassifier(random_state=42,)
arvore.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'features': features,
    'percent_pct': arvore.feature_importances_*100
}).sort_values(by='percent_pct', ascending=False)

feature_importance['acum_pct'] = feature_importance['percent_pct'].cumsum()
feature_importance.sort_values(by='percent_pct', ascending=False)
# best_features = feature_importance[feature_importance['acum_pct'] <= 80]['features'].tolist()
best_features = feature_importance[feature_importance['percent_pct'] > 0]['features'].tolist()

# %% PIPELINE

# Discretizar
tree_discretization = (
   discretisation.DecisionTreeDiscretiser(
      variables=best_features,
      regression=False,
      bin_output='bin_number',
      cv=3,
   ))

# OneHotEncoding
onehot = (
   encoding.OneHotEncoder(
      variables=best_features,
      ignore_format=True,
   ))

# Modelo
model = (
   ensemble.RandomForestClassifier(
      random_state=42,
      n_jobs=1))

# Hiperparametros
params = {
   'criterion': ['gini', 'entropy', 'log_loss'],
   'n_estimators': [100, 500, 1000],
   'min_samples_leaf': [10, 20, 30],
}

# Grid
grid = model_selection.GridSearchCV(
   model,
   param_grid=params,
   cv=3,
   scoring='roc_auc',
   verbose=3
)

# Pipeline
modelo_pipeline = pipeline.Pipeline(
   steps=[
      ('Discretizar', tree_discretization),
      ('Onehot', onehot),
      # ('Model', model),
      ('Grid', grid),
   ]
)

modelo_pipeline.fit(X_train[best_features], y_train)

# TREINO
y_train_predict = modelo_pipeline.predict(X_train[best_features])
y_train_proba = modelo_pipeline.predict_proba(X_train[best_features])[:,1]
cm_train = confusion_matrix(y_train, y_train_predict)
tn_train, fp_train, _, _ = cm_train.ravel()
acc_train = accuracy_score(y_train, y_train_predict)
precision_train = precision_score(y_train, y_train_predict)
recall_train = recall_score(y_train, y_train_predict)
especificidade_train = tn_train / (tn_train + fp_train)
auc_train = roc_auc_score(y_train, y_train_proba)

# TESTE
y_test_predict = modelo_pipeline.predict(X_test[best_features])
y_test_proba = modelo_pipeline.predict_proba(X_test[best_features])[:,1]
cm_test = confusion_matrix(y_test, y_test_predict)
tn_test, fp_test, _, _ = cm_test.ravel()
acc_test = accuracy_score(y_test, y_test_predict)
precision_test = precision_score(y_test, y_test_predict)
recall_test = recall_score(y_test, y_test_predict)
especificidade_test = tn_test / (tn_test + fp_test)
auc_test = roc_auc_score(y_test, y_test_proba)

# OUT OF TIME
X_oot = df_oot[best_features]
y_oot = df_oot[target]
y_oot_predict = modelo_pipeline.predict(X_oot[best_features])
y_oot_proba = modelo_pipeline.predict_proba(X_oot[best_features])[:,1]
cm_oot = confusion_matrix(y_oot, y_oot_predict)
tn_oot, fp_oot, _, _ = cm_oot.ravel()
acc_oot = accuracy_score(y_oot, y_oot_predict)
precision_oot = precision_score(y_oot, y_oot_predict)
recall_oot = recall_score(y_oot, y_oot_predict)
especificidade_oot = tn_oot / (tn_oot + fp_oot)
auc_oot = roc_auc_score(y_oot, y_oot_proba)

# DATAFRAME
dct_metricas = {
   "Métricas": ["Acurácia", "Precisão", "Recall", "Especificidade", "ROC AUC"],
   "Treino": [acc_train, precision_train, recall_train, especificidade_train, auc_train],
   "Teste": [acc_test, precision_test, recall_test, especificidade_test, auc_test],
   "OOT": [acc_oot, precision_oot, recall_oot, especificidade_oot, auc_oot],
   "Descrição": [
      "Quantas vezes o modelo acertou?",
      "Dos que o modelo disse que são positivos, quantos realmente são?",
      "Dos que realmente são positivos, quantos o modelo acertou?",
      "Dos que realmente são negativos, quantos o modelo acertou?",
      "Capacidade geral do modelo de distinguir entre classes",
   ]
}

df_metricas = pd.DataFrame(dct_metricas)

# %%
print(f'Taxa da variável resposta geral: {y.mean()*100:.2f}%')
print(f'Taxa da variável resposta de treino: {y_train.mean()*100:.2f}%')
print(f'Taxa da variável resposta de test: {y_test.mean()*100:.2f}%')

# ROC CURVES
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
fpr_oot, tpr_oot, _ = roc_curve(y_oot, y_oot_proba)

plt.figure(figsize=(8, 6))

# Curva Train
plt.plot(
   fpr_train, 
   tpr_train, 
   label=f'Train AUC = {auc_train*100:.2f}%', color='blue')
# Curva Test
plt.plot(fpr_test, tpr_test, label=f'Test AUC = {auc_test*100:.2f}%', color='green')
# Curva OOT
plt.plot(fpr_oot, tpr_oot, label=f'OOT AUC = {auc_oot*100:.2f}%', color='orange')

# Linha aleatória
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Configurações do gráfico
plt.title('Curvas ROC: Train vs Test vs OOT')
plt.xlabel('Falso Positivo (1 - Especificidade)')
plt.ylabel('Verdadeiro Positivo (Recall)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
df_metricas # type: ignore

# %%
S_model = pd.Series({
   "model": modelo_pipeline,
   "features": best_features,
})

S_model.to_pickle("../data/S_model.pkl")
