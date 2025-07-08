# %%
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# %%
def avaliar_modelo(y_obs, y_pred, y_proba, pos_label=1):
   # Métricas básicas
   acc = accuracy_score(y_obs, y_pred)
   prec = precision_score(y_obs, y_pred, pos_label=pos_label)
   recall = recall_score(y_obs, y_pred, pos_label=pos_label)
   auc = roc_auc_score(y_obs, y_proba)

   # Matriz de confusão para calcular especificidade
   cm = confusion_matrix(y_obs, y_pred)
   tn, fp, _, _ = cm.ravel()
   especificidade = tn / (tn + fp)   

   # Matriz de confusão com porcentagens
   cm = confusion_matrix(y_obs, y_pred)
   group_names = ['VN', 'FP', 'FN', 'VP']
   group_counts = [f"{value}" for value in cm.flatten()]
   group_percentages = [f"{value:.1%}" for value in cm.flatten()/cm.sum()]
   labels = [f"{name}\n{count}\n{percent}" for name, count, percent in zip(group_names, group_counts, group_percentages)]
   labels = np.asarray(labels).reshape(2, 2)  
   plt.figure(figsize=(6, 5))
   sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False,
               # xticklabels=['Previsto Negativo', 'Previsto Positivo'],
               # yticklabels=['Real Negativo', 'Real Positivo']
   )
   plt.title("Matriz de Confusão com Percentuais")
   plt.xlabel("Classe prevista")
   plt.ylabel("Classe real")
   plt.tight_layout()
   plt.show() 

   # Mostrar métricas
   print("\nMétricas de Classificação:")
   print(f"Acurácia:        {acc*100:.2f}% (Quantas vezes o modelo acertou)")
   print(f"Precisão:        {prec*100:.2f}% (Dos que o modelo disse que são positivos, quantos realmente são?)")
   print(f"Recall (Sensib): {recall*100:.2f}% (Dos que realmente são positivos, quantos o modelo acertou?)")
   print(f"Especificidade:  {especificidade*100:.2f}% (Dos que realmente são negativos, quantos o modelo acertou?)")
   print(f"AUC ROC:         {auc*100:.2f}% (Capacidade geral do modelo de distinguir entre classes)")
   # Plot da curva ROC

   fpr, tpr, thresholds = roc_curve(y_obs, y_proba, pos_label=pos_label)
   plt.figure(figsize=(7, 5))
   sns.set_style("whitegrid")
   plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc*100:.2f}%)', color='darkorange')
   plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
   plt.xlabel("Taxa de Falsos Positivos (1 - Especificidade)")
   plt.ylabel("Taxa de Verdadeiros Positivos (Recall)")
   plt.title("Curva ROC")
   plt.legend(loc="lower right")
   plt.show()   

   # Retornar dicionário com as métricas
   return {
       "acurácia": acc,
       "precisão": prec,
       "recall": recall,
       "especificidade": especificidade,
       "AUC": auc
}

# %%
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.5f}'.format)

# %%
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

# %%
db_path = '../data/abt.csv'
df = pd.read_csv(db_path)

print("Dados do DataFrame Bruto: ")
print(f"Dimensões: {df.shape}")
print(f"Variáveis: {df.columns.tolist()}")
print(f"Variáveis Categóricas: {df.select_dtypes(include=['object']).columns.tolist()}")
print(f"Variáveis Numéricas: {df.select_dtypes(include=['int', 'float']).columns.tolist()}")
print(f"{pd.DataFrame(df.info())}")
print(f"{pd.DataFrame(df.isna().sum().sort_values(ascending=False))}")
print(f"{pd.DataFrame(df[df.select_dtypes(include=['object']).columns.tolist()].describe().T)}")
print(f"{pd.DataFrame(df[df.select_dtypes(include=['int', 'float']).columns.tolist()].describe().T)}")

# %%
df['flagChurn'].value_counts(normalize=True)

# %%
oot_filter = df['dtRef'] == df['dtRef'].max()
df_oot = df[oot_filter].copy()
df_oot['flagChurn'].value_counts(normalize=True)

# %%
base_filter = df['dtRef'] != df['dtRef'].max()
df_base = df[base_filter].copy()
df_base['flagChurn'].value_counts(normalize=True)

# %%
target = 'flagChurn'
features = df_base.columns[2:-1]
cat_features = df_base[features].select_dtypes(include=['object']).columns.tolist()
num_features = df_base[features].select_dtypes(include=['int', 'float']).columns.tolist()

y = df_base[target]
X = df_base[features]

# %%
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, 
    y,
    random_state=42,
    stratify=y,
    test_size=0.2)

# %%
print(f'Taxa da variável resposta geral: {y.mean()*100:.2f}%')
print(f'Taxa da variável resposta de treino: {y_train.mean()*100:.2f}%')
print(f'Taxa da variável resposta de test: {y_test.mean()*100:.2f}%')

# %%
df_summary(X_train)

# %%
df_train = X_train.copy()
df_train[target] = y_train.copy()
df_summary(df_train)

# %%
train_analysis = df_train.groupby(by=target).agg(['mean', 'median']).T
train_analysis['diff_abs'] = abs(train_analysis[0] - train_analysis[1])
train_analysis['diff_rel'] = abs(train_analysis[0] / train_analysis[1])*100
train_analysis.sort_values(by=['diff_rel'], ascending=False)

# %%
arvore = tree.DecisionTreeClassifier(
   random_state=42,
   # max_depth=5
)

arvore.fit(X_train, y_train)

# %%
feature_importance = pd.DataFrame({
    'features': features,
    'percent_pct': arvore.feature_importances_*100
}).sort_values(by='percent_pct', ascending=False)

feature_importance['acum_pct'] = feature_importance['percent_pct'].cumsum()
feature_importance.sort_values(by='percent_pct', ascending=False)

feature_importance[feature_importance['percent_pct'] > 0]

# %%
features_uteis = feature_importance[feature_importance['acum_pct'] <= 80]['features'].tolist()
matriz_corr = df_train[features_uteis].corr(method='pearson')

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.tight_layout()
plt.show()

# %%
lst = [
   'qtdeDiasD14', 
   'propAvgQtdeDias', 
   'propAvgQtdePontosPos',
   'propAvgQtdeTransacoes',
   'propAvgMediaTransacoesDias',
   'qtdeDiasD28',
   'qtdePontosPos',
   'qtdeChatMessage',
   'mediaTransacoesDias',]

df_train[lst]

# %%
corr = df_train[lst].corr(method='pearson')
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação CORR')
plt.tight_layout()
plt.show()

# %%
best_features = feature_importance[feature_importance['acum_pct'] <= 80]['features'].tolist()

# %%
from feature_engine import discretisation

tree_discretization = (
   discretisation.DecisionTreeDiscretiser(
      variables=best_features,
      bin_output='bin_number',
      regression=False,
      random_state=42,
      cv=2
   ))

tree_discretization.fit(X_train[best_features], y_train)
X_train_transformed = tree_discretization.transform(X_train[best_features])

# %%
from sklearn import linear_model

reg = (
   linear_model.LogisticRegression(
      penalty=None,
      random_state=42,
))

reg.fit(X_train_transformed, y_train)

# %%
y_train_predict = reg.predict(X_train_transformed)
y_train_proba = reg.predict_proba(X_train_transformed)[:,1]

# %%
print('TREINO')
avaliar_modelo(
   y_train, 
   y_train_predict, 
   y_train_proba)

# %%
X_test_transformed = tree_discretization.transform(X_test[best_features])
y_test_predict = reg.predict(X_test_transformed)
y_test_proba = reg.predict_proba(X_test_transformed)[:,1]

# %%
print('TESTE')
avaliar_modelo(
   y_test, 
   y_test_predict, 
   y_test_proba
)

# %%
X_oot_transformed = tree_discretization.transform(df_oot[best_features])
y_oot_predict = reg.predict(X_oot_transformed)
y_oot_proba = reg.predict_proba(X_oot_transformed)[:,1]

# %%
print('OUT OF TIME')
avaliar_modelo(
   df_oot[target], 
   y_oot_predict, 
   y_oot_proba
)
# %%
