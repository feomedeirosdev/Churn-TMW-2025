# %%
import pandas as pd

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

# %%
pd.DataFrame(df[df.select_dtypes(include=['object']).columns.tolist()].describe().T)

# %%
pd.DataFrame(df[df.select_dtypes(include=['int', 'float']).columns.tolist()].describe().T)

# %%
df['flagChurn'].value_counts(normalize=True)

# %%
oot_filter = df['dtRef'] == df['dtRef'].max()
df_oot = df[oot_filter].copy()

# %%
df_oot['flagChurn'].value_counts(normalize=True)

# %%
base_filter = df['dtRef'] != df['dtRef'].max()
df_base = df[base_filter].copy()

# %%
df_base['flagChurn'].value_counts(normalize=True)

# %%
df_summary(df)

# %%
df_summary(df_oot)

# %%
df_summary(df_base)

# %%
target = 'flagChurn'
features = df_base.columns[2:-1]

# %%
cat_features = df_base[features].select_dtypes(include=['object']).columns.tolist()

# %%
num_features = df_base[features].select_dtypes(include=['int', 'float']).columns.tolist()

# %%
y = df_base[target]
X = df_base[features]

# %%
y.mean()

# %%
df_summary(X)

# %%
df_summary(y)

# %%
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, 
    y,
    random_state=42,
    stratify=y,
    test_size=0.2)

# # %%
# df_summary(X_train)

# # %%
# df_summary(X_test)

# # %%
# df_summary(y_train)

# # %%
# df_summary(y_test)

# %%
print(f'Taxa da variável resposta geral: {y.mean()*100:.2f}%')
print(f'Taxa da variável resposta de treino: {y_train.mean()*100:.2f}%')
print(f'Taxa da variável resposta de test: {y_test.mean()*100:.2f}%')

# %%
df_summary(X_train)

# %%
df_train = X_train.copy()
df_train[target] = y_train.copy()

# %%
df_summary(df_train)

# %%
pd.set_option('display.max_rows', 10)

train_analysis = df_train.groupby(by=target).agg(['mean', 'median']).T

# %%
train_analysis['diff_abs'] = abs(train_analysis[0] - train_analysis[1])

train_analysis['diff_rel'] = abs(train_analysis[0] / train_analysis[1])*100

train_analysis.sort_values(by=['diff_rel'], ascending=False)

# %%
from sklearn import tree

# %%
arvore = tree.DecisionTreeClassifier(
   random_state=42,
   # max_depth=5
)

arvore.fit(X_train, y_train)

# %%
import matplotlib.pyplot as plt

# %%
feature_importance = pd.DataFrame({
    'features': features,
    'percent': arvore.feature_importances_
}).sort_values(by='percent', ascending=False)

# %%
feature_importance['acum'] = feature_importance['acum'].cumsum()
feature_importance[feature_importance['acum'] < 0.95]

# %%
