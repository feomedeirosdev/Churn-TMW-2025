# %%
import pandas as pd

# %%
S_model = pd.read_pickle("../data/S_model.pkl")
model = S_model['model']
features = S_model['features']
df = pd.read_csv("../data/abt.csv")

# %%
amostra = df[df['dtRef'] == df['dtRef'].max()].sample(5)
amostra = amostra.drop(columns=['flagChurn'])

# %%
y_proba = model.predict_proba(amostra[features])[:,1]

# %%
amostra['proba'] = y_proba
print(amostra)