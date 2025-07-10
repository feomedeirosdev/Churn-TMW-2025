# %%
import pandas as pd
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
# mlflow.set_experiment(experiment_id='955180276940558567')

# %%
client = mlflow.client.MlflowClient()
last_version = max([int(i.version) for i in client.get_latest_versions('churn-tmw')])

# %%
model = mlflow.sklearn.load_model(f'models:/churn-tmw/{last_version}')
df = pd.read_csv('../data/abt.csv')
df_oot = df[df['dtRef'] == df['dtRef'].max()]
X = df_oot.sample(5)[model.feature_names_in_]
X['proba'] = model.predict_proba(X)[:,1]
X

