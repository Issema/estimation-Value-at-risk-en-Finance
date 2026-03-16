import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import jarque_bera, t

# -----------------------------
# Chargement et nettoyage des données
# -----------------------------
spy = pd.read_csv("SPY.csv").drop([0,1])
qqq = pd.read_csv("QQQ.csv").drop([0,1])
cac40 = pd.read_csv("FCHI.csv").drop([0,1])

for df in [spy, qqq, cac40]:
    df.rename(columns={'Price': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

# Extraction des prix ajustés
spy_adj = spy[['Adj Close', 'Date']].rename(columns={'Adj Close': 'SPY'})
qqq_adj = qqq[['Adj Close', 'Date']].rename(columns={'Adj Close': 'QQQ'})
cac40_adj = cac40[['Adj Close', 'Date']].rename(columns={'Adj Close': 'FCHI'})

# Fusion
data = spy_adj.merge(qqq_adj, on='Date').merge(cac40_adj, on='Date')
data = data[['Date','SPY','QQQ','FCHI']]  # Date en première colonne

# -----------------------------
# Calcul des rendements
# -----------------------------
returns = np.log(data[['SPY','QQQ','FCHI']] / data[['SPY','QQQ','FCHI']].shift(1)).dropna()
returns['Date'] = data['Date'][1:].values

# Rendement portefeuille
weights = [1/3, 1/3, 1/3]
returns['portfolio'] = returns[['SPY','QQQ','FCHI']].dot(weights)

# -----------------------------
# Visualisation des données
# -----------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution des rendements
plt.figure(figsize=(8,4))
plt.hist(returns['portfolio'], bins=50)
plt.title("Distribution des rendements du portefeuille")
plt.show()

# Série temporelle des rendements
plt.figure(figsize=(10,4))
plt.plot(returns['Date'], returns['portfolio'])
plt.title("Rendements du portefeuille dans le temps")
plt.show()

# Corrélation entre indices
plt.figure(figsize=(6,4))
sns.heatmap(returns[['SPY','QQQ','FCHI']].corr(), annot=True)
plt.title("Corrélation entre indices")
plt.show()

# -----------------------------
# Séparer 20% test pour toutes les méthodes
# -----------------------------
test_size = 0.2
n_test = int(len(returns) * test_size)
train_returns = returns.iloc[:-n_test].reset_index(drop=True)
test_returns  = returns.iloc[-n_test:].reset_index(drop=True)

# -----------------------------
# Méthode Historique
# -----------------------------
window = 250
var_list = []
real_returns = []
dates = []

for i in range(window, len(train_returns)):
    past_data = train_returns['portfolio'].iloc[i-window:i]
    var_hist = past_data.quantile(0.05)
    var_list.append(var_hist)

# On compare uniquement sur les 20% test
var_hist_series = pd.Series(var_list[-n_test:]).reset_index(drop=True)
real_test = test_returns['portfolio'].reset_index(drop=True)
var_results_hist = pd.DataFrame({
    "Date": test_returns['Date'],
    "VaR_hist": var_hist_series,
    "Real_return": real_test
})
var_results_hist["violation"] = var_results_hist["Real_return"] < var_results_hist["VaR_hist"]

violations_hist = var_results_hist["violation"].sum()
expected_hist = 0.05 * len(test_returns)
print("Historique - Nombre de violations :", violations_hist)
print("Historique - Violations attendues :", expected_hist)

# -----------------------------
# Méthode GARCH
# -----------------------------
stat, p = jarque_bera(train_returns['portfolio'])
if p > 0.05:
    print("GARCH: Rendements approx normaux")
else:
    print("GARCH: Rendements non-normaux, on utilise t-student")

train_port = train_returns['portfolio'] * 100
test_port  = test_returns['portfolio'] * 100

model = arch_model(train_port, vol='Garch', p=1, q=1, dist='t')
model_fit = model.fit(disp='off')
volatility = model_fit.conditional_volatility
df_t = model_fit.params['nu']
z = t.ppf(0.05, df_t)
var_garch_full = z * volatility

# On prend seulement la dernière partie correspondant aux indices de test
var_garch_test = var_garch_full[-n_test:].reset_index(drop=True)
violations_garch = test_port.reset_index(drop=True) < var_garch_test
nb_violations_garch = violations_garch.sum()
expected_garch = 0.05 * len(test_port)
print("GARCH - Nombre de violations :", nb_violations_garch)
print("GARCH - Violations attendues :", expected_garch)

# -----------------------------
# Méthode Random Forest
# -----------------------------
data_ml = returns.copy()

data_ml['portfolio'] = data_ml[['SPY','QQQ','FCHI']].mean(axis=1)

window = 20
for col in ['SPY','QQQ','FCHI','portfolio']:
    for i in range(1, window+1):
        data_ml[f'{col}_lag{i}'] = data_ml[col].shift(i)
data_ml = data_ml.dropna().reset_index(drop=True)

X = data_ml[[c for c in data_ml.columns if 'lag' in c]]
y = data_ml['portfolio']

# Découpage train/test identique
X_train = X.iloc[:-n_test]
X_test  = X.iloc[-n_test:]
y_train = y.iloc[:-n_test]
y_test  = y.iloc[-n_test:]

rf = RandomForestRegressor(n_estimators=400, random_state=30)
rf.fit(X_train, y_train)

all_tree_preds = np.array([tree.predict(X_test) for tree in rf.estimators_])
var_day = np.quantile(all_tree_preds, 0.05, axis=0)

#--------------------------------------
#EVALUATION
#--------------------------------------

violations_rf = y_test.values < var_day
nb_violations_rf = violations_rf.sum()
expected_rf = 0.05 * len(y_test)
print("Random Forest - Nombre de violations :", nb_violations_rf)
print("Random Forest - Violations attendues :", expected_rf)

#---------------------------------------
#Regression quantile
#---------------------------------------


import xgboost as xgb

data_ml = returns.copy()
data_ml['portfolio'] = data_ml[['SPY','QQQ','FCHI']].mean(axis=1)

# --- AJOUT ICI : Les indicateurs de risque ---
data_ml['vol_short'] = data_ml['portfolio'].rolling(5).std()
data_ml['vol_long'] = data_ml['portfolio'].rolling(21).std()
data_ml['abs_ret'] = data_ml['portfolio'].abs()
data_ml['min_ret_5d'] = data_ml['portfolio'].rolling(5).min()
# ----------------------------------------------

window = 20
# On ajoute les nouvelles colonnes dans la boucle pour créer des lags
features_to_lag = ['SPY','QQQ','FCHI','portfolio', 'vol_short', 'vol_long', 'abs_ret', 'min_ret_5d']

for col in features_to_lag:
    for i in range(1, window+1):
        data_ml[f'{col}_lag{i}'] = data_ml[col].shift(i)

# On nettoie les NaN créés par le rolling et le shift
data_ml = data_ml.dropna().reset_index(drop=True)

# Définition de X et y
X = data_ml[[c for c in data_ml.columns if 'lag' in c]]
y = data_ml['portfolio']


# Découpage train/test identique
X_train = X.iloc[:-n_test]
X_test  = X.iloc[-n_test:]
y_train = y.iloc[:-n_test]
y_test  = y.iloc[-n_test:]
# ... la suite de ton code (split train/test et XGBoost)

# 1. Configuration du modèle pour le quantile 0.05 (VaR 95%)
# On utilise la fonction de perte 'reg:quantileerror'
alpha = 0.05
model_quantile = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=alpha,
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    random_state=30
)

# 2. Entraînement
model_quantile.fit(X_train, y_train)

# 3. Prédiction directe de la VaR
# Contrairement au RF, ici la sortie du modèle EST la VaR directement
var_quantile_xgb = model_quantile.predict(X_test)

#--------------------------------------
# EVALUATION COMPARATIVE
#--------------------------------------
violations_xgb = y_test.values < var_quantile_xgb
nb_violations_xgb = violations_xgb.sum()

print(f"XGBoost Quantile - Nombre de violations : {nb_violations_xgb}")
print(f"Violations attendues : {expected_rf}")


# alpha = 0.05

# errors = y_test.values - var_day
# pinball_loss = np.mean(np.where(errors >= 0,
#                                 alpha * errors,
#                                 (alpha - 1) * errors))

# print("Pinball Loss :", pinball_loss)

# from sklearn.metrics import mean_squared_error

# mse = mean_squared_error(y_test.values, var_day)

# print("MSE :", mse)


