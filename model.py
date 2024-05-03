deneme_model= merged_df[merged_df["store_nbr"]==2]

train_2 = deneme_model.loc[(deneme_model["date"] < "2017-07-15"), :]
test_2 = deneme_model.loc[(deneme_model["date"] >= "2017-07-15"), :]

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return np.sqrt(mean_squared_error(np.expm1(y), np.expm1(y_pred)))

metrics_per_family = {}
models_per_family = {}

for family in train_2.family.unique():

    new_df = train_2[train_2['family'] == family]


    def objective(trial):

        model_name = trial.suggest_categorical('model', ['LGBM', 'RandomForest', 'XGBoost'])
        if model_name == 'LGBM':
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 50),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 20),
                "verbose": -1
            }
            model = LGBMRegressor(**params)
        elif model_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
            model = RandomForestRegressor(**params)
        elif model_name == 'XGBoost':
            params = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
                'gamma': trial.suggest_loguniform('gamma', 0.1, 1.0),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 100),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 100)
            }
            model = XGBRegressor(**params)

        tscv = TimeSeriesSplit(n_splits=5)
        rmse_scores = []

        cols_to_drop = ["date", "store_nbr", "sales", "family", "year"]
        X = new_df.drop(cols_to_drop, axis=1)
        y = new_df["sales"]

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = train_model(model, X_train, y_train)
            rmse = evaluate_model(model, X_test, y_test)
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)

    best_params = study.best_params
    best_params['family'] = family


    best_model_name = best_params.pop("model")
    if best_model_name == "LGBM":
        model = LGBMRegressor(**best_params)
    elif best_model_name == "RandomForest":
        best_params.pop("family")
        model = RandomForestRegressor(**best_params)
    elif best_model_name == "XGBoost":
        model = XGBRegressor(**best_params)

    cols_to_drop = ["date", "store_nbr", "sales", "family", "year"]
    X_train = new_df.drop(cols_to_drop, axis=1)
    y_train = new_df["sales"]
    model.fit(X_train, y_train)


    models_per_family[family] = model

    family_test_data = test_2[test_2["family"] == family]
    cols_to_drop = ["date", "store_nbr", "sales", "family", "year"]
    X_test = family_test_data.drop(cols_to_drop, axis=1)
    y_test = family_test_data["sales"]

    y_pred = model.predict(X_test)

    Y_pred_original = np.expm1(y_pred)
    Y_val_original = np.expm1(y_test)

    mse_mean = mean_squared_error(Y_val_original, Y_pred_original)
    mae_mean = mean_absolute_error(Y_val_original, Y_pred_original)
    rmse_mean = np.sqrt(mse_mean)

    metrics_per_family[family] = {"Model": best_model_name, "MSE": mse_mean, "MAE": mae_mean, "RMSE": rmse_mean}

    plt.figure(figsize=(10, 6))
    plt.plot(family_test_data["date"], Y_val_original, label="Actual", color="blue")
    plt.plot(family_test_data["date"], Y_pred_original, label="Predicted", color="red")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title(f"{family} Sales Prediction")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

metrics_df = pd.DataFrame.from_dict(metrics_per_family, orient="index")

print(metrics_df)


store = '/content/store'

for family, model in models_per_family.items():

    dosya_adi = f"store_2_{family}_model.pkl"
    dosya_yolu = os.path.join(store, dosya_adi)
    joblib.dump(model, dosya_yolu)
    print(f"{family} modeli {dosya_adi} dosyasına kaydedildi.")

from google.colab import files
for dosya_adi in os.listdir(store):
    files.download(os.path.join(store, dosya_adi)) 
