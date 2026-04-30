from src.preprocess import load_and_preprocess
from src.train import train_all
from src.evaluate import evaluate_all, plot_feature_importance
from src.train import tune_gb
import pickle

X_train, X_test, y_train, y_test = load_and_preprocess("data/telco_churn.csv")

trained_models = train_all(X_train, y_train)
evaluate_all(trained_models, X_test, y_test)

best_gb = tune_gb(X_train, y_train)
evaluate_all({"Tuned GB": best_gb}, X_test, y_test)

# Feature importance from best model
plot_feature_importance(trained_models["XGBoost"], X_train.columns.tolist())

pickle.dump(trained_models["Gradient Boosting"], open("models/gb_model.pkl", "wb"))
pickle.dump(trained_models["XGBoost"],           open("models/xgb_model.pkl", "wb"))
print("✅ Models saved to /models")