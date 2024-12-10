from utils.preprocessor import Preprocessor
from models.randomForest import RandomForestModel
from models.logisticRegressor import LogisticRegressionModel
from models.SGDclassifier import SGDClassifierModel
from models.svm import SVMModel
from models.gradientBoosting import GradientBoostingModel
from models.adaBoostClassifier import AdaBoostModel
from models.knn import KNNModel
from models.xgboost import XGBoostModel
from models.catboost import CatBoostModel

if __name__ == "__main__":
    # Instantiate and preprocess data
    preprocessor = Preprocessor("Datasets\data.csv")
    X_train, X_test, y_train, y_test = preprocessor.get_train_test_data()

    # Models to evaluate
    models = {
        "RandomForest": RandomForestModel(),
        "LogisticRegression": LogisticRegressionModel(),
        "SGDClassifier": SGDClassifierModel(),
        "SVM": SVMModel(),
        "GradientBoosting": GradientBoostingModel(),
        "AdaBoost": AdaBoostModel(),
        "KNN": KNNModel(),
        "XGBoost": XGBoostModel(),
        "CatBoost": CatBoostModel()
    }

    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        best_model, best_params = model.train_with_grid_search(X_train, y_train)
        accuracy = best_model.score(X_test, y_test)
        print(f"Best Parameters for {name}: {best_params}")
        print(f"{name} Model Accuracy: {accuracy}\n")
