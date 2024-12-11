from utils.preprocessor import Preprocessor
from utils.modelResultVisualizer import ModelResultVisualizer
from utils.logger import get_logger
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
    
    logger = get_logger(__name__)
    logger.info("Starting the application...")
    
    try:
    
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
        
        visualizer = ModelResultVisualizer()

        # Train and evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            best_model, best_params = model.train_with_grid_search(X_train, y_train)
            accuracy = best_model.score(X_test, y_test)
            print(f"Best Parameters for {name}: {best_params}")
            print(f"{name} Model Accuracy: {accuracy}\n")
            
            
            # Generate predictions and probabilities for visualization
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
            
            # store results
            visualizer.store_results(name, best_params, accuracy)

            # Plot confusion matrix
            visualizer.plot_confusion_matrix(y_test, y_pred, class_names=['Class 0', 'Class 1'])

            # Plot classification report
            visualizer.plot_classification_report(y_test, y_pred)

            # Plot ROC curve 
            if y_proba is not None:
                visualizer.plot_roc_curve(y_test, y_proba)
                
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        
    logger.info("Application completed.")
