from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

class GradientBoostingModel:
    def __init__(self):
        self.model = GradientBoostingClassifier()

    def train_with_grid_search(self, X_train, y_train):
        param_grid = {
            'learning_rate': [0.1, 0.05, 0.01],
            'n_estimators': [50, 100, 200],
            'criterion': ['friedman_mse', 'squared_error'],
            'max_depth': [3, 5, 8, 15, 20],
            'min_samples_split' : [2, 5, 8, 15, 20]
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_
