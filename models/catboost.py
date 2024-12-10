from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

class CatBoostModel:
    def __init__(self):
        self.model = CatBoostClassifier(verbose=0)

    def train_with_grid_search(self, X_train, y_train):
        param_grid = {
            'iterations': [100, 200],
            'learning_rate': [0.01, 0.1],
            'depth': [3, 6, 10],
            'l2_leaf_reg': [1, 3, 5],
            'subsample': [0.8, 1.0]
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_
