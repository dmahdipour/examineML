from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

class AdaBoostModel:
    def __init__(self):
        self.model = AdaBoostClassifier()

    def train_with_grid_search(self, X_train, y_train):
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5, 1.0],
            'algorithm ':  ['SAMME', 'SAMME.R'],
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_
