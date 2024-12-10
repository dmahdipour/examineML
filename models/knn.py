from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

class KNNModel:
    def __init__(self):
        self.model = KNeighborsClassifier()

    def train_with_grid_search(self, X_train, y_train):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_
