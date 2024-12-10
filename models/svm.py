from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class SVMModel:
    def __init__(self):
        self.model = SVC()
        
    def train_with_grid_search(self, X_train, y_train):
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'max_iter': [100, 200, 300, 400, 500],
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_