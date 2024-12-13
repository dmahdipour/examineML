import xgboost as xgb
from sklearn.model_selection import GridSearchCV

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method='gpu_hist')

    def train_with_grid_search(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 6, 10],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0.0, 1.0],
            'reg_lambda': [0.0, 1.0],
            'gamma': [0.0, 1.0, 3, 6 ,10],
            'min_child_weight': [0.0, 1.0, 5, 8, 10],
            'scale_pos_weight': [0.0, 1.0]
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_
