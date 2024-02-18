from sklearn.model_selection import train_test_split, GridSearchCV  # train_test_split is for splitting the dataset into training and testing sets, GridSearchCV is for tuning parameters
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # RandomForestClassifier and GradientBoostingClassifier are machine learning models
from sklearn.linear_model import LogisticRegression  # LogisticRegression is a machine learning model for binary classification tasks
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # For evaluating the performance of the model
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # StandardScaler is for standardizing features, OneHotEncoder is for converting categorical variables into a form that could be provided to ML algorithms
from sklearn.impute import SimpleImputer  # For handling missing data
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer  # For applying different preprocessing steps to different columns in the input data
from sklearn.pipeline import Pipeline  # For sequentially applying a list of transforms and a final estimator
from sklearn.metrics import roc_auc_score, roc_curve, auc  # For evaluating the performance of the model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
class WinPredictionModel:
    """
    A class for training and evaluating win prediction models.

    """

    def __init__(self, data, target_variable, features, categorical_features, numerical_features):
        """
        Initialize the WinPredictionModel class.

        Args:
        - data (pandas.DataFrame): The input data.
        - target_variable (str): The name of the target variable column.
        - features (list): The list of feature column names.
        - categorical_features (list): The list of categorical feature column names.
        - numerical_features (list): The list of numerical feature column names.
        """
        self.data = data.copy()  
        self.target_variable = target_variable
        self.features = features
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
        }
        self.results = {}
        self.clean_data()  # Ensure data is cleaned at initialization

    def clean_data(self):
        """
        Clean the numeric columns in the data by removing non-numeric characters and converting to float.
        """
        for column in self.numerical_features:
            self.data[column] = pd.to_numeric(self.data[column].astype(str).replace('\xa0', '').replace(',', ''), errors='coerce')
        

    def preprocess(self):
        """
        Preprocess the data by encoding the target variable and applying transformations to numerical and categorical features.

        Returns:
        - X_train (pandas.DataFrame): The preprocessed training features.
        - X_test (pandas.DataFrame): The preprocessed testing features.
        - y_train (pandas.Series): The encoded training target variable.
        - y_test (pandas.Series): The encoded testing target variable.
        """
        y = self.data[self.target_variable].apply(lambda x: 1 if x == 'team win' else 0)
        X = self.data[self.features]
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def define_hyperparameters(self):
        """
        Define the hyperparameters for each model.
        """
        # Define the hyperparameters for the RandomForest, GradientBoosting, and LogisticRegression models
        self.hyperparameters = {
            'RandomForest': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]
            },
            'GradientBoosting': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 10]
            },
            'LogisticRegression': {
                'classifier__C': np.logspace(-4, 4, 4),
                'classifier__penalty': ['l2']
            }
        }

    def train_and_evaluate(self):
        """
        Train and evaluate the models using cross-validation and grid search.
        """
        # Record the start time
        start_time = time.time()
        
        # Preprocess the data and split it into training and testing sets
        X_train, X_test, y_train, y_test = self.preprocess()
        
        # Define the hyperparameters for the models
        self.define_hyperparameters()
        # For each model, create a pipeline with the preprocessor and the model, and add it to the models dictionary
        for model_name, model in self.models.items():
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                       ('classifier', model)])
            
            if model_name in self.hyperparameters:
                randomized_search = RandomizedSearchCV(pipeline,
                                                       param_distributions=self.hyperparameters[model_name],
                                                       n_iter=10,
                                                       cv=5,
                                                       scoring='accuracy',
                                                       n_jobs=-1)
                
                randomized_search.fit(X_train, y_train)
                best_model = randomized_search.best_estimator_
                y_pred = best_model.predict(X_test)
                y_proba = best_model.predict_proba(X_test)[:, 1]

                self.results[model_name] = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'ROC AUC': roc_auc_score(y_test, y_proba),
                    'Best Parameters': randomized_search.best_params_,
                    'Confusion Matrix': confusion_matrix(y_test, y_pred),
                    'Classification Report': classification_report(y_test, y_pred, output_dict=True),
                    'ROC Curve': roc_curve(y_test, y_proba)
                }

                if model_name in ['RandomForest', 'GradientBoosting']:
                    importances = best_model.named_steps['classifier'].feature_importances_
                    feature_names = self.numerical_features + \
                                    list(best_model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(self.categorical_features))
                    feature_importances = dict(zip(feature_names, importances))
                    self.results[model_name]['Feature Importances'] = feature_importances

            else:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                
                self.results[model_name] = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'ROC AUC': roc_auc_score(y_test, y_proba),
                    'F1 Score': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score'],
                    'Confusion Matrix': confusion_matrix(y_test, y_pred),
                    'Classification Report': classification_report(y_test, y_pred, output_dict=True),
                    'ROC Curve': roc_curve(y_test, y_proba)

                }

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken to train and evaluate: {elapsed_time:.2f} seconds")

    def display_results(self):
        """
        Display the evaluation results for each model.
        """
        # Find the best models for each metric
        best_accuracy_model = max(self.results.items(), key=lambda x: x[1]['Accuracy'])
        best_precision_model = max(self.results.items(), key=lambda x: x[1]['Classification Report']['weighted avg']['precision'])
        best_recall_model = max(self.results.items(), key=lambda x: x[1]['Classification Report']['weighted avg']['recall'])
        best_roc_auc_model = max(self.results.items(), key=lambda x: x[1]['ROC AUC'])

        # Print the best models and their scores
        print(f"Best Model by Accuracy: {best_accuracy_model[0]} with Accuracy: {best_accuracy_model[1]['Accuracy']:.2f}\n")
        print(f"Best Model by Precision: {best_precision_model[0]} with Precision: {best_precision_model[1]['Classification Report']['weighted avg']['precision']:.2f}\n")
        print(f"Best Model by Recall: {best_recall_model[0]} with Recall: {best_recall_model[1]['Classification Report']['weighted avg']['recall']:.2f}\n")
        print(f"Best Model by ROC AUC: {best_roc_auc_model[0]} with ROC AUC: {best_roc_auc_model[1]['ROC AUC']:.2f}\n")

        # Print the results for each model
        for model_name, metrics in self.results.items():
            print(f"Results for {model_name}:")
            print(f"Accuracy: {metrics['Accuracy']:.2f}")
            print(f"Precision: {metrics['Classification Report']['weighted avg']['precision']:.2f}")
            print(f"Recall: {metrics['Classification Report']['weighted avg']['recall']:.2f}")
            print(f"F1 Score : {metrics['Classification Report']['weighted avg']['f1-score']:.2f}")
            print(f"ROC AUC: {metrics['ROC AUC']:.2f}")
            if 'Best Parameters' in metrics:
                print("Best Parameters:")
                for param_name, param_value in metrics['Best Parameters'].items():
                    print(f"  {param_name}: {param_value}")
            if 'Feature Importances' in metrics:
                print("Feature Importances:")
                for feature_name, feature_importance in sorted(metrics['Feature Importances'].items(), key=lambda item: item[1], reverse=True):
                    print(f"  {feature_name}: {feature_importance}")
            print("\n")

    def visualize_roc_curves(self):
        """
        Visualize the ROC curves for each model.
        """
        plt.figure(figsize=(10, 6))

        # Plot the ROC curve for each model
        for model_name, metrics in self.results.items():
            fpr, tpr, _ = metrics['ROC Curve']
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {metrics['ROC AUC']:.2f})")

        # Viz
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()
