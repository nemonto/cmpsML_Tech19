#%% MODULE BEGINS
module_name = '<model_performance>'
'''
Version: <3.0>
Description:
<The implementation of ML models and performance assessments.>
Authors:
<Drew Hutchinson, Zichuo Wang>
Date Created : <4-19-2025>
Date Last Updated: <5-8-2025>
Doc:
<***>
Notes:
<***>
'''
#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os
#os.chdir("./../..")
#
#custom imports
#other imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import joblib

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, auc, classification_report)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


#%% USER INTERFACE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% CONFIGURATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% INITIALIZATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% DECLARATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Global declarations Start Here

# file path declaration
base_dir = Path(__file__).resolve().parent.parent
input_path = base_dir / "CODE" / "INPUT" / "feature_extracted_data.xlsx"
input_train_path = base_dir / "CODE" / "INPUT" / "TRAIN" / "train.xlsx"
input_test_path = base_dir / "CODE" / "INPUT" / "TEST" / "val_test.xlsx"
output_path = base_dir / "CODE" / "OUTPUT" / "feature_extracted_data.xlsx"
output_kf_cv_path= base_dir / "CODE" / "OUTPUT" / "model performance" / "gride_search_ann_kfold_results.xlsx"
model_path = base_dir / "CODE" / "MODEL"
plots_dir = base_dir / "CODE" / "OUTPUT" / "model performance"
plots_dir.mkdir(parents=True, exist_ok=True)

#Class definitions Start Here
#Function definitions Start Here
def data_setting(df):
    #df_whole = df['Whole']
    df_train, df_val, df_test =  df['Train_noise_std'], df['Val_std'], df['Test_std']

    X_train = df_train.drop(columns=['species','island_Biscoe','island_Dream','island_Torgersen'])
    y_train = df_train['species']
    X_val = df_val.drop(columns=['species','island_Biscoe','island_Dream','island_Torgersen'])
    y_val = df_val['species']
    X_test = df_test.drop(columns=['species','island_Biscoe','island_Dream','island_Torgersen'])
    y_test = df_test['species']

    return X_train, y_train, X_val, y_val, X_test, y_test

def noisify(df):
    noise_std_dict = {
        'bill_length_mm': 3.0,
        'bill_depth_mm': 1.5,
        'flipper_length_mm': 7.0,
        'body_mass_g': 150.0
    }

    df = df.copy()

    for col, std in noise_std_dict.items():
        if col in df.columns:
            noise = np.random.normal(loc=0, scale=std, size=df[col].shape)
            df[col] += noise
        else:
            print(f"Column '{col}' not found in the dataset.")

    return df

def model_initialization():
    return {
        'GRID_SEARCH_ANN': None,

        'RANDOM_ANN': MLPClassifier(
            solver='adam',
            batch_size=60,
            max_iter=50,
            early_stopping=True,
            random_state=42,
            hidden_layer_sizes=(6,3),
            activation = 'relu',
            alpha = 0.001,
            learning_rate_init = 0.05,
            validation_fraction=0.2
        ),

        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            p=2
        ),

        'DT': DecisionTreeClassifier(
            criterion='gini',
            max_depth=4,
            min_samples_split=5,
            random_state=42
        ),

        'SVM': SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42
        )

    }

def grid_search_ann(X, y):
    print("\nRunning GridSearch for ANN...")
    # define pipeline and hyperparameter grid
    ann_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            solver='adam',
            batch_size='auto',
            max_iter=300,
            early_stopping=False,
            random_state=42
        ))
    ])

    param_grid = {
        "clf__hidden_layer_sizes": [(8, 4), (6, 3),(4,2)],
        "clf__activation": ['relu', 'tanh'],
        "clf__alpha": [0.0001, 0.001],
        "clf__learning_rate_init": [0.001, 0.005, 0.01]
    }

    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=ann_pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X, y)

    print(f"Best parameters for ANN: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.6f}")

    # evaluate the best model on the same KFold to get fold-wise scores
    best_ann_model = grid_search.best_estimator_
    y_pred = best_ann_model.predict(X)
    ann_clf = best_ann_model.named_steps["clf"]
    if hasattr(ann_clf, 'loss_curve_'):
        plt.plot(ann_clf.loss_curve_, label= "GRID_SEARCH_ANN (training loss)")
    else:
        print(f"No loss_curve_ for GRID_SEARCH_ANN")
    plt.title("Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / f"GRID_SEARCH_ANN_training_loss_curve.png")
    plt.show()
    scores = cross_val_score(best_ann_model, X, y, cv=skf, scoring='accuracy')

    # save results to Excel
    df_scores = pd.DataFrame({
        "Fold": list(range(1, len(scores)+1)),
        "Accuracy": scores
    })
    df_scores.loc["Mean"] = ["Mean", scores.mean()]
    df_scores.to_excel(output_kf_cv_path, index=False)

    print(f"K-Fold results saved to: {output_kf_cv_path}")

    print("\nClassification report on full dataset:")
    print(classification_report(y, y_pred, target_names=['Adelie', 'Chinstrap', 'Gentoo']))

    return best_ann_model

def model_training(models, X_train, y_train):
    trained_models = {}

    for name, model in models.items():
        if name == 'GRID_SEARCH_ANN':
            trained_models[name] = model

        elif name == 'RANDOM_ANN':
            model.fit(X_train, y_train)
            trained_models[name] = model
            if hasattr(model, 'loss_curve_'):
                plt.plot(model.loss_curve_, label=f'{name} (training loss)')
            else:
                print(f"No loss_curve_ for {name}")
            plt.title("Training Loss Curve")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plots_dir / f"{name}_training_loss_curve.png")
            plt.show()

        else:
            model.fit(X_train, y_train)
            trained_models[name] = model
    print(trained_models)
    return trained_models

def model_evaluation(model, X_test, y_test, model_name, plots_dir):
    class_names = ['Adelie', 'Chinstrap', 'Gentoo']
    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{model_name}_confusion_matrix.png")
    plt.close()

    print(f"\n{model_name} Confusion Matrix:\n", cm_df)

    # ROC Curve for Multiclass
    auc_scores = []
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], 'k--', label="Random")
        plt.title(f"{model_name} ROC Curve (Multiclass)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{model_name}_roc_curve.png")
        plt.close()

        print(f"{model_name} Average ROC AUC: {np.mean(auc_scores):.4f}")
    else:
        print(f"{model_name} does not support predict_proba, skipping ROC AUC.")

    # Overall Metrics (macro average for multiclass)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # Specificity for multiclass (1 - FPR), approximated
    specificity_list = []
    for i in range(len(class_names)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_list.append(specificity)
    specificity = np.mean(specificity_list)

    print(f"\n{model_name} Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Specificity (avg): {specificity:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": np.mean(auc_scores) if auc_scores else None
    }

    return y_pred, cm, metrics

def save_model(model, name):
    joblib.dump(model, model_path / f"{name.lower()}_model.pkl")

def plot_evaluation(metrics):

    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    excel_path = plots_dir / "model_metrics_summary.xlsx"
    metrics_df.to_excel(excel_path)

    plt.figure(figsize=(10, 6))
    metrics_df.plot(kind='bar', figsize=(12, 6), colormap='Set2', edgecolor='black')
    plt.title("Model Comparison on Evaluation Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    plt.grid(axis='y')
    plt.tight_layout()

    plot_path = plots_dir / "model_comparison_bar_chart.png"
    plt.savefig(plot_path)
    plt.show()

def robustness_on_noisy_test(models, X_test_base, y_test, plots_dir, rounds=10):

    results = {name: {"accuracy": [], "recall": [], "f1_score": []} for name in models.keys()}

    for round_idx in range(rounds):
        X_test_noisy = noisify(X_test_base)

        for name, model in models.items():
            y_pred = model.predict(X_test_noisy)

            acc = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            results[name]["accuracy"].append(acc)
            results[name]["recall"].append(recall)
            results[name]["f1_score"].append(f1)

    all_results = []
    for metric in ["accuracy", "recall", "f1_score"]:
        for model_name, scores in results.items():
            for score in scores[metric]:
                all_results.append({
                    "Model": model_name,
                    "Metric": metric.capitalize(),
                    "Score": score
                })

    df_plot = pd.DataFrame(all_results)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_plot, x="Model", y="Score", hue="Metric", palette="Set2")
    plt.title("Robustness Test (Noisy Test Set) – Overall Box Plot")
    plt.ylim(0, 1.05)
    plt.grid(axis='y')
    plt.tight_layout()
    plot_path = plots_dir / "noisy_test_robustness_combined_boxplot.png"
    plt.savefig(plot_path)
    plt.show()

    return df_plot

def main():
    np.set_printoptions(precision=6, suppress=True)
    #read from base_dir / "CODE" / "INPUT" / "feature_extracted_data.xlsx"
    df = pd.read_excel(input_path, sheet_name=None)

    #initialize the subsets
    X_train, y_train, X_val, y_val, X_test, y_test, = data_setting(df)

    #initialize the models
    models = model_initialization()
    #training the models
    #applying grid search and k-fold validation on ANN
    best_ann_model = grid_search_ann(X_train,y_train)
    models['GRID_SEARCH_ANN'] = best_ann_model
    trained_models = model_training(models, X_train, y_train)

    #model evaluation
    KNN_res, KNN_cm, KNN_metrics = model_evaluation(trained_models['KNN'], X_test, y_test, 'KNN', plots_dir)
    DT_res, DT_cm, DT_metrics = model_evaluation(trained_models['DT'], X_test, y_test, 'DT', plots_dir)
    SVM_res, SVM_cm, SVM_metrics = model_evaluation(trained_models['SVM'], X_test, y_test, 'SVM', plots_dir)
    RANDOM_ANN_res, RANDOM_ANN_cm, RANDOM_ANN_metrics = model_evaluation(trained_models['RANDOM_ANN'], X_test, y_test, 'RANDOM_ANN', plots_dir)
    GRID_SEARCH_ANN_res, GRID_SEARCH_ANN_cm, GRID_SEARCH_ANN_metrics = model_evaluation(trained_models['GRID_SEARCH_ANN'], X_test, y_test, 'GRID_SEARCH_ANN', plots_dir)

    all_metrics = {
        "KNN": KNN_metrics,
        "DT": DT_metrics,
        "SVM": SVM_metrics,
        "RANDOM_ANN": RANDOM_ANN_metrics,
        "GRID_SEARCH_ANN": GRID_SEARCH_ANN_metrics,
    }

    plot_evaluation(all_metrics)

    for name, model in trained_models.items():
        save_model(model, name)

    #robustness test
    robustness_on_noisy_test(trained_models, X_test, y_test, plots_dir, rounds=10)


#
#%% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here

#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
#TEST Code
main()
