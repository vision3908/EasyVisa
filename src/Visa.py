


# -*- coding: utf-8 -*-
"""
EasyVisa Machine Learning Project with MLflow Integration
Predicts visa approval status using ensemble learning methods
"""

# ============================================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================================

# Core data manipulation libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# XGBoost
from xgboost import XGBClassifier

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

print("=" * 80)
print("LIBRARIES IMPORTED SUCCESSFULLY")
print("=" * 80)


# ============================================================================
# STEP 2: CONFIGURE MLFLOW
# ============================================================================

# Set MLflow tracking URI (local directory)
mlflow.set_tracking_uri("file:./mlruns")

# Set experiment name
mlflow.set_experiment("EasyVisa_Prediction")

print("\n" + "=" * 80)
print("MLFLOW CONFIGURED")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experiment: {mlflow.get_experiment_by_name('EasyVisa_Prediction').name}")
print("=" * 80)


# ============================================================================
# STEP 3: LOAD AND PREPARE DATASET
# ============================================================================

def load_data(filepath):
    """Load visa dataset from CSV file"""
    try:
        data = pd.read_csv("data/EasyVisa.csv")
        print(f"\n✓ Dataset loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
        return data
    except FileNotFoundError:
        print(f"\n✗ ERROR: File '{filepath}' not found!")
        print("Please ensure 'EasyVisa.csv' is in the same directory as this script.")
        return None

# Load the dataset
data = load_data('EasyVisa.csv')

if data is None:
    print("\nExiting due to missing dataset...")
    exit(1)
# ============================================================================
# STEP 4: DATA PREPROCESSING
# ============================================================================

def preprocess_data(data):
    """Clean and prepare data for modeling"""
    
    # Fix negative employee counts
    data["no_of_employees"] = abs(data["no_of_employees"])
    
    # Drop case_id (unique identifier)
    data = data.drop('case_id', axis=1)
    
    # Create a copy
    df = data.copy()
    
    print("\n✓ Data preprocessing completed")
    print(f"  - Fixed {(data['no_of_employees'] < 0).sum()} negative employee counts")
    print(f"  - Dropped case_id column")
    print(f"  - Final shape: {df.shape}")
    
    return df

data = preprocess_data(data)


# ============================================================================
# STEP 5: FEATURE ENGINEERING
# ============================================================================

def encode_features(data):
    """Encode categorical variables"""
    
    # Separate target variable
    X = data.drop(['case_status'], axis=1)
    y = data['case_status']
    
    # One-hot encoding for categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Encode target variable
    y = y.map({'Denied': 0, 'Certified': 1})
    
    print(f"\n✓ Feature engineering completed")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Target classes: {y.nunique()}")
    print(f"  - Class distribution: {dict(y.value_counts())}")
    
    return X, y

X, y = encode_features(data)


# ============================================================================
# STEP 6: TRAIN-VALIDATION-TEST SPLIT
# ============================================================================

# First split: separate test set (15%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=1, stratify=y
)

# Second split: separate validation set (15% of remaining = ~12.75% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15, random_state=1, stratify=y_train_val
)

print(f"\n✓ Data split completed")
print(f"  - Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  - Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"  - Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")


# ============================================================================
# STEP 7: HANDLE CLASS IMBALANCE
# ============================================================================

# Oversampling with SMOTE
smote = SMOTE(sampling_strategy=1.0, random_state=1)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

# Undersampling
rus = RandomUnderSampler(sampling_strategy=1.0, random_state=1)
X_train_un, y_train_un = rus.fit_resample(X_train, y_train)

print(f"\n✓ Class imbalance handled")
print(f"  - Original: {len(y_train)} samples")
print(f"  - Oversampled: {len(y_train_over)} samples")
print(f"  - Undersampled: {len(y_train_un)} samples")


# ============================================================================
# STEP 8: MODEL PERFORMANCE EVALUATION FUNCTION
# ============================================================================

def model_performance_classification_sklearn(model, X, y):
    """
    Calculate and return model performance metrics
    """
    pred = model.predict(X)
    
    # Calculate metrics
    acc = accuracy_score(y, pred)
    recall = recall_score(y, pred)
    precision = precision_score(y, pred)
    f1 = f1_score(y, pred)
    
    # Create performance dataframe
    df_perf = pd.DataFrame(
        {
            "Accuracy": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
        },
        index=[0],
    )
    
    return df_perf


# ============================================================================
# STEP 9: TRAIN MODELS WITH MLFLOW TRACKING
# ============================================================================

def train_and_log_model(model, model_name, X_train, y_train, X_val, y_val, sampling_method):
    """
    Train model and log to MLflow
    """
    with mlflow.start_run(run_name=f"{model_name}_{sampling_method}"):
        
        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("sampling_method", sampling_method)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Calculate metrics
        train_metrics = {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "train_precision": precision_score(y_train, train_pred),
            "train_recall": recall_score(y_train, train_pred),
            "train_f1": f1_score(y_train, train_pred)
        }
        
        val_metrics = {
            "val_accuracy": accuracy_score(y_val, val_pred),
            "val_precision": precision_score(y_val, val_pred),
            "val_recall": recall_score(y_val, val_pred),
            "val_f1": f1_score(y_val, val_pred)
        }
        
        # Log metrics
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(val_metrics)
        
        # Log model
        signature = infer_signature(X_train, train_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        
        # Print results
        print(f"\n{model_name} ({sampling_method}):")
        print(f"  Train F1: {train_metrics['train_f1']:.4f}")
        print(f"  Val F1: {val_metrics['val_f1']:.4f}")
        
        return model, val_metrics['val_f1']


print("\n" + "=" * 80)
print("TRAINING MODELS WITH MLFLOW TRACKING")
print("=" * 80)

# Train baseline models
models_to_train = [
    (RandomForestClassifier(random_state=1, n_estimators=100), "RandomForest", X_train_over, y_train_over, "oversampled"),
    (AdaBoostClassifier(random_state=1), "AdaBoost", X_train_over, y_train_over, "oversampled"),
    (GradientBoostingClassifier(random_state=1), "GradientBoosting", X_train_over, y_train_over, "oversampled"),
    (XGBClassifier(random_state=1, use_label_encoder=False, eval_metric='logloss'), "XGBoost", X_train_over, y_train_over, "oversampled"),
]

trained_models = []
for model, name, X_tr, y_tr, sampling in models_to_train:
    trained_model, val_f1 = train_and_log_model(model, name, X_tr, y_tr, X_val, y_val, sampling)
    trained_models.append((trained_model, name, val_f1))


# ============================================================================
# STEP 10: HYPERPARAMETER TUNING WITH MLFLOW
# ============================================================================

print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING")
print("=" * 80)

# Define custom scorer
from sklearn.metrics import make_scorer
scorer = make_scorer(f1_score)

# Tune Gradient Boosting
with mlflow.start_run(run_name="GradientBoosting_Tuned"):
    
    mlflow.log_param("tuning_method", "RandomizedSearchCV")
    mlflow.log_param("cv_folds", 5)
    
    model = GradientBoostingClassifier(random_state=1)
    
    param_grid = {
        "n_estimators": [100, 125, 150, 175, 200],
        "learning_rate": [0.1, 0.05, 0.01, 0.005],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "max_features": ["sqrt", "log2", 0.3, 0.5]
    }
    
    randomized_cv = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        n_jobs=-2,
        scoring=scorer,
        cv=5,
        random_state=1
    )
    
    randomized_cv.fit(X_train_over, y_train_over)
    
    # Log best parameters
    for param, value in randomized_cv.best_params_.items():
        mlflow.log_param(f"best_{param}", value)
    
    mlflow.log_metric("best_cv_score", randomized_cv.best_score_)
    
    # Train final model with best parameters
    tuned_gbm = randomized_cv.best_estimator_
    tuned_gbm.fit(X_train_over, y_train_over)
    
    # Evaluate
    train_pred = tuned_gbm.predict(X_train_over)
    val_pred = tuned_gbm.predict(X_val)
    test_pred = tuned_gbm.predict(X_test)
    
    # Log final metrics
    mlflow.log_metric("final_train_f1", f1_score(y_train_over, train_pred))
    mlflow.log_metric("final_val_f1", f1_score(y_val, val_pred))
    mlflow.log_metric("final_test_f1", f1_score(y_test, test_pred))
    mlflow.log_metric("final_test_accuracy", accuracy_score(y_test, test_pred))
    mlflow.log_metric("final_test_precision", precision_score(y_test, test_pred))
    mlflow.log_metric("final_test_recall", recall_score(y_test, test_pred))
    
    # Log model
    signature = infer_signature(X_train_over, train_pred)
    mlflow.sklearn.log_model(tuned_gbm, "tuned_model", signature=signature)
    
    # Save feature importance plot
    plt.figure(figsize=(12, 8))
    feature_names = X_train.columns
    importances = tuned_gbm.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # Top 20 features
    
    plt.title("Top 20 Feature Importances")
    plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()
    
    print(f"\n✓ Tuned Gradient Boosting Model")
    print(f"  Best CV Score: {randomized_cv.best_score_:.4f}")
    print(f"  Test F1 Score: {f1_score(y_test, test_pred):.4f}")
    print(f"  Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")


# ============================================================================
# STEP 11: FINAL MODEL EVALUATION AND REPORTING
# ============================================================================

print("\n" + "=" * 80)
print("FINAL MODEL PERFORMANCE ON TEST SET")
print("=" * 80)

test_perf = model_performance_classification_sklearn(tuned_gbm, X_test, y_test)
print(test_perf)

# Confusion Matrix
cm = confusion_matrix(y_test, test_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Classification Report
print(f"\nClassification Report:")
print(classification_report(y_test, test_pred, target_names=['Denied', 'Certified']))


# ============================================================================
# STEP 12: SAVE FINAL RESULTS
# ============================================================================

# Save test predictions
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': test_pred,
    'Correct': y_test == test_pred
})
results_df.to_csv('test_predictions.csv', index=False)
print(f"\n✓ Test predictions saved to 'test_predictions.csv'")
# Save test predictions
results_df.to_csv('test_predictions.csv', index=False)

# Save trained model
import joblib
joblib.dump(tuned_gbm, "models/visa_model.pkl")
joblib.dump(X_train.columns.tolist(), "models/feature_names.pkl")

print("Model saved successfully tp models folder!")

print("\n" + "=" * 80)
print("MLFLOW EXPERIMENT TRACKING COMPLETE")
print("=" * 80)
print("\nTo view results in MLflow UI, run:")
print("  mlflow ui")
print("\nThen open: http://localhost:5000")
print("=" * 80)
