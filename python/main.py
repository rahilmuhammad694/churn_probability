"""
Customer Churn Prediction System
A complete end-to-end ML project for predicting customer churn with EDA, 
feature engineering, model training, and evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictionSystem:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.feature_importance = None
        
    def generate_sample_data(self, n_samples=5000):
        """Generate realistic synthetic customer data"""
        np.random.seed(42)
        
        data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 70, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'tenure_months': np.random.randint(1, 72, n_samples),
            'monthly_charges': np.random.uniform(20, 150, n_samples),
            'total_charges': np.random.uniform(100, 8000, n_samples),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
            'payment_method': np.random.choice(['Electronic check', 'Credit card', 'Bank transfer', 'Mailed check'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
            'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'num_support_calls': np.random.poisson(2, n_samples),
            'num_late_payments': np.random.poisson(1, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable with realistic correlations
        churn_prob = (
            0.1 +  # base probability
            0.3 * (df['contract_type'] == 'Month-to-month') +
            0.2 * (df['tenure_months'] < 12) +
            0.15 * (df['num_support_calls'] > 3) +
            0.1 * (df['num_late_payments'] > 2) +
            0.15 * (df['monthly_charges'] > 100) -
            0.2 * (df['tech_support'] == 'Yes')
        )
        
        df['churn'] = (np.random.random(n_samples) < churn_prob).astype(int)
        
        return df
    
    def exploratory_data_analysis(self, df):
        """Perform comprehensive EDA"""
        print("="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        print("\n1. Dataset Overview:")
        print(f"Shape: {df.shape}")
        print(f"\nChurn Distribution:\n{df['churn'].value_counts()}")
        print(f"Churn Rate: {df['churn'].mean()*100:.2f}%")
        
        print("\n2. Missing Values:")
        print(df.isnull().sum())
        
        print("\n3. Numerical Features Summary:")
        print(df.describe())
        
        # Visualizations
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Customer Churn Analysis', fontsize=16)
        
        # Churn by contract type
        pd.crosstab(df['contract_type'], df['churn'], normalize='index').plot(
            kind='bar', ax=axes[0,0], color=['#2ecc71', '#e74c3c']
        )
        axes[0,0].set_title('Churn Rate by Contract Type')
        axes[0,0].set_ylabel('Proportion')
        
        # Tenure distribution
        df[df['churn']==0]['tenure_months'].hist(ax=axes[0,1], bins=30, alpha=0.5, label='No Churn', color='green')
        df[df['churn']==1]['tenure_months'].hist(ax=axes[0,1], bins=30, alpha=0.5, label='Churn', color='red')
        axes[0,1].set_title('Tenure Distribution')
        axes[0,1].legend()
        
        # Monthly charges
        df.boxplot(column='monthly_charges', by='churn', ax=axes[0,2])
        axes[0,2].set_title('Monthly Charges by Churn')
        
        # Support calls
        pd.crosstab(df['num_support_calls'], df['churn']).plot(ax=axes[1,0], kind='bar')
        axes[1,0].set_title('Churn by Number of Support Calls')
        
        # Age distribution
        df[df['churn']==0]['age'].hist(ax=axes[1,1], bins=20, alpha=0.5, label='No Churn', color='green')
        df[df['churn']==1]['age'].hist(ax=axes[1,1], bins=20, alpha=0.5, label='Churn', color='red')
        axes[1,1].set_title('Age Distribution')
        axes[1,1].legend()
        
        # Correlation heatmap
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numerical_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1,2])
        axes[1,2].set_title('Feature Correlations')
        
        plt.tight_layout()
        plt.savefig('churn_eda.png', dpi=300, bbox_inches='tight')
        print("\n📊 EDA visualizations saved as 'churn_eda.png'")
        
    def feature_engineering(self, df):
        """Create new features and prepare data"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        df_fe = df.copy()
        
        # Create new features
        df_fe['avg_monthly_charges'] = df_fe['total_charges'] / (df_fe['tenure_months'] + 1)
        df_fe['charges_to_tenure_ratio'] = df_fe['monthly_charges'] / (df_fe['tenure_months'] + 1)
        df_fe['is_new_customer'] = (df_fe['tenure_months'] < 6).astype(int)
        df_fe['high_spender'] = (df_fe['monthly_charges'] > df_fe['monthly_charges'].median()).astype(int)
        df_fe['high_support_usage'] = (df_fe['num_support_calls'] > 3).astype(int)
        df_fe['has_late_payments'] = (df_fe['num_late_payments'] > 0).astype(int)
        
        print("New features created:")
        print("- avg_monthly_charges")
        print("- charges_to_tenure_ratio")
        print("- is_new_customer")
        print("- high_spender")
        print("- high_support_usage")
        print("- has_late_payments")
        
        return df_fe
    
    def preprocess_data(self, df):
        """Encode categorical variables and scale features"""
        df_processed = df.copy()
        
        # Drop customer_id
        df_processed = df_processed.drop('customer_id', axis=1)
        
        # Encode categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
        
        return df_processed
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train and compare multiple models"""
        print("\n" + "="*60)
        print("MODEL TRAINING & EVALUATION")
        print("="*60)
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            results[name] = {
                'model': model,
                'auc': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"✓ AUC Score: {auc_score:.4f}")
            print(f"✓ Cross-Val AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Select best model
        best_model_name = max(results, key=lambda x: results[x]['auc'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   AUC Score: {results[best_model_name]['auc']:.4f}")
        
        return results, best_model_name
    
    def evaluate_model(self, X_test, y_test, results, best_model_name):
        """Detailed evaluation of the best model"""
        print("\n" + "="*60)
        print("DETAILED MODEL EVALUATION")
        print("="*60)
        
        y_pred = results[best_model_name]['y_pred']
        y_pred_proba = results[best_model_name]['y_pred_proba']
        
        print(f"\n{best_model_name} Performance:")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicted')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        axes[1].plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n📊 Evaluation plots saved as 'model_evaluation.png'")
        
        # Feature Importance (if applicable)
        if hasattr(self.best_model, 'feature_importances_'):
            self.plot_feature_importance(X_test)
    
    def plot_feature_importance(self, X):
        """Plot feature importance for tree-based models"""
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        plt.figure(figsize=(10, 6))
        plt.title('Top 15 Feature Importances')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [X.columns[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("📊 Feature importance saved as 'feature_importance.png'")
    
    def predict_churn_risk(self, customer_data):
        """Predict churn probability for new customers"""
        proba = self.best_model.predict_proba(customer_data)[:, 1]
        
        risk_levels = []
        for p in proba:
            if p < 0.3:
                risk_levels.append('Low Risk')
            elif p < 0.6:
                risk_levels.append('Medium Risk')
            else:
                risk_levels.append('High Risk')
        
        return proba, risk_levels

def main():
    print("🚀 Customer Churn Prediction System")
    print("="*60)
    
    # Initialize system
    system = ChurnPredictionSystem()
    
    # Generate data
    print("\n📊 Generating synthetic customer data...")
    df = system.generate_sample_data(n_samples=5000)
    
    # EDA
    system.exploratory_data_analysis(df)
    
    # Feature Engineering
    df = system.feature_engineering(df)
    
    # Preprocessing
    print("\n🔧 Preprocessing data...")
    df_processed = system.preprocess_data(df)
    
    # Split data
    X = df_processed.drop('churn', axis=1)
    y = df_processed['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    X_train_scaled = system.scaler.fit_transform(X_train)
    X_test_scaled = system.scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Train models
    results, best_model_name = system.train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Evaluate
    system.evaluate_model(X_test_scaled, y_test, results, best_model_name)
    
    # Example prediction
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    print("="*60)
    sample_customer = X_test_scaled.iloc[:3]
    probas, risks = system.predict_churn_risk(sample_customer)
    
    for i, (prob, risk) in enumerate(zip(probas, risks)):
        print(f"\nCustomer {i+1}:")
        print(f"  Churn Probability: {prob:.2%}")
        print(f"  Risk Level: {risk}")
    
    print("\n" + "="*60)
    print("✅ Analysis Complete!")
    print("="*60)
    print("\nKey Outputs:")
    print("1. churn_eda.png - Exploratory data analysis visualizations")
    print("2. model_evaluation.png - Model performance metrics")
    print("3. feature_importance.png - Most important features")
    
if __name__ == "__main__":
    main()
    