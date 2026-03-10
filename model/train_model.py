import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from utils.preprocessing import clean_text

def train_best_model():
    print("🚀 Starting Model Training Pipeline...")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fake_path = os.path.join(base_dir, 'data', 'Fake.csv')
    true_path = os.path.join(base_dir, 'data', 'True.csv')
    
    # Load datasets
    print("📂 Loading datasets...")
    try:
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)
    except Exception as e:
        print(f"❌ Error loading CSV files: {e}")
        return

    # Add labels
    df_fake['label'] = 0
    df_true['label'] = 1
    
    # Combine datasets
    df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)
    
    # Create content column
    print("📝 Preparing data...")
    df['content'] = df['title'] + " " + df['text']
    
    # Take a subset if the dataset is too large (optimization for speed if needed)
    # df = df.sample(20000, random_state=42) 
    
    # Preprocess text
    print("🧹 Preprocessing text (this may take a while)...")
    df['content'] = df['content'].apply(clean_text)
    
    # Feature extraction tuning
    print("🔢 Vectorizing text using TF-IDF (Bigrams, Stopword tuning)...")
    tfidf = TfidfVectorizer(
        max_features=10000, 
        ngram_range=(1, 2), 
        stop_words='english',
        max_df=0.9,
        min_df=5
    )
    X = tfidf.fit_transform(df['content'])
    y = df['label']
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models and Hyperparameter Grids
    model_params = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=2000),
            "params": {
                "C": [0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"]
            }
        },
        "Linear SVM": {
            "model": LinearSVC(max_iter=2000),
            "params": {
                "C": [0.1, 1, 10]
            }
        },
        "Naive Bayes": {
            "model": MultinomialNB(),
            "params": {
                "alpha": [0.1, 0.5, 1.0]
            }
        }
    }
    
    best_acc = 0
    best_model = None
    best_name = ""
    performance_metrics = {}
    
    print("-" * 50)
    for name, mp in model_params.items():
        print(f"🤖 Tuning and Training {name}...")
        clf = GridSearchCV(mp['model'], mp['params'], cv=3, scoring='accuracy', n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # Get best estimator
        model = clf.best_estimator_
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        performance_metrics[name] = acc
        
        print(f"📊 {name} Performance (Best Params: {clf.best_params_}):")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print("-" * 30)
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    print(f"✅ Best Model: {best_name} (Accuracy: {best_acc:.4f})")
    
    # Save best model, vectorizer, and metadata
    model_dir = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(best_model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(tfidf, os.path.join(model_dir, 'vectorizer.pkl'))
    
    # Metadata for UI
    metadata = {
        "best_model_name": best_name,
        "best_accuracy": best_acc,
        "all_models": performance_metrics
    }
    joblib.dump(metadata, os.path.join(model_dir, 'model_metadata.pkl'))
    
    print(f"💾 Model and Metadata saved to {model_dir}")

if __name__ == "__main__":
    train_best_model()
