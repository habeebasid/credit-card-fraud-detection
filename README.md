# 💳 Credit Card Fraud Detection

An end-to-end machine learning project that detects fraudulent credit card transactions using advanced ML techniques to handle highly imbalanced data.

## 🎯 Project Overview

This project tackles the challenge of identifying fraudulent credit card transactions in a highly imbalanced dataset where fraud cases represent only 0.17% of all transactions. The system compares baseline and advanced machine learning models, implementing techniques like SMOTE to handle class imbalance.

### Key Features
- **Highly Imbalanced Dataset**: 99.83% legitimate vs 0.17% fraudulent transactions
- **Two Model Approaches**: Logistic Regression (baseline) vs XGBoost with SMOTE (advanced)
- **Interactive Web App**: Real-time fraud detection with Streamlit dashboard
- **Comprehensive Evaluation**: ROC-AUC, PR-AUC, Precision-Recall optimization

## 📊 Dataset

**Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- **Size**: 284,807 transactions
- **Features**: 30 (Time, V1-V28 PCA features, Amount)
- **Target**: Class (0: Legitimate, 1: Fraud)
- **Imbalance Ratio**: 1:577

⚠️ **Dataset NOT included in repository** (143 MB - exceeds GitHub limit)

### Download Dataset

**Method 1: Manual Download**
1. Visit [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place in `data/` folder

**Method 2: Kaggle API (Recommended)**
```bash
# Install Kaggle CLI
pip install kaggle

# Setup credentials (get from kaggle.com/settings/account)
# Place kaggle.json in ~/.kaggle/

# Download dataset
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/
unzip data/creditcardfraud.zip -d data/
rm data/creditcardfraud.zip
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- Conda or virtualenv

### Setup Steps
```bash
# 1. Clone repository
git clone https://github.com/habeebasid/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Create conda environment
conda create -n credit-fraud python=3.10 -y
conda activate credit-fraud

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset (see Dataset section above)

# 5. Launch Jupyter for exploration
jupyter notebook

# 6. Run notebooks in order:
#    - notebooks/eda_and_modeling.ipynb
#    - notebooks/model_experiments.ipynb (your experiments)
```

## 📁 Project Structure
```
credit-card-fraud-detection/
│
├── data/
│   ├── .gitkeep
│   └── creditcard.csv          # Download separately
│
├── notebooks/
│   ├── eda_and_modeling.ipynb  # Exploratory analysis
│   └── model_experiments.ipynb # Model experiments (rough work)
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data loading and preprocessing
│   ├── train_model.py          # Model training functions
│   └── predict.py              # Prediction utilities
│
├── models/
│   ├── logistic_regression_model.joblib
│   ├── xgboost_model.joblib
│   └── results/                # Evaluation metrics
│
├── app.py                      # Streamlit web application
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md
```

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Class distribution analysis
- Feature correlation study
- Transaction amount patterns
- Time-based fraud detection

### 2. Data Preprocessing
- Train-test split (80-20) with stratification
- Feature scaling (StandardScaler for Amount and Time)
- Handling class imbalance

### 3. Model Development

#### Baseline Model: Logistic Regression
- Simple, interpretable model
- Class weight balancing
- Fast training and inference

#### Advanced Model: XGBoost + SMOTE
- SMOTE oversampling for balanced training
- Gradient boosting for complex patterns
- Hyperparameter optimization
- Feature importance analysis

### 4. Evaluation Metrics
- **ROC-AUC**: Overall discrimination ability
- **PR-AUC**: Performance on imbalanced data
- **Recall**: Catching actual fraud (business priority)
- **Precision**: Minimizing false alarms
- **F1-Score**: Harmonic mean of precision and recall

## 📈 Results

| Model | ROC-AUC | PR-AUC | Recall (Fraud) | Precision (Fraud) | F1-Score |
|-------|---------|--------|----------------|-------------------|----------|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| XGBoost + SMOTE | TBD | TBD | TBD | TBD | TBD |

*Results will be updated after model training*

### Key Findings
- High recall achieved to minimize missed fraud cases
- SMOTE significantly improved fraud detection
- XGBoost captured non-linear patterns effectively
- Feature importance revealed key fraud indicators

## 🎨 Streamlit Web Application

Interactive dashboard for real-time fraud detection.
```bash
# Launch app
streamlit run app.py
```

**Features:**
- Single transaction prediction
- Batch prediction from CSV
- Model performance visualization
- Feature importance display
- Risk level assessment

## 💻 Usage Examples

### Train Models
```python
from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_baseline_model, train_xgboost_model

# Load and preprocess
df = load_data('data/creditcard.csv')
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# Train models
baseline_model = train_baseline_model(X_train, y_train)
xgb_model = train_xgboost_model(X_train, y_train, use_smote=True)
```

### Make Predictions
```python
from src.predict import load_model, predict_transaction

# Load model
model = load_model('models/xgboost_model.joblib')

# Predict
prediction, probability = predict_transaction(model, transaction_data)
print(f"Fraud Probability: {probability*100:.2f}%")
```

## 🔑 Key Learnings

1. **Handling Imbalanced Data**
   - SMOTE oversampling
   - Class weight adjustment
   - Appropriate metric selection

2. **Business Metric Focus**
   - Prioritizing recall (catching fraud)
   - Balancing false positives vs false negatives
   - Cost-sensitive learning

3. **Model Comparison**
   - Baseline vs advanced models
   - Interpretability vs performance tradeoff
   - Feature importance analysis

4. **Production Considerations**
   - Model deployment with Streamlit
   - Real-time prediction capability
   - User-friendly interface design

## 🚀 Future Improvements

- [ ] Deep Learning models (Neural Networks, Autoencoders)
- [ ] Real-time transaction monitoring
- [ ] Explainability with SHAP values
- [ ] A/B testing framework
- [ ] Model retraining pipeline
- [ ] API endpoint for predictions
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)

## 📚 Technologies Used

- **Python 3.10**
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn, XGBoost, Imbalanced-learn
- **Web App**: Streamlit
- **Version Control**: Git, GitHub
- **Environment**: Conda

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Your Name**
- GitHub: [@habeebasid](https://github.com/habeebasid)


## Acknowledgments

- Dataset: [Machine Learning Group - ULB](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Inspiration: Real-world fraud detection challenges
- Community: Kaggle and Stack Overflow contributors

---

⭐ **Star this repo** if you find it helpful!

📧 **Questions?** Feel free to open an issue or reach out.