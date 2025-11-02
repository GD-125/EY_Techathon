# Dataset Links for Loan ERP System Training and Evaluation

This document provides comprehensive dataset links for training and evaluating machine learning models in the Loan ERP System.

## 📊 Recommended Datasets

### 1. **Home Credit Default Risk** (Kaggle)
- **URL**: https://www.kaggle.com/c/home-credit-default-risk/data
- **Size**: 300,000+ samples
- **Features**: 120+ features including:
  - Application data
  - Credit bureau data
  - Previous loan applications
  - Installment payments
  - Credit card balance
- **Target**: Loan default prediction
- **Use Case**: Primary dataset for credit risk modeling
- **Format**: CSV files
- **License**: Competition data (free to use for learning)

### 2. **German Credit Data** (UCI Machine Learning Repository)
- **URL**: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
- **Size**: 1,000 samples
- **Features**: 20 attributes including:
  - Credit history
  - Employment status
  - Personal status
  - Loan purpose
  - Housing status
- **Target**: Good/Bad credit classification
- **Use Case**: Classic benchmark for credit scoring models
- **Format**: CSV
- **License**: Public domain

### 3. **Lending Club Loan Data** (Kaggle)
- **URL**: https://www.kaggle.com/datasets/wordsforthewise/lending-club
- **Size**: 2.2 million+ samples (2007-2018)
- **Features**: 140+ features including:
  - Loan amount and interest rate
  - Borrower income and employment
  - Credit history metrics
  - Loan status and performance
- **Target**: Loan status (Fully Paid, Charged Off, Default)
- **Use Case**: Real-world P2P lending data analysis
- **Format**: CSV
- **License**: Public dataset

### 4. **Give Me Some Credit** (Kaggle Competition)
- **URL**: https://www.kaggle.com/c/GiveMeSomeCredit/data
- **Size**: 150,000 training samples
- **Features**: 10 features including:
  - Revolving utilization
  - Age
  - 30-60-90 days past due
  - Debt ratio
  - Monthly income
  - Number of open credit lines
- **Target**: Binary classification (serious delinquency)
- **Use Case**: Credit scoring and risk assessment
- **Format**: CSV
- **License**: Competition data

### 5. **Default of Credit Card Clients** (UCI)
- **URL**: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
- **Size**: 30,000 samples
- **Features**: 24 features including:
  - Credit limit
  - Gender, education, marital status
  - Age
  - Payment history (6 months)
  - Bill statements
  - Previous payments
- **Target**: Default payment next month (binary)
- **Use Case**: Credit card default prediction
- **Format**: XLS/CSV
- **License**: Public domain

### 6. **LendingClub Accepted and Rejected Loans** (Kaggle)
- **URL**: https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv
- **Size**: Multiple datasets (2007-2018)
- **Features**: Comprehensive loan application data
- **Use Case**: Analyzing both approved and rejected loans
- **Format**: CSV
- **License**: Public dataset

### 7. **Prosper Loan Data** (Kaggle)
- **URL**: https://www.kaggle.com/datasets/nurudeenabdulsalaam/prosper-loan-dataset
- **Size**: 113,937 loans
- **Features**: 81 variables including:
  - Borrower rate
  - Prosper rating
  - Listing category
  - Income range
  - Employment status
- **Use Case**: P2P lending analysis
- **Format**: CSV
- **License**: Public dataset

### 8. **HELOC Dataset** (FICO Community)
- **URL**: https://community.fico.com/s/explainable-machine-learning-challenge
- **Size**: 10,000+ samples
- **Features**: 23 features (anonymized)
- **Target**: Risk performance (Good/Bad)
- **Use Case**: Explainable AI for credit decisions
- **Format**: CSV
- **License**: FICO Community License

## 🎯 Dataset Selection Guide

### For Initial Prototyping:
- **Start with**: German Credit Data (small, well-documented)
- **Reason**: Easy to understand, quick to train, good for testing pipeline

### For Production Models:
- **Use**: Home Credit Default Risk or Lending Club
- **Reason**: Large sample size, comprehensive features, real-world data

### For Explainability Focus:
- **Use**: HELOC Dataset
- **Reason**: Designed specifically for explainable AI applications

### For Benchmarking:
- **Use**: Give Me Some Credit
- **Reason**: Well-established benchmark with community solutions

## 📥 How to Use These Datasets

### 1. Download the Dataset
```bash
# Example using Kaggle API
kaggle competitions download -c home-credit-default-risk
```

### 2. Load into the System
- Use the **Data Upload** feature in the admin panel
- Upload CSV or XLSX files
- System will automatically:
  - Validate data quality
  - Analyze features
  - Generate predictions with explainability
  - Calculate model metrics

### 3. Expected Data Format
The system expects these columns (minimum):
- `application_id` or `id` - Unique identifier
- `annual_income` or `income` - Applicant income
- `loan_amount` - Requested loan amount
- `credit_score` - Credit score (300-850 range)
- `loan_status` - Target variable (for training)

Optional but recommended:
- `age`
- `employment_months`
- `existing_debt`
- `payment_history_score`
- `education_level`
- `marital_status`

## 🔧 Data Preprocessing Tips

### 1. Handle Missing Values
```python
# The system automatically handles missing values
# Numeric: Median imputation
# Categorical: Mode imputation
```

### 2. Feature Engineering
The system automatically creates:
- `loan_to_income_ratio`
- `debt_to_income_ratio`
- Encoded categorical variables

### 3. Data Quality Checks
- Validates credit scores (300-850 range)
- Checks for negative income values
- Detects duplicates
- Calculates data quality score

## 📊 Model Training Recommendations

### Train-Test Split
- Training: 70-80%
- Validation: 10-15%
- Test: 10-15%

### Cross-Validation
- Use 5-fold or 10-fold cross-validation
- Stratified sampling for imbalanced datasets

### Evaluation Metrics
The system calculates:
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

## 🎓 Learning Resources

### Tutorials
- [Kaggle Learn - Credit Risk](https://www.kaggle.com/learn/credit-risk-model)
- [UCI ML Repository Guides](https://archive.ics.uci.edu/ml/index.php)

### Papers
- "Credit Scoring with Machine Learning: A Review" (2020)
- "Explainable AI for Credit Risk Assessment" (2021)

### Books
- "Credit Risk Analytics" by Bart Baesens
- "Machine Learning for Credit Risk" by Iain Brown

## ⚠️ Important Notes

### Data Privacy
- Never use real customer data without proper authorization
- Anonymize sensitive information
- Comply with GDPR, CCPA, and other regulations

### Bias Consideration
- Check for demographic bias in models
- Use fairness metrics
- Implement bias mitigation techniques

### Model Validation
- Always validate on out-of-sample data
- Monitor model performance over time
- Retrain periodically with new data

## 🔄 Data Update Frequency

### Recommended Schedule:
- **Daily**: Process new applications
- **Weekly**: Update model performance metrics
- **Monthly**: Retrain models with recent data
- **Quarterly**: Full model review and validation
- **Annually**: Complete model rebuild if needed

## 📞 Support

For questions about datasets or data processing:
- Check the [QUICKSTART.md](./QUICKSTART.md) guide
- Review [COMPLETE_DOCUMENTATION.md](./docs/COMPLETE_DOCUMENTATION.md)
- Submit issues on GitHub

---

**Last Updated**: 2025-10-30
**Version**: 1.0.0
