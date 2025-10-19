# AI for Sustainable Development Goal 3: Health Classification

**Week 2 Assignment: Supervised Learning for Health Prediction**

A machine learning classifier that predicts whether a patient has a health condition based on routine vitals and clinical inputs, supporting SDG 3 (Good Health and Well-being) through early disease detection.

## 📊 Dataset

**Source**: Synthetic health dataset (1000 samples) generated with realistic correlations between clinical features and disease presence.

**Features**:
- `age`: Patient age (18-90 years)
- `sex`: Patient gender (Male/Female)
- `blood_pressure`: Systolic blood pressure (60-250 mmHg)
- `heart_rate`: Resting heart rate (30-200 bpm)
- `temperature`: Body temperature (90-110°F)
- `glucose`: Blood glucose level (50-400 mg/dL)
- `BMI`: Body Mass Index (10-60 kg/m²)
- `chest_pain`: Chest pain presence (Yes/No)
- `fatigue`: Fatigue symptoms (Yes/No)
- `diabetes`: Diabetes history (Yes/No)
- `hypertension`: Hypertension history (Yes/No)

**Target**: `disease_present` (0: No disease, 1: Disease present)

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Open the notebook: `notebooks/colab_notebook.ipynb`
2. Upload to Google Colab
3. Install requirements: `pip install pandas numpy scikit-learn matplotlib seaborn joblib`
4. Run all cells sequentially

### Option 2: Local Python Script
```bash
# Install requirements
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# Run the complete pipeline
python src/health_model.py
```

## 📈 Results Summary

**Model Performance** (to be filled after running):
- **Logistic Regression ROC AUC**: [To be computed]
- **Random Forest ROC AUC**: [To be computed]
- **Best Model**: [To be determined]
- **Accuracy**: [To be computed]
- **F1-Score**: [To be computed]

## 📁 Project Structure

```
AI-for-SDG3-Health/
├── data/
│   └── health_data.csv              # Health dataset
├── models/
│   ├── rf_health_model.joblib      # Trained Random Forest model
│   └── scaler.joblib                # Feature scaler
├── notebooks/
│   ├── colab_notebook.ipynb         # Google Colab notebook
│   └── figures/                     # Generated visualizations
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── feature_importance.png
├── src/
│   └── health_model.py             # Modular Python script
├── README.md                        # This file
├── article.md                       # Technical article
└── pitch_deck.txt                   # Presentation outline
```

## 🔬 Methodology

1. **Data Loading**: Attempts to download UCI Heart Disease dataset, falls back to synthetic data
2. **Preprocessing**: Handles missing values, clips impossible values, creates engineered features
3. **Feature Engineering**: One-hot encoding for categorical variables, pulse pressure calculation
4. **Model Training**: Logistic Regression (baseline) and Random Forest (100 trees, max_depth=10)
5. **Evaluation**: 5-fold cross-validation, comprehensive metrics (Accuracy, Precision, Recall, F1, ROC AUC)
6. **Visualization**: Confusion matrix, ROC curves, feature importance plots

## 🎯 SDG 3 Impact

This project directly supports **Sustainable Development Goal 3: Good Health and Well-being** by:

- **Early Detection**: Identifying patients at risk for health conditions
- **Preventive Care**: Enabling proactive healthcare interventions
- **Resource Optimization**: Helping healthcare systems prioritize high-risk patients
- **Health Equity**: Providing accessible health screening tools

## ⚖️ Ethics & Limitations

### Privacy Considerations
- Synthetic dataset protects patient privacy
- No real patient data used in this demonstration
- Real-world deployment requires proper data governance

### Bias Considerations
- Model performance may vary across demographic groups
- Training data should be representative of target population
- Regular bias audits recommended for production use

### Clinical Validation
- **Important**: This is a demonstration model, not a clinical tool
- Real-world deployment requires extensive clinical validation
- Should not replace professional medical judgment
- Requires regulatory approval for clinical use

### Limitations
- Synthetic data may not capture real-world complexity
- Model performance on synthetic data may not generalize
- Limited to the specific features included in training
- Requires continuous monitoring and retraining

## 🔧 Technical Requirements

- Python 3.7+
- pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
- Google Colab (for notebook execution)

## 📚 References

- [Sustainable Development Goals - SDG 3](https://www.un.org/sustainabledevelopment/health/)
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## 🤝 Contributing

This is a Week 2 assignment project. For questions or improvements, please refer to the course materials and instructor guidance.

---

**Disclaimer**: This project is for educational purposes only. The model is not intended for clinical use and should not replace professional medical advice or diagnosis.