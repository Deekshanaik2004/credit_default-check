#  Credit Default Prediction

A machine learning-powered web application for predicting credit default risk using XGBoost algorithm. This project provides both a web interface and API endpoints for credit risk assessment.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

##  Project Overview

This project implements a comprehensive credit default prediction system that helps financial institutions assess the risk of loan default. The system uses machine learning algorithms to analyze customer data and provide risk predictions with confidence scores.

### Key Features

-  **Machine Learning Model**: XGBoost-based prediction model
-  **Web Interface**: User-friendly Streamlit application
-  **Real-time Predictions**: Instant credit risk assessment
-  **API Endpoints**: RESTful API for integration
-  **Performance Metrics**: Model evaluation and validation
-  **Data Security**: Secure handling of sensitive financial data

##  Project Structure

```
credit_default/
├── app.py                 # Streamlit web application
├── train.py              # Model training script
├── cre.py                # Credit risk evaluation module
├── xgb_model.pkl         # Trained XGBoost model
├── credit_data_500.csv   # Sample credit dataset
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Git ignore rules
```

##  Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Deekshanaik2004/credit_default-check.git
   cd credit_default-check
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

##  Usage

### Training the Model

To train or retrain the XGBoost model:

```bash
python train.py
```

This will:
- Load the credit dataset
- Split data into training and testing sets
- Train the XGBoost classifier
- Save the model as `xgb_model.pkl`
- Display model performance metrics

### Running the Web Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Web Interface Features

- **Input Fields:**
  - Age (18-100 years)
  - Annual Income (₹10,000 - ₹200,000)
  - Credit Score (300-850)
  - Loan Amount (₹1,000 - ₹1,000,000)

- **Output:**
  - Risk Assessment (High/Low Risk)
  - Default Probability Score
  - Visual indicators and recommendations

##  API Usage

### Prediction Endpoint

**POST** `/predict`

**Request Body:**
```json
{
  "age": 35,
  "income": 75000,
  "credit_score": 720,
  "loan_amount": 150000
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.15,
  "risk_level": "Low Risk",
  "message": "Unlikely to Default"
}
```

##  Model Performance

The XGBoost model has been trained on credit data with the following performance metrics:

- **Accuracy**: ~85%
- **ROC-AUC Score**: ~0.89
- **Precision**: ~0.82
- **Recall**: ~0.78

##  Technical Details

### Dependencies

- **streamlit==1.28.1**: Web application framework
- **pandas==2.1.3**: Data manipulation and analysis
- **xgboost==2.0.2**: Gradient boosting framework
- **scikit-learn==1.3.2**: Machine learning utilities
- **joblib==1.3.2**: Model serialization
- **numpy==1.24.3**: Numerical computing

### Model Features

The model analyzes various credit-related features:

- **Demographic Data**: Age, income level
- **Credit History**: Credit score, payment history
- **Financial Indicators**: Debt-to-income ratio, loan amount
- **Behavioral Patterns**: Previous credit usage

##  Configuration

### Environment Variables

Create a `.env` file for custom configurations:

```env
MODEL_PATH=xgb_model.pkl
DATA_PATH=credit_data_500.csv
DEBUG_MODE=False
```

### Model Parameters

The XGBoost model uses the following default parameters:

```python
XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
```

##  Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Add tests** (if applicable)
5. **Commit your changes**
   ```bash
   git commit -m "Add: your feature description"
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request**

### Development Setup

For development, install additional dependencies:

```bash
pip install -r requirements-dev.txt
```


##  Acknowledgments

- XGBoost team for the excellent gradient boosting library
- Streamlit for the intuitive web framework
- Scikit-learn for machine learning utilities
- The open-source community for inspiration and support

##  Support

If you encounter any issues or have questions:

- **Create an issue** on GitHub
- **Email**: [dnaik1374@gmail.com]
- **Documentation**: Check the [Wiki](https://github.com/Deekshanaik2004/credit_default-check/wiki)

##  Future Enhancements

- [ ] Add more machine learning algorithms
- [ ] Implement model explainability (SHAP)
- [ ] Add data visualization dashboard
- [ ] Support for batch predictions
- [ ] Integration with external credit bureaus
- [ ] Real-time model retraining
- [ ] Mobile application
- [ ] Multi-language support

---

 **Star this repository if you find it helpful!**

**Made by Deeksha**
