# Prediction - Diet & Health Risk Predictor

A machine learning system designed to estimate health risks based on dietary habits and lifestyle factors.

## Overview

Prediction Model Z analyzes a person's nutrition intake and lifestyle patterns to predict the risk of developing health issues like diabetes or high blood pressure. The system provides clear risk scores and actionable recommendations to help people make healthier choices before problems develop.

## Features

- **Dietary Analysis**: Tracks sugar, salt, and fat intake to understand eating habits
- **Lifestyle Monitoring**: Considers sleep hours, daily steps, and water consumption
- **Pattern Recognition**: Learns from real nutrition and health outcome data to identify harmful patterns
- **Risk Scoring**: Calculates a clear percentage showing the chance of health issues
- **Factor Importance**: Highlights which factors (high sugar, high salt, etc.) raise risk the most

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Assessment

Run the interactive health assessment app:

```bash
python app.py
```

Enter your daily dietary and lifestyle information when prompted to receive:
- Your personal risk score
- Risk level classification
- Customized health recommendations

### Using the Model Programmatically

```python
from diet_predictor import DietPredictor
from sample_data import generate_sample_data

# Initialize and train model
predictor = DietPredictor()
X, y = generate_sample_data(n_samples=1000)
predictor.train(X, y)

# Assess risk for a person
person_data = {
    'sugar_intake_g': 65,
    'salt_intake_g': 7.5,
    'fat_intake_g': 80,
    'sleep_hours': 6,
    'steps_per_day': 5000,
    'water_intake_ml': 1500
}

result = predictor.predict_risk(person_data)
print(f"Risk Score: {result['risk_score']}%")
print(f"Risk Level: {result['risk_level']}")

```

## Model Features

The model analyzes six key factors:

1. **Sugar Intake** (grams/day) - Recommended: < 50g
2. **Salt Intake** (grams/day) - Recommended: < 6g
3. **Fat Intake** (grams/day) - Recommended: < 70g
4. **Sleep Hours** (hours/night) - Recommended: 7-9 hours
5. **Steps Per Day** - Recommended: 8,000-10,000 steps
6. **Water Intake** (milliliters/day) - Recommended: > 2,000ml

