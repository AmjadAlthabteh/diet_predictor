import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class DietPredictor:
    """
    Prediction Model Z - A machine learning system designed to estimate health risks
    based on dietary habits and lifestyle factors.
    """

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = [
            'sugar_intake_g',
            'salt_intake_g',
            'fat_intake_g',
            'sleep_hours',
            'steps_per_day',
            'water_intake_ml'
        ]
        self.is_trained = False

    def prepare_data(self, data):
        """Prepare and scale input data."""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        return self.scaler.transform(data[self.feature_names])

    def train(self, X, y):
        """
        Train the model on nutrition and health outcome data.

        Parameters:
        X: DataFrame with columns: sugar_intake_g, salt_intake_g, fat_intake_g,
           sleep_hours, steps_per_day, water_intake_ml
        y: Target labels (0: healthy, 1: at risk for diabetes/high blood pressure)
        """
        X_scaled = self.scaler.fit_transform(X[self.feature_names])
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print("Model trained successfully!")

    def predict_risk(self, dietary_data):
        """
        Calculate risk score showing the chance of health issues.

        Parameters:
        dietary_data: dict with keys matching feature_names

        Returns:
        dict with risk_score (probability) and risk_level
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_prepared = self.prepare_data(dietary_data)
        risk_probability = self.model.predict_proba(X_prepared)[0][1]

        if risk_probability < 0.3:
            risk_level = "Low"
        elif risk_probability < 0.6:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        return {
            'risk_score': round(risk_probability * 100, 2),
            'risk_level': risk_level
        }

    def get_feature_importance(self):
        """
        Highlight which factors raise the risk the most.

        Returns:
        DataFrame showing feature importance rankings
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing features")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df

    def get_recommendations(self, dietary_data):
        """
        Provide recommendations for healthier choices.

        Parameters:
        dietary_data: dict with current dietary habits

        Returns:
        list of recommendations
        """
        recommendations = []

        if dietary_data['sugar_intake_g'] > 50:
            recommendations.append("Reduce sugar intake - aim for less than 50g per day")

        if dietary_data['salt_intake_g'] > 6:
            recommendations.append("Lower salt consumption - target less than 6g per day")

        if dietary_data['fat_intake_g'] > 70:
            recommendations.append("Decrease fat intake - recommended limit is 70g per day")

        if dietary_data['sleep_hours'] < 7:
            recommendations.append("Increase sleep - aim for 7-9 hours per night")

        if dietary_data['steps_per_day'] < 8000:
            recommendations.append("Increase physical activity - target 8000-10000 steps daily")

        if dietary_data['water_intake_ml'] < 2000:
            recommendations.append("Drink more water - aim for at least 2000ml per day")

        if not recommendations:
            recommendations.append("Great job! Keep maintaining your healthy habits")

        return recommendations

    def save_model(self, filepath='diet_predictor_model.pkl'):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        joblib.dump({'model': self.model, 'scaler': self.scaler}, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='diet_predictor_model.pkl'):
        """Load a previously trained model."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from sample_data import generate_sample_data

    # Generate sample training data
    X, y = generate_sample_data(n_samples=1000)

    # Initialize and train model
    predictor = DietPredictor()
    predictor.train(X, y)

    # Show feature importance
    print("\nFeature Importance:")
    print(predictor.get_feature_importance())

    # Example prediction
    test_person = {
        'sugar_intake_g': 75,
        'salt_intake_g': 8,
        'fat_intake_g': 85,
        'sleep_hours': 5,
        'steps_per_day': 3000,
        'water_intake_ml': 1200
    }

    result = predictor.predict_risk(test_person)
    print(f"\nRisk Assessment:")
    print(f"Risk Score: {result['risk_score']}%")
    print(f"Risk Level: {result['risk_level']}")

    print("\nRecommendations:")
    for rec in predictor.get_recommendations(test_person):
        print(f"- {rec}")
