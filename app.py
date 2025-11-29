from diet_predictor import DietPredictor
from sample_data import generate_sample_data


def main():
    print("=" * 60)
    print("PREDICTION MODEL Z - Health Risk Assessment System")
    print("=" * 60)
    print("\nAnalyzing dietary habits and lifestyle factors...")
    print("Identifying harmful patterns for diabetes & high blood pressure\n")

    # Initialize and train the model
    print("Loading model...")
    predictor = DietPredictor()

    # Generate and train on sample data
    X, y = generate_sample_data(n_samples=1000)
    predictor.train(X, y)

    # Show feature importance
    print("\n" + "=" * 60)
    print("RISK FACTORS - Most Important to Least Important:")
    print("=" * 60)
    importance = predictor.get_feature_importance()
    for idx, row in importance.iterrows():
        feature_name = row['feature'].replace('_', ' ').title()
        importance_pct = row['importance'] * 100
        print(f"{feature_name:30s} {importance_pct:6.2f}%")

    # Interactive assessment
    print("\n" + "=" * 60)
    print("PERSONAL HEALTH RISK ASSESSMENT")
    print("=" * 60)

    try:
        print("\nEnter your daily dietary and lifestyle information:")

        sugar = float(input("Sugar intake (grams): "))
        salt = float(input("Salt intake (grams): "))
        fat = float(input("Fat intake (grams): "))
        sleep = float(input("Sleep hours: "))
        steps = float(input("Steps per day: "))
        water = float(input("Water intake (milliliters): "))

        user_data = {
            'sugar_intake_g': sugar,
            'salt_intake_g': salt,
            'fat_intake_g': fat,
            'sleep_hours': sleep,
            'steps_per_day': steps,
            'water_intake_ml': water
        }

        # Get risk assessment
        result = predictor.predict_risk(user_data)

        print("\n" + "=" * 60)
        print("RISK ASSESSMENT RESULTS")
        print("=" * 60)
        print(f"\nRisk Score: {result['risk_score']}%")
        print(f"Risk Level: {result['risk_level']}")

        # Color coding based on risk
        if result['risk_level'] == 'Low':
            print("\nStatus: You're doing well! Keep up the healthy habits.")
        elif result['risk_level'] == 'Moderate':
            print("\nStatus: Some improvements recommended to reduce risk.")
        else:
            print("\nStatus: High risk detected. Consider making changes soon.")

        # Get recommendations
        recommendations = predictor.get_recommendations(user_data)
        print("\n" + "=" * 60)
        print("PERSONALIZED RECOMMENDATIONS")
        print("=" * 60)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

        print("\n" + "=" * 60)
        print("Make healthier choices before problems develop!")
        print("=" * 60)

    except ValueError as e:
        print(f"\nError: Please enter valid numeric values")
    except KeyboardInterrupt:
        print("\n\nAssessment cancelled.")


if __name__ == "__main__":
    main()
