import pickle
import pandas as pd

class NoShowPredictor:
    def __init__(self):
        """
        Initializes the predictor, loading the pre-trained model.
        """
        self.model = self.load_model('trained_model.pkl')

    def load_model(self, model_path):
        """
        Loads the serialized machine learning model from a file.
        """
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            # For now, return None if the model isn't found.
            # In a real scenario, you might raise an error or handle this differently.
            print("Warning: Model file not found. Predictor will not work.")
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:

        """
        Makes predictions on a batch of appointments (a DataFrame).
        """

        if self.model is None:
        # This is an important check to ensure the model loaded correctly.
            raise FileNotFoundError("Model file not found or failed to load.")
    
    # --- 1. Preprocess the incoming data ---
    # This must be IDENTICAL to the preprocessing in your notebook.
    
    # Example: Convert gender to numeric format
        df_processed = features_df.copy()
        if 'gender' in df_processed.columns:
            df_processed['gender_encoded'] = df_processed['gender'].map({'M': 1, 'F': 0})
    
    # --- 2. Select the final feature columns ---
    # The model expects the columns in a specific order. Get it from the model itself.
    # This makes your code robust even if you retrain the model with different features.
        model_features = self.model.feature_names_in_
    
    # Make sure all required columns are present
        for col in model_features:
            if col not in df_processed.columns:
                raise ValueError(f"Missing required column: {col}")
            
    # Select and reorder columns to match the model's expectation
        data_for_prediction = df_processed[model_features]

    # --- 3. Make predictions ---
    # Use predict_proba to get the probability of a "no-show" (class 1)
        probabilities = self.model.predict_proba(data_for_prediction)[:, 1]
    
    # --- 4. Format the output ---
    # Create a new DataFrame with the results
        result_df = pd.DataFrame()
        result_df['risk_score'] = probabilities
        result_df['risk_level'] = result_df['risk_score'].apply(self.calculate_risk_level)
        result_df['predicted_outcome'] = (result_df['risk_score'] >= 0.5).astype(int)
    
        return result_df

    def calculate_risk_level(self, probability):
        """Converts a probability score to a categorical risk level."""
        if probability < 0.25:
            return "Low"
        elif probability < 0.60:
            return "Medium"
        else:
            return "High"