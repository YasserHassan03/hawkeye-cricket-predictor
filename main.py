import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

class CricketScorePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def parse_innings_data(self, innings_data):
        """Extract features from innings data"""
        try:
            features = []
            if not isinstance(innings_data, dict):
                raise ValueError(f"Expected innings_data to be a dictionary, got {type(innings_data)}")
                
            if 'overs' not in innings_data:
                raise ValueError("No 'overs' key found in innings data")
                
            overs = innings_data['overs']
            if not isinstance(overs, list):
                raise ValueError(f"Expected overs to be a list, got {type(overs)}")
                
            for over in overs:
                if not isinstance(over, dict):
                    raise ValueError(f"Expected over to be a dictionary, got {type(over)}")
                    
                over_num = over.get('over')
                deliveries = over.get('deliveries', [])
                
                if not isinstance(deliveries, list):
                    raise ValueError(f"Expected deliveries to be a list, got {type(deliveries)}")
                
                # Calculate over statistics
                runs_in_over = sum(d.get('runs', {}).get('total', 0) for d in deliveries)
                wickets_in_over = sum(1 for d in deliveries if 'wickets' in d)
                balls_in_over = len(deliveries)
                
                # Create feature row
                feature_row = {
                    'over_number': over_num,
                    'runs_scored': runs_in_over,
                    'wickets_lost': wickets_in_over,
                    'balls_bowled': balls_in_over,
                    'run_rate': runs_in_over / (balls_in_over/6) if balls_in_over > 0 else 0
                }
                features.append(feature_row)
            
            return pd.DataFrame(features)
        except Exception as e:
            print(f"Error in parse_innings_data: {str(e)}")
            raise
    
    def prepare_training_data(self, match_data):
        """Prepare training data from a single match"""
        try:
            all_features = []
            all_targets = []
            
            # Extract innings data from the match
            if 'innings' not in match_data:
                raise ValueError("No 'innings' key found in match data")
                
            for innings_idx, innings in enumerate(match_data['innings']):
                print(f"Processing innings {innings_idx}")
                innings_df = self.parse_innings_data(innings)
                
                # Calculate cumulative statistics
                innings_df['total_runs'] = innings_df['runs_scored'].cumsum()
                innings_df['total_wickets'] = innings_df['wickets_lost'].cumsum()
                innings_df['total_balls'] = innings_df['balls_bowled'].cumsum()
                innings_df['cumulative_run_rate'] = innings_df['total_runs'] / (innings_df['total_balls']/6)
                
                # Get final score for this innings
                final_score = innings_df['total_runs'].iloc[-1]
                
                # Create prediction points at each over
                for idx in range(len(innings_df)):
                    features = innings_df.iloc[idx].to_dict()
                    all_features.append(features)
                    all_targets.append(final_score)
            
            if not all_features:
                raise ValueError("No valid features extracted from the match data")
                
            X = pd.DataFrame(all_features)
            y = np.array(all_targets)
            
            return X, y
        except Exception as e:
            print(f"Error in prepare_training_data: {str(e)}")
            raise

    def train(self, X, y):
        """Train the prediction model with proper train/validation/test splits"""
        try:
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, 
                test_size=0.2,  # 20% for final testing
                random_state=42
            )
            
            # Second split: separate training and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=0.25,  # 25% of remaining 80% = 20% of total
                random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_pred = self.model.predict(X_val)
            val_metrics = {
                'val_mse': mean_squared_error(y_val, val_pred),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_r2': r2_score(y_val, val_pred)
            }
            
            # Evaluate on test set
            test_pred = self.model.predict(X_test)
            test_metrics = {
                'test_mse': mean_squared_error(y_test, test_pred),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'test_r2': r2_score(y_test, test_pred)
            }
            
            return {
                'validation_metrics': val_metrics,
                'test_metrics': test_metrics,
                'split_sizes': {
                    'train': len(X_train),
                    'validation': len(X_val),
                    'test': len(X_test)
                }
            }
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            raise
    def predict_final_score(self, current_innings_data):
        """Predict final score for an ongoing innings"""
        current_df = self.parse_innings_data(current_innings_data)
        
        # Calculate cumulative statistics
        current_df['total_runs'] = current_df['runs_scored'].cumsum()
        current_df['total_wickets'] = current_df['wickets_lost'].cumsum()
        current_df['total_balls'] = current_df['balls_bowled'].cumsum()
        current_df['cumulative_run_rate'] = current_df['total_runs'] / (current_df['total_balls']/6)
        
        # Make prediction using the latest state
        latest_state = current_df.iloc[-1:]
        predicted_score = self.model.predict(latest_state)[0]
        
        return predicted_score

def load_match_data(file_path):
    """Load match data from a JSON file"""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            print(f"Loaded data structure: {type(data)}")
            print(f"Found {len(data.get('innings', []))} innings in the match")
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in data file")

def main():
    # Load match data from file
    data_file_path = './1459381.json'
    try:
        match_data = load_match_data(data_file_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}")
        return

    # Initialize predictor
    predictor = CricketScorePredictor()
    
    # Prepare training data
    try:
        X, y = predictor.prepare_training_data(match_data)
        print(f"\nPrepared training data with {len(X)} samples")
    except Exception as e:
        print(f"Error preparing training data: {e}")
        return

    # Train model
    try:
        all_metrics = predictor.train(X, y)
        
        print("\nValidation Set Metrics:")
        val_metrics = all_metrics['validation_metrics']
        print(f"Mean Squared Error: {val_metrics['val_mse']:.2f}")
        print(f"Mean Absolute Error: {val_metrics['val_mae']:.2f}")
        print(f"R² Score: {val_metrics['val_r2']:.2f}")
        
        print("\nTest Set Metrics:")
        test_metrics = all_metrics['test_metrics']
        print(f"Mean Squared Error: {test_metrics['test_mse']:.2f}")
        print(f"Mean Absolute Error: {test_metrics['test_mae']:.2f}")
        print(f"R² Score: {test_metrics['test_r2']:.2f}")
        
        print("\nData Split Sizes:")
        splits = all_metrics['split_sizes']
        print(f"Training samples: {splits['train']}")
        print(f"Validation samples: {splits['validation']}")
        print(f"Test samples: {splits['test']}")
        
    except Exception as e:
        print(f"Error in training: {e}")
        return
    
    # Make prediction for the latest innings
    try:
        current_innings = match_data['innings'][-1]
        predicted_score = predictor.predict_final_score(current_innings)
        print(f"\nPredicted Final Score: {predicted_score:.0f}")
    except Exception as e:
        print(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()