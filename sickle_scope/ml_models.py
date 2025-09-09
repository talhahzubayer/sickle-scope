"""
SickleScope Machine Learning Models

Severity prediction models for sickle cell disease using Random Forest and other ML approaches.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path
import json
import warnings


class SickleMLPredictor:
    """Machine Learning predictor for sickle cell disease severity assessment."""
    
    def __init__(self, model_type: str = 'random_forest', verbose: bool = False):
        """Initialise the ML predictor.
        
        Args:
            model_type: Type of model to use ('random_forest', 'gradient_boost')
            verbose: Enable verbose logging
        """
        self.model_type = model_type
        self.verbose = verbose
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_importance = None
        self.training_features = []
        self.is_trained = False
        
        # Initialise model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        if self.verbose:
            print(f"Initialised {model_type} predictor")
    
    def _create_training_dataset(self, hbb_variants_db: Dict) -> pd.DataFrame:
        """Create training dataset from HBB variants database and literature sources.
        
        Args:
            hbb_variants_db: HBB variants database containing clinical data
            
        Returns:
            DataFrame with training features and labels
        """
        training_data = []
        
        # Extract pathogenic variants with clinical impact data
        pathogenic_variants = hbb_variants_db.get('pathogenic_variants', {})
        
        for variant_id, variant_data in pathogenic_variants.items():
            clinical_impact = variant_data.get('clinical_impact', {})
            
            # Generate samples for different genotype combinations
            for genotype, impact_data in clinical_impact.items():
                if isinstance(impact_data, dict) and 'severity' in impact_data:
                    # Extract features
                    features = {
                        'variant_id': variant_id,
                        'severity_score': variant_data.get('severity_score', 5),
                        'chromosome': variant_data.get('chromosome', 11),
                        'position': variant_data.get('position', 0),
                        'genotype_type': self._encode_genotype_type(genotype),
                        'disease_type': self._encode_disease_type(variant_data.get('disease', 'unknown')),
                        'beta_globin_production': self._get_beta_globin_production(variant_data),
                        'population_frequency': self._get_max_population_frequency(variant_data.get('frequency', {})),
                        'hbf_modifying_score': 0  # Default for pathogenic variants
                    }
                    
                    # Add target severity
                    severity = impact_data['severity']
                    severity_category = self._map_severity_to_category(severity)
                    
                    features['severity_category'] = severity_category
                    training_data.append(features)
        
        # Add modifier variants data
        modifier_variants = hbb_variants_db.get('modifier_variants', {})
        for gene, variants in modifier_variants.items():
            for variant_id, variant_data in variants.items():
                # Generate benign/protective samples
                features = {
                    'variant_id': variant_id,
                    'severity_score': abs(variant_data.get('modifier_score', -2)),
                    'chromosome': variant_data.get('chromosome', 2),
                    'position': variant_data.get('position', 0),
                    'genotype_type': 1,  # Heterozygous modifier
                    'disease_type': 0,   # Protective/benign
                    'beta_globin_production': 1.0,  # Normal production
                    'population_frequency': self._get_max_population_frequency(variant_data.get('frequency', {})),
                    'hbf_modifying_score': abs(variant_data.get('modifier_score', -2)),
                    'severity_category': 'mild'  # Protective effect
                }
                training_data.append(features)
        
        # Add synthetic normal/wild-type samples
        for i in range(50):  # Add normal samples for balance
            features = {
                'variant_id': 'normal_' + str(i),
                'severity_score': 0,
                'chromosome': 11,
                'position': 5227000 + i,  # Variants in HBB region
                'genotype_type': 0,  # Homozygous normal
                'disease_type': 0,   # Normal
                'beta_globin_production': 1.0,
                'population_frequency': 0.9,  # High frequency (normal)
                'hbf_modifying_score': 0,
                'severity_category': 'normal'
            }
            training_data.append(features)
        
        df = pd.DataFrame(training_data)
        
        if self.verbose:
            print(f"Created training dataset with {len(df)} samples")
            print(f"Severity distribution: {df['severity_category'].value_counts().to_dict()}")
        
        return df
    
    def _encode_genotype_type(self, genotype_desc: str) -> int:
        """Encode genotype type to numerical value."""
        mapping = {
            'homozygous': 2,
            'heterozygous': 1,
            'compound_het_with_hbs': 2,
            'compound_het_with_hbc': 2,
            'compound_het_with_beta_thal': 2,
            'compound_het_with_hbe': 2
        }
        return mapping.get(genotype_desc, 0)
    
    def _encode_disease_type(self, disease: str) -> int:
        """Encode disease type to numerical severity scale."""
        mapping = {
            'sickle_cell_disease': 4,
            'beta_zero_thalassemia': 4,
            'beta_plus_thalassemia': 3,
            'hemoglobin_c_disease': 2,
            'hemoglobin_e_disease': 2,
            'hemoglobin_d_disease': 1,
            'unknown': 1
        }
        return mapping.get(disease, 0)
    
    def _get_beta_globin_production(self, variant_data: Dict) -> float:
        """Extract beta globin production level from variant data."""
        return variant_data.get('beta_globin_production', 
               1.0 if variant_data.get('thalassemia_type') != 'beta_zero' else 0.0)
    
    def _get_max_population_frequency(self, freq_data: Dict) -> float:
        """Get maximum population frequency from frequency data."""
        if not freq_data:
            return 0.0
        return max([float(freq) for freq in freq_data.values() if isinstance(freq, (int, float))])
    
    def _map_severity_to_category(self, severity: str) -> str:
        """Map clinical severity descriptions to standard categories."""
        mapping = {
            'severe': 'severe',
            'most_severe': 'severe',
            'moderate': 'moderate',
            'moderate_to_severe': 'severe',
            'intermediate': 'moderate',
            'mild': 'mild',
            'asymptomatic': 'mild',
            'variable_mild_to_moderate': 'moderate'
        }
        return mapping.get(severity, 'moderate')
    
    def prepare_features(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features from analysis results for ML prediction.
        
        Args:
            results_df: Results DataFrame from variant analysis
            
        Returns:
            DataFrame with ML-ready features matching training format
        """
        features_df = results_df.copy()
        
        # Map analysis results to training feature format
        # Training features: ['severity_score', 'genotype_type', 'disease_type', 
        #                    'beta_globin_production', 'population_frequency', 'hbf_modifying_score']
        
        # severity_score: Use existing risk_score scaled to match training range
        features_df['severity_score'] = features_df.get('risk_score', 0) / 10.0  # Scale to 0-10 range
        
        # genotype_type: Map genotype to numerical values
        features_df['genotype_type'] = features_df['genotype'].map({
            '0/0': 0, '0|0': 0,  # Homozygous reference
            '0/1': 1, '0|1': 1,  # Heterozygous
            '1/1': 2, '1|1': 2   # Homozygous alternate
        }).fillna(0)
        
        # disease_type: Based on pathogenic status and severity
        features_df['disease_type'] = 0  # Default normal
        features_df.loc[features_df['is_pathogenic'], 'disease_type'] = 3  # Pathogenic variants
        features_df.loc[features_df['is_modifier'], 'disease_type'] = 1    # Modifier variants
        
        # beta_globin_production: Estimate based on variant type
        features_df['beta_globin_production'] = 1.0  # Default normal
        features_df.loc[features_df['is_pathogenic'], 'beta_globin_production'] = 0.5  # Reduced
        features_df.loc[features_df['genotype'] == '1/1', 'beta_globin_production'] = 0.2  # Very reduced for homozygous
        
        # population_frequency: Estimate based on variant classification
        features_df['population_frequency'] = 0.9  # Default high (normal variants)
        features_df.loc[features_df['is_pathogenic'], 'population_frequency'] = 0.01  # Low for pathogenic
        features_df.loc[features_df['is_modifier'], 'population_frequency'] = 0.15    # Moderate for modifiers
        
        # hbf_modifying_score: Based on modifier status
        features_df['hbf_modifying_score'] = 0.0  # Default no modification
        features_df.loc[features_df['is_modifier'], 'hbf_modifying_score'] = 2.0  # Modifier effect
        
        # Select training feature columns
        training_features = [
            'severity_score', 'genotype_type', 'disease_type', 
            'beta_globin_production', 'population_frequency', 'hbf_modifying_score'
        ]
        
        # Ensure all columns exist
        for col in training_features:
            if col not in features_df.columns:
                features_df[col] = 0.0
        
        return features_df[training_features]
    
    def train_model(self, hbb_variants_db: Dict) -> Dict:
        """Train the severity prediction model.
        
        Args:
            hbb_variants_db: HBB variants database for training data
            
        Returns:
            Dictionary with training results and metrics
        """
        if self.verbose:
            print("Creating training dataset...")
        
        # Create training dataset
        training_df = self._create_training_dataset(hbb_variants_db)
        
        # Prepare features and labels
        feature_columns = [
            'severity_score', 'genotype_type', 'disease_type', 
            'beta_globin_production', 'population_frequency', 'hbf_modifying_score'
        ]
        
        X = training_df[feature_columns]
        y = training_df['severity_category']
        
        self.training_features = feature_columns
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train model
        if self.verbose:
            print(f"Training {self.model_type} model...")
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_scaled, y_encoded, cv=5, scoring='accuracy'
        )
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                feature_columns, self.model.feature_importances_
            ))
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test)
        
        self.is_trained = True
        
        results = {
            'model_type': self.model_type,
            'training_samples': len(training_df),
            'feature_count': len(feature_columns),
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'classification_report': classification_report(
                y_test, y_pred, target_names=self.label_encoder.classes_, output_dict=True
            ),
            'feature_importance': self.feature_importance,
            'severity_categories': list(self.label_encoder.classes_)
        }
        
        if self.verbose:
            print(f"Model training completed:")
            print(f"  Training accuracy: {train_score:.3f}")
            print(f"  Test accuracy: {test_score:.3f}")
            print(f"  CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
            
            if self.feature_importance:
                print("  Top features:")
                sorted_features = sorted(self.feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)
                for feature, importance in sorted_features[:3]:
                    print(f"    {feature}: {importance:.3f}")
        
        return results
    
    def predict_severity(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict severity categories for given features.
        
        Args:
            features_df: DataFrame with prepared features
            
        Returns:
            DataFrame with predictions and confidence scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure features are in correct order
        expected_features = [
            'severity_score', 'genotype_type', 'disease_type', 
            'beta_globin_production', 'population_frequency', 'hbf_modifying_score'
        ]
        
        # Prepare features for prediction
        X = features_df[expected_features]
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Convert back to original labels
        predicted_categories = self.label_encoder.inverse_transform(predictions)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'predicted_severity': predicted_categories,
            'confidence_score': probabilities.max(axis=1)
        })
        
        # Add probability columns for each category
        for i, category in enumerate(self.label_encoder.classes_):
            results[f'prob_{category}'] = probabilities[:, i]
        
        return results
    
    def save_model(self, model_path: str) -> None:
        """Save trained model to file.
        
        Args:
            model_path: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_importance': self.feature_importance,
            'training_features': self.training_features,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, model_path)
        
        if self.verbose:
            print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load trained model from file.
        
        Args:
            model_path: Path to load model from
        """
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_importance = model_data.get('feature_importance')
        self.training_features = model_data.get('training_features', [])
        self.model_type = model_data.get('model_type', 'random_forest')
        self.is_trained = True
        
        if self.verbose:
            print(f"Model loaded from {model_path}")


class SeverityPredictor:
    """Wrapper class for easy integration with SickleAnalyser."""
    
    def __init__(self, verbose: bool = False):
        """Initialise severity predictor.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.ml_predictor = None
        self.is_initialised = False
    
    def initialise_model(self, hbb_variants_db: Dict) -> Dict:
        """Initialise and train the ML model.
        
        Args:
            hbb_variants_db: HBB variants database
            
        Returns:
            Training results dictionary
        """
        self.ml_predictor = SickleMLPredictor(verbose=self.verbose)
        training_results = self.ml_predictor.train_model(hbb_variants_db)
        self.is_initialised = True
        
        return training_results
    
    def predict_severity(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Predict severity for analysis results.
        
        Args:
            results_df: Analysis results DataFrame
            
        Returns:
            DataFrame with ML predictions
        """
        if not self.is_initialised:
            raise ValueError("Model must be initialised before making predictions")
        
        # Prepare features
        features_df = self.ml_predictor.prepare_features(results_df)
        
        # Make predictions
        predictions = self.ml_predictor.predict_severity(features_df)
        
        return predictions
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_initialised:
            return {'status': 'not_initialised'}
        
        return {
            'status': 'trained',
            'model_type': self.ml_predictor.model_type,
            'feature_importance': self.ml_predictor.feature_importance,
            'feature_count': len(self.ml_predictor.training_features),
            'severity_categories': list(self.ml_predictor.label_encoder.classes_) if self.ml_predictor.label_encoder else []
        }