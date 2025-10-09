"""Feature engineering for materials using matminer."""
import numpy as np
import pandas as pd
from typing import Dict, List
from pymatgen.core.composition import Composition
from matminer.featurizers.composition import (
    ElementProperty,
    Stoichiometry,
    ValenceOrbital,
    IonProperty
)


class MaterialsFeatureEngineer:
    """Extract numerical features from chemical compositions."""
    
    def __init__(self):
        """Initialize featurizers."""
        # Element property statistics
        self.elem_prop = ElementProperty.from_preset("magpie")
        
        # Stoichiometric features
        self.stoich = Stoichiometry()
        
        # Valence orbital features
        self.valence = ValenceOrbital()
        
        # Ion property features
        self.ion_prop = IonProperty()
        
        self.feature_names = None
    
    def extract_features(self, formula: str) -> np.ndarray:
        """
        Extract numerical features from chemical formula.
        
        Args:
            formula: Chemical formula string
            
        Returns:
            1D numpy array of features
        """
        comp = Composition(formula)
        
        # Create a single-row dataframe for featurization
        df = pd.DataFrame({"composition": [comp]})
        
        # Apply all featurizers
        features = []
        
        # Element property features (most important)
        try:
            elem_features = self.elem_prop.featurize(comp)
            features.extend(elem_features)
        except Exception as e:
            print(f"Warning: ElementProperty featurization failed: {e}")
            features.extend([0.0] * len(self.elem_prop.feature_labels()))
        
        # Stoichiometric features
        try:
            stoich_features = self.stoich.featurize(comp)
            features.extend(stoich_features)
        except Exception as e:
            print(f"Warning: Stoichiometry featurization failed: {e}")
            features.extend([0.0] * len(self.stoich.feature_labels()))
        
        # Valence orbital features
        try:
            valence_features = self.valence.featurize(comp)
            features.extend(valence_features)
        except Exception as e:
            print(f"Warning: ValenceOrbital featurization failed: {e}")
            features.extend([0.0] * len(self.valence.feature_labels()))
        
        # Ion property features
        try:
            ion_features = self.ion_prop.featurize(comp)
            features.extend(ion_features)
        except Exception as e:
            print(f"Warning: IonProperty featurization failed: {e}")
            features.extend([0.0] * len(self.ion_prop.feature_labels()))
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        if self.feature_names is None:
            names = []
            names.extend(self.elem_prop.feature_labels())
            names.extend(self.stoich.feature_labels())
            names.extend(self.valence.feature_labels())
            names.extend(self.ion_prop.feature_labels())
            self.feature_names = names
        return self.feature_names
    
    def featurize_batch(self, formulas: List[str]) -> np.ndarray:
        """
        Extract features for multiple formulas.
        
        Args:
            formulas: List of chemical formula strings
            
        Returns:
            2D numpy array of shape (n_samples, n_features)
        """
        features_list = []
        for formula in formulas:
            try:
                features = self.extract_features(formula)
                features_list.append(features)
            except Exception as e:
                print(f"Error featurizing {formula}: {e}")
                # Use zero features as fallback
                features_list.append(np.zeros(len(self.get_feature_names())))
        
        return np.vstack(features_list)


class SimpleFeatureEngineer:
    """Simplified feature engineer for quick prototyping."""
    
    @staticmethod
    def extract_basic_features(formula: str) -> Dict[str, float]:
        """
        Extract basic compositional features.
        
        Args:
            formula: Chemical formula string
            
        Returns:
            Dictionary of basic features
        """
        comp = Composition(formula)
        
        features = {
            "num_elements": len(comp.elements),
            "avg_atomic_mass": np.mean([el.atomic_mass for el in comp.elements]),
            "avg_electronegativity": np.mean([el.X for el in comp.elements]),
            "avg_atomic_radius": np.mean([el.atomic_radius for el in comp.elements]),
            "total_atoms": sum(comp.values()),
        }
        
        return features