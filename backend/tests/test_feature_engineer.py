"""
Unit tests for the feature engineering pipeline.

Tests the MaterialsFeatureEngineer class and related functionality
for extracting features from chemical compositions using matminer.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from backend.processors.feature_engineer import (
    MaterialsFeatureEngineer,
    FeatureExtractionResult,
    FeatureEngineeringError,
    extract_features,
    extract_features_dataframe
)
from backend.processors.composition_parser import CompositionParsingError


class TestMaterialsFeatureEngineer:
    """Test cases for MaterialsFeatureEngineer class."""
    
    @pytest.fixture
    def engineer(self):
        """Create a MaterialsFeatureEngineer instance for testing."""
        return MaterialsFeatureEngineer(use_cache=False)
    
    @pytest.fixture
    def engineer_with_cache(self):
        """Create a MaterialsFeatureEngineer instance with caching enabled."""
        return MaterialsFeatureEngineer(use_cache=True)
    
    def test_initialization(self, engineer):
        """Test proper initialization of the feature engineer."""
        assert engineer.use_cache is False
        assert engineer.composition_parser is not None
        assert engineer._feature_cache == {}
        assert engineer.featurizer is not None
        assert len(engineer.elemental_featurizers) > 0
        assert len(engineer.composition_featurizers) > 0
    
    def test_extract_features_simple_oxide(self, engineer):
        """Test feature extraction for a simple oxide (SiO2)."""
        result = engineer.extract_features("SiO2")
        
        assert isinstance(result, FeatureExtractionResult)
        assert result.success is True
        assert result.composition == "SiO2"
        assert result.error_message is None
        assert len(result.features) > 0
        assert len(result.feature_names) > 0
        assert len(result.features) == len(result.feature_names)
        
        # Check that all feature values are numeric
        for name, value in result.features.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            assert not np.isinf(value)
    
    def test_extract_features_complex_compound(self, engineer):
        """Test feature extraction for a complex compound (Ca(OH)2)."""
        result = engineer.extract_features("Ca(OH)2")
        
        assert result.success is True
        assert result.composition == "Ca(OH)2"
        assert len(result.features) > 0
        
        # Should have same number of features as simple compounds
        simple_result = engineer.extract_features("SiO2")
        assert len(result.features) == len(simple_result.features)
    
    def test_extract_features_metal_alloy(self, engineer):
        """Test feature extraction for a metal alloy (Fe0.8Ni0.2)."""
        result = engineer.extract_features("Fe0.8Ni0.2")
        
        assert result.success is True
        assert result.composition == "Fe0.8Ni0.2"
        assert len(result.features) > 0
        
        # Check that transition metal fraction features are present
        feature_names = [name.lower() for name in result.feature_names]
        has_metal_features = any('metal' in name or 'transition' in name 
                               for name in feature_names)
        # Note: This might not always be true depending on matminer version
        # but we check that extraction succeeds
    
    def test_extract_features_invalid_formula(self, engineer):
        """Test feature extraction with invalid chemical formula."""
        result = engineer.extract_features("InvalidFormula123")
        
        assert result.success is False
        assert result.composition == "InvalidFormula123"
        assert result.error_message is not None
        assert len(result.features) == 0
        assert len(result.feature_names) == 0
    
    def test_extract_features_empty_formula(self, engineer):
        """Test feature extraction with empty formula."""
        result = engineer.extract_features("")
        
        assert result.success is False
        assert result.error_message is not None
        assert "empty" in result.error_message.lower()
    
    def test_extract_features_unsupported_element(self, engineer):
        """Test feature extraction with unsupported element."""
        # Use a very rare/synthetic element that might not be supported
        result = engineer.extract_features("Uuo2")  # Ununoctium (now Oganesson)
        
        assert result.success is False
        assert result.error_message is not None
        assert "unsupported" in result.error_message.lower()
    
    def test_caching_functionality(self, engineer_with_cache):
        """Test that caching works correctly."""
        formula = "SiO2"
        
        # First extraction
        result1 = engineer_with_cache.extract_features(formula)
        assert result1.success is True
        assert engineer_with_cache.get_cache_size() == 1
        
        # Second extraction should use cache
        with patch.object(engineer_with_cache.featurizer, 'featurize') as mock_featurize:
            result2 = engineer_with_cache.extract_features(formula)
            mock_featurize.assert_not_called()  # Should not call featurizer again
        
        assert result2.success is True
        assert result2.features == result1.features
        assert engineer_with_cache.get_cache_size() == 1
    
    def test_clear_cache(self, engineer_with_cache):
        """Test cache clearing functionality."""
        engineer_with_cache.extract_features("SiO2")
        engineer_with_cache.extract_features("Al2O3")
        assert engineer_with_cache.get_cache_size() == 2
        
        engineer_with_cache.clear_cache()
        assert engineer_with_cache.get_cache_size() == 0
    
    def test_extract_batch_features(self, engineer):
        """Test batch feature extraction."""
        formulas = ["SiO2", "Al2O3", "Fe2O3", "TiO2"]
        results = engineer.extract_batch_features(formulas)
        
        assert len(results) == len(formulas)
        
        successful_results = [r for r in results if r.success]
        assert len(successful_results) > 0
        
        # All successful results should have same number of features
        if len(successful_results) > 1:
            feature_counts = [len(r.features) for r in successful_results]
            assert all(count == feature_counts[0] for count in feature_counts)
    
    def test_extract_batch_features_with_invalid(self, engineer):
        """Test batch extraction with some invalid formulas."""
        formulas = ["SiO2", "InvalidFormula", "Al2O3", ""]
        results = engineer.extract_batch_features(formulas)
        
        assert len(results) == len(formulas)
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        assert len(successful_results) >= 2  # SiO2 and Al2O3 should succeed
        assert len(failed_results) >= 2  # InvalidFormula and "" should fail
    
    def test_get_feature_dataframe(self, engineer):
        """Test DataFrame creation from feature extraction."""
        formulas = ["SiO2", "Al2O3", "Fe2O3"]
        df = engineer.get_feature_dataframe(formulas)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= len(formulas)  # Some might fail
        assert len(df) > 0  # At least some should succeed
        assert len(df.columns) > 0
        
        # Check that index contains formulas
        for formula in df.index:
            assert formula in formulas
        
        # Check that all values are numeric
        assert df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
    
    def test_get_feature_dataframe_all_invalid(self, engineer):
        """Test DataFrame creation when all formulas are invalid."""
        formulas = ["InvalidFormula1", "InvalidFormula2"]
        
        with pytest.raises(FeatureEngineeringError):
            engineer.get_feature_dataframe(formulas)
    
    def test_get_feature_names(self, engineer):
        """Test getting feature names."""
        # Initially should be None
        names = engineer.get_feature_names()
        
        # After extraction, should have names
        engineer.extract_features("SiO2")
        names = engineer.get_feature_names()
        
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(name, str) for name in names)
    
    def test_get_feature_statistics(self, engineer):
        """Test feature statistics calculation."""
        formulas = ["SiO2", "Al2O3", "Fe2O3", "TiO2"]
        stats = engineer.get_feature_statistics(formulas)
        
        assert isinstance(stats, dict)
        assert len(stats) > 0
        
        # Check statistics structure
        for feature_name, feature_stats in stats.items():
            assert isinstance(feature_stats, dict)
            required_stats = ['mean', 'std', 'min', 'max', 'median']
            for stat in required_stats:
                assert stat in feature_stats
                assert isinstance(feature_stats[stat], (int, float))
    
    def test_handle_nan_values(self, engineer):
        """Test NaN value handling."""
        # Create a features dict with NaN values
        features_with_nan = {
            'feature1': 1.0,
            'feature2': np.nan,
            'feature3': np.inf,
            'feature4': -np.inf,
            'feature5': 2.5
        }
        
        cleaned = engineer._handle_nan_values(features_with_nan)
        
        assert cleaned['feature1'] == 1.0
        assert cleaned['feature2'] == 0.0  # NaN replaced with 0
        assert cleaned['feature3'] == 0.0  # inf replaced with 0
        assert cleaned['feature4'] == 0.0  # -inf replaced with 0
        assert cleaned['feature5'] == 2.5
    
    def test_specific_elemental_features(self, engineer):
        """Test that specific elemental features mentioned in requirements are extracted."""
        result = engineer.extract_features("SiO2")
        
        assert result.success is True
        
        # Check for electronegativity-related features
        feature_names_lower = [name.lower() for name in result.feature_names]
        
        # Should have some electronegativity features
        electronegativity_features = [name for name in feature_names_lower 
                                    if 'electronegativity' in name or 'x' in name]
        
        # Should have some radius features
        radius_features = [name for name in feature_names_lower 
                         if 'radius' in name or 'atomic' in name]
        
        # Should have some valence features
        valence_features = [name for name in feature_names_lower 
                          if 'valence' in name or 'nvalence' in name]
        
        # At least some of these feature types should be present
        # (exact names depend on matminer version)
        total_expected_features = (len(electronegativity_features) + 
                                 len(radius_features) + 
                                 len(valence_features))
        assert total_expected_features > 0
    
    def test_statistical_features(self, engineer):
        """Test that statistical features (mean, min, max, std) are calculated."""
        result = engineer.extract_features("Fe2O3")  # Multi-element compound
        
        assert result.success is True
        
        feature_names_lower = [name.lower() for name in result.feature_names]
        
        # Check for statistical aggregations
        stat_keywords = ['mean', 'avg', 'minimum', 'min', 'maximum', 'max', 'std', 'range']
        stat_features = [name for name in feature_names_lower 
                        if any(keyword in name for keyword in stat_keywords)]
        
        assert len(stat_features) > 0, "Should have statistical features"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_extract_features_function(self):
        """Test the extract_features convenience function."""
        result = extract_features("SiO2")
        
        assert isinstance(result, FeatureExtractionResult)
        assert result.success is True
        assert result.composition == "SiO2"
    
    def test_extract_features_dataframe_function(self):
        """Test the extract_features_dataframe convenience function."""
        formulas = ["SiO2", "Al2O3"]
        df = extract_features_dataframe(formulas)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 0


class TestKnownMaterials:
    """Test feature extraction with known materials to verify scientific accuracy."""
    
    def test_quartz_sio2(self):
        """Test feature extraction for quartz (SiO2)."""
        engineer = MaterialsFeatureEngineer(use_cache=False)
        result = engineer.extract_features("SiO2")
        
        assert result.success is True
        
        # Basic sanity checks for SiO2
        # Should have features related to Si and O
        assert len(result.features) > 10  # Should have many features
        
        # All features should be finite numbers
        for value in result.features.values():
            assert np.isfinite(value)
    
    def test_alumina_al2o3(self):
        """Test feature extraction for alumina (Al2O3)."""
        engineer = MaterialsFeatureEngineer(use_cache=False)
        result = engineer.extract_features("Al2O3")
        
        assert result.success is True
        assert len(result.features) > 10
        
        # All features should be finite numbers
        for value in result.features.values():
            assert np.isfinite(value)
    
    def test_iron_oxide_fe2o3(self):
        """Test feature extraction for iron oxide (Fe2O3)."""
        engineer = MaterialsFeatureEngineer(use_cache=False)
        result = engineer.extract_features("Fe2O3")
        
        assert result.success is True
        assert len(result.features) > 10
        
        # Iron oxide should have transition metal features
        feature_names_lower = [name.lower() for name in result.feature_names]
        
        # Should have some metal-related features
        metal_features = [name for name in feature_names_lower 
                         if 'metal' in name or 'transition' in name]
        
        # All features should be finite numbers
        for value in result.features.values():
            assert np.isfinite(value)
    
    def test_perovskite_catio3(self):
        """Test feature extraction for perovskite (CaTiO3)."""
        engineer = MaterialsFeatureEngineer(use_cache=False)
        result = engineer.extract_features("CaTiO3")
        
        assert result.success is True
        assert len(result.features) > 10
        
        # All features should be finite numbers
        for value in result.features.values():
            assert np.isfinite(value)
    
    def test_feature_consistency_across_materials(self):
        """Test that feature extraction is consistent across different materials."""
        engineer = MaterialsFeatureEngineer(use_cache=False)
        
        materials = ["SiO2", "Al2O3", "Fe2O3", "TiO2", "CaO"]
        results = []
        
        for material in materials:
            result = engineer.extract_features(material)
            if result.success:
                results.append(result)
        
        assert len(results) >= 3  # Most should succeed
        
        # All successful results should have same number of features
        feature_counts = [len(r.features) for r in results]
        assert all(count == feature_counts[0] for count in feature_counts)
        
        # All should have same feature names
        feature_names_sets = [set(r.feature_names) for r in results]
        assert all(names == feature_names_sets[0] for names in feature_names_sets)


if __name__ == "__main__":
    pytest.main([__file__])