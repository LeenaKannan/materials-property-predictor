"""Unit tests for composition parser."""
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.processors.composition_parser import CompositionParser


class TestCompositionParser:
    """Test cases for CompositionParser."""
    
    def test_simple_formula(self):
        """Test parsing simple formulas."""
        result = CompositionParser.parse_formula("SiO2")
        assert "Si" in result
        assert "O" in result
        assert result["Si"] == pytest.approx(0.333, abs=0.01)
        assert result["O"] == pytest.approx(0.667, abs=0.01)
    
    def test_complex_formula(self):
        """Test parsing complex formulas."""
        result = CompositionParser.parse_formula("Fe2O3")
        assert "Fe" in result
        assert "O" in result
        assert result["Fe"] == pytest.approx(0.4, abs=0.01)
        assert result["O"] == pytest.approx(0.6, abs=0.01)
    
    def test_perovskite_formula(self):
        """Test parsing perovskite formula."""
        result = CompositionParser.parse_formula("CaTiO3")
        assert "Ca" in result
        assert "Ti" in result
        assert "O" in result
        assert len(result) == 3
    
    def test_single_element(self):
        """Test parsing single element."""
        result = CompositionParser.parse_formula("Fe")
        assert "Fe" in result
        assert result["Fe"] == 1.0
    
    def test_invalid_formula_empty(self):
        """Test empty formula raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            CompositionParser.parse_formula("")
    
    def test_invalid_formula_syntax(self):
        """Test invalid syntax raises error."""
        with pytest.raises(ValueError, match="Invalid chemical formula"):
            CompositionParser.parse_formula("Si@O2")
    
    def test_unsupported_element(self):
        """Test unsupported element raises error."""
        with pytest.raises(ValueError, match="Unsupported elements"):
            CompositionParser.parse_formula("XyZ2")
    
    def test_validate_formula_valid(self):
        """Test validation of valid formula."""
        is_valid, msg = CompositionParser.validate_formula("SiO2")
        assert is_valid
        assert msg == ""
    
    def test_validate_formula_invalid(self):
        """Test validation of invalid formula."""
        is_valid, msg = CompositionParser.validate_formula("InvalidFormula123")
        assert not is_valid
        assert len(msg) > 0
    
    def test_normalize_formula(self):
        """Test formula normalization."""
        normalized = CompositionParser.normalize_formula("Si2O4")
        assert normalized == "SiO2"
    
    def test_get_element_amounts(self):
        """Test getting absolute element amounts."""
        amounts = CompositionParser.get_element_amounts("Fe2O3")
        assert amounts["Fe"] == 2
        assert amounts["O"] == 3
    
    def test_whitespace_handling(self):
        """Test handling of whitespace."""
        result1 = CompositionParser.parse_formula("SiO2")
        result2 = CompositionParser.parse_formula("  SiO2  ")
        assert result1 == result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])