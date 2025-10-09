"""
Unit tests for chemical composition parsing and validation.
"""

import pytest
from backend.processors.composition_parser import (
    CompositionParser,
    ParsedComposition,
    CompositionParsingError,
    parse_formula,
    validate_formula,
    get_supported_elements
)


class TestCompositionParser:
    """Test cases for CompositionParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CompositionParser()
    
    def test_simple_formulas(self):
        """Test parsing of simple chemical formulas."""
        # Single element
        result = self.parser.parse("Si")
        assert result.elements == {"Si": 1.0}
        assert result.total_atoms == 1.0
        assert result.formula == "Si"
        
        # Element with count
        result = self.parser.parse("O2")
        assert result.elements == {"O": 2.0}
        assert result.total_atoms == 2.0
        
        # Binary compound
        result = self.parser.parse("SiO2")
        assert result.elements == {"Si": 1.0, "O": 2.0}
        assert result.total_atoms == 3.0
        
        # Complex compound
        result = self.parser.parse("Fe2O3")
        assert result.elements == {"Fe": 2.0, "O": 3.0}
        assert result.total_atoms == 5.0
    
    def test_decimal_counts(self):
        """Test parsing formulas with decimal counts."""
        result = self.parser.parse("Si0.5O2")
        assert result.elements == {"Si": 0.5, "O": 2.0}
        assert result.total_atoms == 2.5
        
        result = self.parser.parse("Ca1.5Al2.5O6")
        assert result.elements == {"Ca": 1.5, "Al": 2.5, "O": 6.0}
        assert result.total_atoms == 10.0
    
    def test_parentheses_formulas(self):
        """Test parsing formulas with parentheses."""
        # Simple parentheses
        result = self.parser.parse("Ca(OH)2")
        assert result.elements == {"Ca": 1.0, "O": 2.0, "H": 2.0}
        assert result.total_atoms == 5.0
        
        # Nested parentheses
        result = self.parser.parse("Mg3(PO4)2")
        assert result.elements == {"Mg": 3.0, "P": 2.0, "O": 8.0}
        assert result.total_atoms == 13.0
        
        # Decimal multiplier
        result = self.parser.parse("Ca(OH)1.5")
        assert result.elements == {"Ca": 1.0, "O": 1.5, "H": 1.5}
        assert result.total_atoms == 4.0
    
    def test_complex_formulas(self):
        """Test parsing of complex chemical formulas."""
        # Multiple element types
        result = self.parser.parse("CaCO3")
        assert result.elements == {"Ca": 1.0, "C": 1.0, "O": 3.0}
        assert result.total_atoms == 5.0
        
        # Mixed parentheses and regular elements
        result = self.parser.parse("Al2(SO4)3")
        assert result.elements == {"Al": 2.0, "S": 3.0, "O": 12.0}
        assert result.total_atoms == 17.0
        
        # Repeated elements
        result = self.parser.parse("CaOH2O")
        assert result.elements == {"Ca": 1.0, "O": 2.0, "H": 2.0}
        assert result.total_atoms == 5.0
    
    def test_whitespace_handling(self):
        """Test handling of whitespace in formulas."""
        result = self.parser.parse("  SiO2  ")
        assert result.elements == {"Si": 1.0, "O": 2.0}
        assert result.formula == "SiO2"
    
    def test_invalid_formulas(self):
        """Test error handling for invalid formulas."""
        # Empty formula
        with pytest.raises(CompositionParsingError, match="Formula must be a non-empty string"):
            self.parser.parse("")
        
        with pytest.raises(CompositionParsingError, match="Formula cannot be empty"):
            self.parser.parse("   ")
        
        # None input
        with pytest.raises(CompositionParsingError, match="Formula must be a non-empty string"):
            self.parser.parse(None)
        
        # Invalid characters
        with pytest.raises(CompositionParsingError, match="Formula contains invalid characters"):
            self.parser.parse("Si@O2")
        
        with pytest.raises(CompositionParsingError, match="Formula contains invalid characters"):
            self.parser.parse("SiO2!")
        
        # Unmatched parentheses
        with pytest.raises(CompositionParsingError, match="Unmatched parentheses"):
            self.parser.parse("Ca(OH2")
        
        with pytest.raises(CompositionParsingError, match="Unmatched parentheses"):
            self.parser.parse("CaOH)2")
        
        # Invalid element names
        with pytest.raises(CompositionParsingError, match="No valid elements found"):
            self.parser.parse("si02")  # lowercase element
        
        with pytest.raises(CompositionParsingError, match="No valid elements found"):
            self.parser.parse("abc123")  # invalid element names
        
        # Zero or negative counts
        with pytest.raises(CompositionParsingError, match="Invalid count"):
            self.parser.parse("Si0O2")
        
        with pytest.raises(CompositionParsingError, match="Formula contains invalid characters"):
            self.parser.parse("Si-1O2")
    
    def test_unsupported_elements(self):
        """Test error handling for unsupported elements."""
        # Single unsupported element
        with pytest.raises(CompositionParsingError, match="Unsupported elements: Xx"):
            self.parser.parse("XxO2")
        
        # Multiple unsupported elements
        with pytest.raises(CompositionParsingError, match="Unsupported elements"):
            self.parser.parse("XxYyO2")
    
    def test_supported_elements(self):
        """Test that common elements are supported."""
        supported = self.parser.get_supported_elements()
        
        # Check some common elements
        common_elements = {'H', 'C', 'N', 'O', 'Si', 'Al', 'Fe', 'Ca', 'Mg', 'Na', 'K'}
        assert common_elements.issubset(supported)
        
        # Test parsing with various supported elements
        test_formulas = [
            "H2O",
            "NaCl", 
            "CaCO3",
            "Al2O3",
            "SiO2",
            "Fe2O3",
            "MgSO4",
            "KNO3"
        ]
        
        for formula in test_formulas:
            result = self.parser.parse(formula)
            assert isinstance(result, ParsedComposition)
            assert result.total_atoms > 0
    
    def test_validate_formula_method(self):
        """Test the validate_formula method."""
        # Valid formulas
        is_valid, error = self.parser.validate_formula("SiO2")
        assert is_valid is True
        assert error == ""
        
        is_valid, error = self.parser.validate_formula("Ca(OH)2")
        assert is_valid is True
        assert error == ""
        
        # Invalid formulas
        is_valid, error = self.parser.validate_formula("XxO2")
        assert is_valid is False
        assert "Unsupported elements" in error
        
        is_valid, error = self.parser.validate_formula("")
        assert is_valid is False
        assert "non-empty string" in error
    
    def test_normalize_composition(self):
        """Test composition normalization to atomic fractions."""
        composition = {"Si": 1.0, "O": 2.0}
        normalized = self.parser.normalize_composition(composition)
        
        expected = {"Si": 1.0/3.0, "O": 2.0/3.0}
        assert abs(normalized["Si"] - expected["Si"]) < 1e-10
        assert abs(normalized["O"] - expected["O"]) < 1e-10
        assert abs(sum(normalized.values()) - 1.0) < 1e-10
        
        # Test with empty composition
        with pytest.raises(CompositionParsingError, match="Cannot normalize empty composition"):
            self.parser.normalize_composition({})
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single character elements
        result = self.parser.parse("H")
        assert result.elements == {"H": 1.0}
        
        # Two character elements
        result = self.parser.parse("Al")
        assert result.elements == {"Al": 1.0}
        
        # Large numbers
        result = self.parser.parse("Si1000O2000")
        assert result.elements == {"Si": 1000.0, "O": 2000.0}
        assert result.total_atoms == 3000.0
        
        # Very small decimal numbers
        result = self.parser.parse("Si0.001O2")
        assert result.elements == {"Si": 0.001, "O": 2.0}
        assert result.total_atoms == 2.001


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_parse_formula_function(self):
        """Test the parse_formula convenience function."""
        result = parse_formula("SiO2")
        assert isinstance(result, ParsedComposition)
        assert result.elements == {"Si": 1.0, "O": 2.0}
    
    def test_validate_formula_function(self):
        """Test the validate_formula convenience function."""
        is_valid, error = validate_formula("SiO2")
        assert is_valid is True
        assert error == ""
        
        is_valid, error = validate_formula("XxO2")
        assert is_valid is False
        assert "Unsupported elements" in error
    
    def test_get_supported_elements_function(self):
        """Test the get_supported_elements convenience function."""
        elements = get_supported_elements()
        assert isinstance(elements, set)
        assert len(elements) > 0
        assert "Si" in elements
        assert "O" in elements


class TestRealWorldFormulas:
    """Test with real-world chemical formulas."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CompositionParser()
    
    def test_common_materials(self):
        """Test parsing of common material formulas."""
        test_cases = [
            ("SiO2", {"Si": 1.0, "O": 2.0}),  # Quartz
            ("Al2O3", {"Al": 2.0, "O": 3.0}),  # Alumina
            ("Fe2O3", {"Fe": 2.0, "O": 3.0}),  # Iron oxide
            ("CaCO3", {"Ca": 1.0, "C": 1.0, "O": 3.0}),  # Calcite
            ("TiO2", {"Ti": 1.0, "O": 2.0}),  # Titanium dioxide
            ("ZnO", {"Zn": 1.0, "O": 1.0}),  # Zinc oxide
            ("MgO", {"Mg": 1.0, "O": 1.0}),  # Magnesia
            ("NaCl", {"Na": 1.0, "Cl": 1.0}),  # Salt
        ]
        
        for formula, expected in test_cases:
            result = self.parser.parse(formula)
            assert result.elements == expected, f"Failed for {formula}"
    
    def test_complex_materials(self):
        """Test parsing of complex material formulas."""
        test_cases = [
            ("Ca(OH)2", {"Ca": 1.0, "O": 2.0, "H": 2.0}),  # Calcium hydroxide
            ("Mg3(PO4)2", {"Mg": 3.0, "P": 2.0, "O": 8.0}),  # Magnesium phosphate
            ("Al2(SO4)3", {"Al": 2.0, "S": 3.0, "O": 12.0}),  # Aluminum sulfate
            ("Ca3(PO4)2", {"Ca": 3.0, "P": 2.0, "O": 8.0}),  # Calcium phosphate
        ]
        
        for formula, expected in test_cases:
            result = self.parser.parse(formula)
            assert result.elements == expected, f"Failed for {formula}"
    
    def test_perovskite_formulas(self):
        """Test parsing of perovskite-type formulas."""
        # These often have fractional compositions
        result = self.parser.parse("BaTiO3")
        assert result.elements == {"Ba": 1.0, "Ti": 1.0, "O": 3.0}
        
        # Test with decimal compositions (common in solid solutions)
        result = self.parser.parse("Ba0.5Sr0.5TiO3")
        expected = {"Ba": 0.5, "Sr": 0.5, "Ti": 1.0, "O": 3.0}
        assert result.elements == expected