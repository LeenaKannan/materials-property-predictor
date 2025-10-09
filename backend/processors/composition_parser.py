"""Chemical composition parsing and validation."""
import re
from typing import Dict, Tuple
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element


class CompositionParser:
    """Parser for chemical formulas with validation."""
    
    SUPPORTED_ELEMENTS = set([el.symbol for el in Element])
    
    @staticmethod
    def parse_formula(formula: str) -> Dict[str, float]:
        """
        Parse chemical formula string to element dictionary.
        
        Args:
            formula: Chemical formula string (e.g., "SiO2", "Fe2O3")
            
        Returns:
            Dictionary mapping element symbols to fractional amounts
            
        Raises:
            ValueError: If formula is invalid or contains unsupported elements
        """
        if not formula or not formula.strip():
            raise ValueError("Formula cannot be empty")
        
        # Remove whitespace
        formula = formula.strip()
        
        try:
            # Use pymatgen's Composition class for robust parsing
            comp = Composition(formula)
            
            # Convert to fractional composition
            element_dict = comp.fractional_composition.as_dict()
            
            # Validate all elements are supported
            unsupported = set(element_dict.keys()) - CompositionParser.SUPPORTED_ELEMENTS
            if unsupported:
                raise ValueError(
                    f"Unsupported elements: {', '.join(unsupported)}. "
                    f"Please use standard element symbols."
                )
            
            return element_dict
            
        except Exception as e:
            if "unsupported" in str(e).lower():
                raise
            raise ValueError(
                f"Invalid chemical formula '{formula}'. "
                f"Please use standard chemical notation (e.g., 'SiO2', 'Fe2O3', 'CaTiO3'). "
                f"Error: {str(e)}"
            )
    
    @staticmethod
    def validate_formula(formula: str) -> Tuple[bool, str]:
        """
        Validate chemical formula without raising exceptions.
        
        Args:
            formula: Chemical formula string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            CompositionParser.parse_formula(formula)
            return True, ""
        except ValueError as e:
            return False, str(e)
    
    @staticmethod
    def get_element_amounts(formula: str) -> Dict[str, int]:
        """
        Get absolute element amounts from formula.
        
        Args:
            formula: Chemical formula string
            
        Returns:
            Dictionary mapping element symbols to integer amounts
        """
        comp = Composition(formula)
        return {str(el): int(amt) for el, amt in comp.items()}
    
    @staticmethod
    def normalize_formula(formula: str) -> str:
        """
        Normalize formula to reduced form.
        
        Args:
            formula: Chemical formula string
            
        Returns:
            Normalized formula string
        """
        comp = Composition(formula)
        return comp.reduced_formula


# Example usage and validation
def get_example_formulas() -> list:
    """Get list of example formulas for user guidance."""
    return [
        "SiO2",
        "Fe2O3",
        "CaTiO3",
        "Al2O3",
        "NaCl",
        "TiO2",
        "BaTiO3",
        "MgO",
        "ZnO",
        "Cu2O"
    ]