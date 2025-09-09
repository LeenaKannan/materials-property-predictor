
# Chemical composition parsing and validation module.
"""
This module provides functionality to parse chemical formulas into element dictionaries
and validate formula correctness and element support.
"""

import re
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass


@dataclass
class ParsedComposition:
    # Represents a parsed chemical composition.
    elements: Dict[str, float]
    total_atoms: float
    formula: str


class CompositionParsingError(Exception):
    # Exception raised when composition parsing fails.
    pass


class CompositionParser:
    # Parser for chemical composition formulas.
    
    # Supported elements (common elements in materials science)
    SUPPORTED_ELEMENTS = {
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm'
    }
    
    def __init__(self):
        # Initialize the composition parser.
        
        # Regex pattern to match element-count pairs
        # Matches: Element (uppercase + optional lowercase) followed by optional number
        self.element_pattern = re.compile(r'([A-Z][a-z]?)(\d*\.?\d*)')
        
        # Pattern to match parentheses with multipliers
        self.parentheses_pattern = re.compile(r'\(([^)]+)\)(\d*\.?\d*)')
    
    def parse(self, formula: str) -> ParsedComposition:
        """
        Parse a chemical formula string into element composition.
        
        Args:
            formula: Chemical formula string (e.g., "SiO2", "Ca(OH)2")
            
        Returns:
            ParsedComposition object with element dictionary and metadata
            
        Raises:
            CompositionParsingError: If formula is invalid or contains unsupported elements
        """
        if not formula or not isinstance(formula, str):
            raise CompositionParsingError("Formula must be a non-empty string")
        
        # Remove whitespace and validate basic format
        formula = formula.strip()
        if not formula:
            raise CompositionParsingError("Formula cannot be empty")
        
        # Check for invalid characters
        if not re.match(r'^[A-Za-z0-9().]+$', formula):
            raise CompositionParsingError(
                f"Formula contains invalid characters: {formula}"
            )
        
        # Check for unmatched parentheses
        if formula.count('(') != formula.count(')'):
            raise CompositionParsingError(f"Unmatched parentheses in formula: {formula}")
        
        # Check for invalid parentheses patterns (closing before opening)
        paren_count = 0
        for char in formula:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count < 0:
                    raise CompositionParsingError(f"Unmatched parentheses in formula: {formula}")
        
        try:
            # Handle parentheses first
            expanded_formula = self._expand_parentheses(formula)
            
            # Parse element-count pairs
            elements = self._parse_elements(expanded_formula)
            
            # Validate elements are supported
            self._validate_elements(elements.keys())
            
            # Calculate total atoms
            total_atoms = sum(elements.values())
            
            return ParsedComposition(
                elements=elements,
                total_atoms=total_atoms,
                formula=formula
            )
            
        except Exception as e:
            if isinstance(e, CompositionParsingError):
                raise
            raise CompositionParsingError(f"Failed to parse formula '{formula}': {str(e)}")
    
    def _expand_parentheses(self, formula: str) -> str:
        """
        Expand parentheses in chemical formulas.
        
        Args:
            formula: Formula with potential parentheses
            
        Returns:
            Formula with parentheses expanded
        """
        # Keep expanding until no more parentheses
        while '(' in formula:
            match = self.parentheses_pattern.search(formula)
            if not match:
                raise CompositionParsingError(f"Unmatched parentheses in formula: {formula}")
            
            group_content = match.group(1)
            multiplier = match.group(2)
            multiplier = float(multiplier) if multiplier else 1.0
            
            # Parse elements within parentheses
            group_elements = self._parse_elements(group_content)
            
            # Multiply by the group multiplier
            expanded_part = ""
            for element, count in group_elements.items():
                new_count = count * multiplier
                if new_count == int(new_count):
                    new_count = int(new_count)
                expanded_part += f"{element}{new_count if new_count != 1 else ''}"
            
            # Replace the parentheses group with expanded form
            formula = formula[:match.start()] + expanded_part + formula[match.end():]
        
        return formula
    
    def _parse_elements(self, formula: str) -> Dict[str, float]:
        """
        Parse element-count pairs from a formula string.
        
        Args:
            formula: Formula string without parentheses
            
        Returns:
            Dictionary mapping element symbols to counts
        """
        elements = {}
        
        # Find all element-count matches
        matches = self.element_pattern.findall(formula)
        
        if not matches:
            raise CompositionParsingError(f"No valid elements found in formula: {formula}")
        
        # Check if entire formula was matched
        matched_length = sum(len(match[0]) + len(match[1]) for match in matches)
        if matched_length != len(formula):
            raise CompositionParsingError(f"Invalid formula format: {formula}")
        
        for element, count_str in matches:
            # Default count is 1 if not specified
            count = float(count_str) if count_str else 1.0
            
            if count <= 0:
                raise CompositionParsingError(f"Invalid count for element {element}: {count}")
            
            # Add to existing count if element appears multiple times
            elements[element] = elements.get(element, 0.0) + count
        
        return elements
    
    def _validate_elements(self, elements: Set[str]) -> None:
        """
        Validate that all elements are supported.
        
        Args:
            elements: Set of element symbols to validate
            
        Raises:
            CompositionParsingError: If any element is not supported
        """
        unsupported = elements - self.SUPPORTED_ELEMENTS
        if unsupported:
            raise CompositionParsingError(
                f"Unsupported elements: {', '.join(sorted(unsupported))}. "
                f"Supported elements: {', '.join(sorted(self.SUPPORTED_ELEMENTS))}"
            )
    
    def validate_formula(self, formula: str) -> Tuple[bool, str]:
        """
        Validate a chemical formula without parsing.
        
        Args:
            formula: Chemical formula string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            self.parse(formula)
            return True, ""
        except CompositionParsingError as e:
            return False, str(e)
    
    def get_supported_elements(self) -> Set[str]:
        """
        Get the set of supported element symbols.
        
        Returns:
            Set of supported element symbols
        """
        return self.SUPPORTED_ELEMENTS.copy()
    
    def normalize_composition(self, composition: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize composition to atomic fractions (sum = 1.0).
        
        Args:
            composition: Dictionary of element symbols to counts
            
        Returns:
            Dictionary of element symbols to atomic fractions
        """
        total = sum(composition.values())
        if total == 0:
            raise CompositionParsingError("Cannot normalize empty composition")
        
        return {element: count / total for element, count in composition.items()}


# Convenience functions for common operations
def parse_formula(formula: str) -> ParsedComposition:
    """Parse a chemical formula using the default parser."""
    parser = CompositionParser()
    return parser.parse(formula)


def validate_formula(formula: str) -> Tuple[bool, str]:
    """Validate a chemical formula using the default parser."""
    parser = CompositionParser()
    return parser.validate_formula(formula)


def get_supported_elements() -> Set[str]:
    """Get supported elements using the default parser."""
    parser = CompositionParser()
    return parser.get_supported_elements()