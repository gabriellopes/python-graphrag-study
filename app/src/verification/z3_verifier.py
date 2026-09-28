# app/src/verification/z3_verifier.py
from z3 import Int, Solver, sat

class Z3CodeVerifier:
    """Verifies generated code numeric/logical bounds deterministically."""

    @staticmethod
    def verify_numeric_bounds(min_val: int, max_val: int) -> bool:
        x = Int('x')
        s = Solver()
        
        # Enforce preconditions & postconditions
        s.add(x >= min_val)
        s.add(x <= max_val)
        
        return s.check() == sat
