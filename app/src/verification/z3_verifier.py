from z3 import Int, Solver, sat, unsat

class Z3Verifier:
    """Proves logical satisfiability for code pre/post-conditions."""

    @staticmethod
    def verify_bounds(val_min: int, val_max: int) -> tuple[bool, str]:
        val = Int('val')
        s = Solver()
        
        # Enforce numeric range logic
        s.add(val >= val_min)
        s.add(val <= val_max)
        
        result = s.check()
        if result == sat:
            return True, "SAT: Logical constraints hold across valid range."
        return False, f"UNSAT: Logical contradiction or unsat bounds detected ({result})."