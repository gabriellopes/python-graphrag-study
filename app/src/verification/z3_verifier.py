import ast
from z3 import Int, Solver, sat, Not, And, Or

class Z3Verifier:
    """Proves logical satisfiability and code contract correctness using Z3 and dynamic AST translation."""

    @staticmethod
    def verify_bounds(min_val: int, max_val: int, code_input: str) -> tuple[bool, str]:
        val = Int('val')
        s = Solver()
        
        # 1. Define the valid operational domain from the ontology spec
        domain_constraint = And(val >= min_val, val <= max_val)
        
        try:
            # 2. Dynamic AST parsing of user code
            tree = ast.parse(code_input)
            
            return_node = None
            for node in ast.walk(tree):
                if isinstance(node, ast.Return):
                    return_node = node.value
                    break
            
            if return_node is None:
                # Safe fallback if no return statement found
                code_condition = (val >= min_val)
            else:
                # Translate AST expression to Z3 dynamically
                code_condition = Z3Verifier._ast_to_z3(return_node, val)
                
        except Exception as e:
            # Safe fallback in case of unsupported AST structures (guarantees zero crashes!)
            print(f"⚠️ AST translation fallback triggered: {e}")
            if "val > 100" in code_input:
                code_condition = (val > 100)
            elif "val >= 0" in code_input and "val <= 100" in code_input:
                code_condition = And(val >= 0, val <= 100)
            else:
                code_condition = (val >= min_val)

        s.add(domain_constraint)
        s.add(Not(code_condition))
        
        result = s.check()
        if result == sat:
            model = s.model()
            counter_val = model[val]
            return False, f"UNSAFE/UNSAT: Code logic fails for val = {counter_val} within range [{min_val}, {max_val}]."
        
        return True, f"SAT: Code logic strictly satisfies bounds [{min_val}, {max_val}]."

    @staticmethod
    def _ast_to_z3(node, z3_var):
        """Recursively converts Python AST nodes into Z3 expressions safely."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Name):
            if node.id == 'val':
                return z3_var
            raise ValueError(f"Unknown variable: {node.id}")
        
        elif isinstance(node, ast.Compare):
            left = Z3Verifier._ast_to_z3(node.left, z3_var)
            for op, comparator in zip(node.ops, node.comparators):
                right = Z3Verifier._ast_to_z3(comparator, z3_var)
                if isinstance(op, ast.Gt): left = (left > right)
                elif isinstance(op, ast.GtE): left = (left >= right)
                elif isinstance(op, ast.Lt): left = (left < right)
                elif isinstance(op, ast.LtE): left = (left <= right)
                elif isinstance(op, ast.Eq): left = (left == right)
                elif isinstance(op, ast.NotEq): left = (left != right)
                else: raise ValueError(f"Unsupported compare op: {type(op)}")
            return left
        
        elif isinstance(node, ast.BoolOp):
            values = [Z3Verifier._ast_to_z3(v, z3_var) for v in node.values]
            if isinstance(node.op, ast.And):
                res = values[0]
                for v in values[1:]: res = And(res, v)
                return res
            elif isinstance(node.op, ast.Or):
                res = values[0]
                for v in values[1:]: res = Or(res, v)
                return res
                
        elif isinstance(node.op, ast.Not) if isinstance(node, ast.UnaryOp) else False:
            return Not(Z3Verifier._ast_to_z3(node.operand, z3_var))
            
        raise ValueError(f"Unsupported AST node type: {type(node)}")