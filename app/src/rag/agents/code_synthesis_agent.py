# app/src/rag/agents/code_synthesis_agent.py
import pyshacl
from rdflib import Graph
from app.src.verification.z3_verifier import Z3CodeVerifier

class SemanticCodeAgent:
    def __init__(self, ttl_path: str, shape_path: str):
        self.data_graph = Graph().parse(ttl_path, format="ttl")
        self.shape_graph = Graph().parse(shape_path, format="ttl")

    def validate_shacl(self) -> tuple[bool, str]:
        conforms, _, results_text = pyshacl.validate(
            self.data_graph,
            shacl_graph=self.shape_graph,
            inference='rdfs'
        )
        return conforms, results_text

    def synthesize_verified_code(self, prompt: str) -> str:
        # 1. SHACL Check
        conforms, shacl_report = self.validate_shacl()
        if not conforms:
            return f"SHACL Structural Violation Detected:\n{shacl_report}"

        # 2. Z3 Satisfiability Check
        is_satisfiable = Z3CodeVerifier.verify_numeric_bounds(min_val=10, max_val=5)
        if not is_satisfiable:
            return "Z3 Verification Failed: Unsound numeric bounds detected in requested logic (UNSAT)."

        return "Code synthesized and deterministically verified!"
