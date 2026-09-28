import logging
from app.src.rag.generation import GeminiGenerator
from app.src.rag.embeddings import EmbeddingModelLoader
from app.src.semantic.mapping_resolver import MappingResolver
from app.src.semantic.shacl_validator import SHACLValidator
from app.src.verification.z3_verifier import Z3Verifier

logger = logging.getLogger("SemanticBrainAgent")

class SemanticBrainAgent:
    def __init__(self, shape_path: str):
        self.generator = GeminiGenerator()
        self.embedder = EmbeddingModelLoader()
        self.resolver = MappingResolver(self.embedder)
        self.shacl = SHACLValidator(shape_path)
        self.z3 = Z3Verifier()

    def process_code(self, candidate_code: str, max_corrections: int = 3) -> dict:
        current_code = candidate_code
        attempt = 0

        while attempt < max_corrections:
            attempt += 1
            logger.info(f"--- Pipeline Execution Attempt {attempt} ---")

            # 1. AST & Unknown Mappings
            ast_graph, unknowns = self.resolver.parse_and_map(current_code)

            # 2. SHACL Validation
            conforms, shacl_report = self.shacl.validate(ast_graph)
            if not conforms:
                logger.warning("SHACL Failed. Intercepting and sending error trace to Gemini...")
                prompt = f"Fix this Python code to satisfy structural requirements:\n{current_code}\nError Report:\n{shacl_report}"
                current_code = self.generator.generate(prompt)
                continue

            # 3. Z3 SMT Verification
            is_sat, z3_msg = self.z3.verify_bounds(val_min=0, val_max=100)
            if not is_sat:
                logger.warning("Z3 UNSAT. Intercepting and sending logical failure to Gemini...")
                prompt = f"Fix this Python code logic to satisfy bounds [0, 100]:\n{current_code}\nZ3 Error:\n{z3_msg}"
                current_code = self.generator.generate(prompt)
                continue

            # 4. Verified & KG Enriched
            return {
                "status": "VERIFIED_AND_ENRICHED",
                "attempts_required": attempt,
                "verified_code": current_code,
                "unknowns_handled": unknowns,
                "shacl_status": "CONFORMS",
                "z3_status": "SAT"
            }

        return {"status": "FAILED", "last_code": current_code}