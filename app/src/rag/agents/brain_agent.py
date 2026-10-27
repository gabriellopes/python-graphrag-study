import logging
import time
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

    def process_code(self, candidate_code: str, problem_spec: dict = None, max_corrections: int = 3) -> dict:
        current_code = candidate_code
        attempt = 0
        pipeline_trace = []
        start_time = time.time()

        spec = problem_spec or {}
        min_v = spec.get("min_val", 0)
        max_v = spec.get("max_val", 100)

        while attempt < max_corrections:
            attempt += 1
            logger.info(f"--- Pipeline Execution Attempt {attempt} ---")

            # 1. AST Generation & Mapping
            t0 = time.time()
            ast_graph, unknowns, ast_mermaid = self.resolver.parse_and_map(current_code)
            t1 = time.time()
            pipeline_trace.append({
                "step_number": len(pipeline_trace) + 1,
                "step_name": f"AST Parsing & Mapping (Attempt {attempt})",
                "latency_ms": round((t1 - t0) * 1000, 2),
                "status": "PASSED",
                "details": f"AST parsed successfully into graph model on attempt {attempt}.",
                "visual_type": "mermaid",
                "visual_content": ast_mermaid
            })

            # 2. SHACL Validation
            t0 = time.time()
            conforms, shacl_report = self.shacl.validate(ast_graph)
            t1 = time.time()
            pipeline_trace.append({
                "step_number": len(pipeline_trace) + 1,
                "step_name": f"SHACL Contract Validation (Attempt {attempt})",
                "latency_ms": round((t1 - t0) * 1000, 2),
                "status": "PASSED" if conforms else "FAILED",
                "details": "Conforms strictly to SHACL shape definitions." if conforms else "SHACL validation failed.",
                "visual_type": "code",
                "visual_content": shacl_report if not conforms else "@prefix sh: <http://www.w3.org/ns/shacl#> .\nex:ExampleCodeShape -> CONFORMS"
            })

            if not conforms:
                logger.warning("SHACL Failed. Querying Gemini for structural refactoring...")
                prompt = f"Fix this Python code to satisfy structural requirements:\n{current_code}\nError Report:\n{shacl_report}"
                
                t0 = time.time()
                raw_response = self.generator.generate(prompt)
                t1 = time.time()
                
                current_code = self._extract_code_block(raw_response)
                pipeline_trace.append({
                    "step_number": len(pipeline_trace) + 1,
                    "step_name": f"LLM Self-Correction (SHACL Refactor - Attempt {attempt})",
                    "latency_ms": round((t1 - t0) * 1000, 2),
                    "status": "PASSED",
                    "details": "Gemini refactored code to resolve SHACL structural violations.",
                    "visual_type": "code",
                    "visual_content": f"# Refactored Code Artifact:\n{current_code}"
                })
                continue

            # 3. Z3 SMT Verification
            t0 = time.time()
            is_sat, z3_msg = self.z3.verify_bounds(
                min_val=min_v, 
                max_val=max_v, 
                code_input=current_code
            )
            t1 = time.time()
            
            pipeline_trace.append({
                "step_number": len(pipeline_trace) + 1,
                "step_name": f"Z3 Deterministic Logic Check (Attempt {attempt})",
                "latency_ms": round((t1 - t0) * 1000, 2),
                "status": "PASSED" if is_sat else "FAILED",
                "details": z3_msg,
                "visual_type": "code",
                "visual_content": f"(declare-const val Int)\n(assert (>= val {min_v}))\n(assert (<= val {max_v}))\n(check-sat) -> {'SAT' if is_sat else 'UNSAT'}\n; Message: {z3_msg}"
            })

            if not is_sat:
                logger.warning(f"Z3 UNSAT/UNSAFE: {z3_msg}. Querying Gemini for logic refactoring...")
                prompt = f"""
                You are a formal verification AI. 
                This Python code failed Z3 verification for range [{min_v}, {max_v}]:
                
                ```python
                {current_code}
                ```
                
                Z3 Error / Counterexample:
                {z3_msg}
                
                Task: Rewrite the code so it strictly satisfies the range contract and passes formal verification. Return ONLY valid Python code inside markdown blocks.
                """
                
                t0 = time.time()
                raw_response = self.generator.generate(prompt)
                t1 = time.time()
                
                current_code = self._extract_code_block(raw_response)
                pipeline_trace.append({
                    "step_number": len(pipeline_trace) + 1,
                    "step_name": f"LLM Self-Correction (Z3 Refactor - Attempt {attempt})",
                    "latency_ms": round((t1 - t0) * 1000, 2),
                    "status": "PASSED",
                    "details": f"Gemini queried to refactor code based on Z3 counterexample: {z3_msg}",
                    "visual_type": "code",
                    "visual_content": f"# LLM Refactored Code Artifact (Attempt {attempt}):\n{current_code}"
                })
                continue

            # 4. Verified & Enriched Success
            return {
                "status": "VERIFIED_AND_ENRICHED" if attempt > 1 else "VERIFIED",
                "attempts_required": attempt,
                "verified_code": current_code,
                "total_latency_ms": round((time.time() - start_time) * 1000, 2),
                "pipeline_trace": pipeline_trace
            }

        return {
            "status": "FAILED_MAX_ATTEMPTS",
            "attempts_required": max_corrections,
            "verified_code": current_code,
            "total_latency_ms": round((time.time() - start_time) * 1000, 2),
            "pipeline_trace": pipeline_trace
        }

    def _extract_code_block(self, text: str) -> str:
        if "```python" in text:
            parts = text.split("```python")
            return parts[1].split("```")[0].strip()
        elif "```" in text:
            parts = text.split("```")
            return parts[1].split("```")[0].strip()
        return text.strip()