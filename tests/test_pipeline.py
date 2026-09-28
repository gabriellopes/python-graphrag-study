import pytest
from app.src.rag.agents.brain_agent import SemanticBrainAgent

def test_full_semantic_pipeline():
    agent = SemanticBrainAgent(shape_path="app/data/shapes/code_shapes.ttl")
    
    # Intentionally slightly flawed snippet
    raw_code = "def check_bounds(val):\n    return val >= 0"
    
    result = agent.process_code(raw_code)
    
    assert result["status"] == "VERIFIED_AND_ENRICHED"
    assert result["shacl_status"] == "CONFORMS"
    assert result["z3_status"] == "SAT"
    print(f"\n[Pipeline Result]:\n{result}")

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])