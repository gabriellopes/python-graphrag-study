import pyshacl
from rdflib import Graph

class SHACLValidator:
    """Validates RDF AST graphs against code_shapes.ttl."""

    def __init__(self, shape_path: str):
        self.shape_graph = Graph().parse(shape_path, format="ttl")

    def validate(self, data_graph: Graph) -> tuple[bool, str]:
        conforms, _, results_text = pyshacl.validate(
            data_graph,
            shacl_graph=self.shape_graph,
            inference="rdfs"
        )
        return conforms, results_text