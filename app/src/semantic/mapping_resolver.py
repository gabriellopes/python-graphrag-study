import ast
from rdflib import Graph, URIRef, Literal, RDF, Namespace
from app.src.rag.embeddings import EmbeddingModelLoader

EX = Namespace("http://example.org/kg-study#")

class MappingResolver:
    """Parses Python AST, resolves known/unknown ontology mappings, and converts AST to RDF triples."""

    def __init__(self, embedder: EmbeddingModelLoader):
        self.embedder = embedder
        # Known registered entities in our ontology graph
        self.known_entities = {
            "check_bounds": EX.check_bounds,
            "val": EX.val_parameter,
            "int": EX.IntegerType
        }

    def parse_and_map(self, code_str: str) -> tuple[Graph, list[str]]:
        tree = ast.parse(code_str)
        ast_graph = Graph()
        ast_graph.bind("ex", EX)
        
        example_uri = EX.ExGenerated_01
        ast_graph.add((example_uri, RDF.type, EX.Example))
        ast_graph.add((example_uri, EX.codeSnippet, Literal(code_str)))

        unknown_mappings = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                ast_graph.add((example_uri, EX.hasFunctionName, Literal(node.name)))
                if node.name in self.known_entities:
                    ast_graph.add((example_uri, EX.mapsToFunction, self.known_entities[node.name]))
                else:
                    # Trigger vector similarity fallback for Unknown Mapping
                    unknown_mappings.append(node.name)
                    vec = self.embedder.get_embedding(node.name)
                    # Infer closest concept
                    ast_graph.add((example_uri, EX.mapsToUnknownConcept, Literal(f"vector_dim_{len(vec)}")))

        return ast_graph, unknown_mappings