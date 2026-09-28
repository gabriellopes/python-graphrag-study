import ast
from rdflib import Graph, Literal, RDF, Namespace
from app.src.rag.embeddings import EmbeddingModelLoader

EX = Namespace("http://example.org/kg-study#")

class MappingResolver:
    """Parses Python AST, resolves ontology mappings, and formats graph visualizations."""

    def __init__(self, embedder: EmbeddingModelLoader):
        self.embedder = embedder
        self.known_entities = {
            "check_bounds": EX.check_bounds,
            "val": EX.val_parameter,
            "int": EX.IntegerType
        }

    def generate_ast_mermaid(self, code_str: str) -> str:
        """Converts Python code into a Mermaid tree visual graph."""
        tree = ast.parse(code_str)
        lines = ["graph TD"]
        
        def _walk_ast(node, parent_id="root"):
            node_id = f"node_{id(node)}"
            node_label = type(node).__name__
            
            if isinstance(node, ast.FunctionDef):
                node_label = f"FunctionDef: {node.name}"
            elif isinstance(node, ast.arg):
                node_label = f"arg: {node.arg}"
            elif isinstance(node, ast.Name):
                node_label = f"Name: {node.id}"
            elif isinstance(node, ast.Constant):
                node_label = f"Constant: {node.value}"
            elif isinstance(node, ast.Compare):
                node_label = "Compare (op)"

            lines.append(f'    {parent_id}["{parent_id.split("_")[0]}"] --> {node_id}["{node_label}"]')
            
            for child in ast.iter_child_nodes(node):
                _walk_ast(child, node_id)

        _walk_ast(tree, "Module")
        return "\n".join(lines)

    def parse_and_map(self, code_str: str) -> tuple[Graph, list[str], str]:
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
                    unknown_mappings.append(node.name)

        mermaid_graph = self.generate_ast_mermaid(code_str)
        return ast_graph, unknown_mappings, mermaid_graph