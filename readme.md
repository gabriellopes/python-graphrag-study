
# python Study Case 4 deterministically self-improving code-writer z3 verifier shacl-shaped minimal program (Semantic Self-Learner Python Brain)


- clone the project; add an .env with your GEMINI_API_KEY and run uvicorn app-api.main:app --reload after having properly pip installing the requirements (pip install -r requirements.txt)


## minor bugs 

sometimes the display code will be froze due to buffering at the page; just refresh it and try a new problem;

- some problems wont need a memory cached solution in order to re-run and pass, as the brain has already stored it as a possible solution;

- if the unknown mapping is not working, probably the memory buffer is full and/or the connection to the LLM is not working properly;

## Incoming Version

A new version of this project is under work, in order to incrementally enrich the Memory of the Agent in order to propose coding-solutions for the problems yet not covered by the Semantic Brain;

Ex:
    'AST translation fallback triggered: Unsupported AST node type: <class 'ast.BinOp'>'

    > Agents proposes a new code; 
    > Test it, Validates it;
    > Leverage the new solution to its inner memory (either in Buffer, Sqlite, or Cloudy)
    > Even without inserting new code to the actual project, the Agent will Remember the solution, and use it, for those cases yet unsolved.

## exec pipeline

Here is the detailed breakdown of the production pipeline execution flow:

- User Intent & Code Ingestion: 
    The user or client submits a Python code implementation targeting a specific problem URI in the Knowledge Graph (e.g., ex:Problem5_6).

- Abstract Syntax Tree (AST) Parsing: 
    The system parses the submitted code into Python AST nodes to inspect function definitions, parameters, return statements, and internal variable logic.

- Ontology Alignment & Entity Resolution:
    The AST visitor extracts code entities and checks them against the local Turtle ontology (modules.ttl, TXT.ttl) to link functions, classes, and properties to known RDF URIs.

- Unknown Mapping Handling: 
    If unmapped classes or properties are encountered, the system queries the Google Gemini vector store (gemini-embedding-001) via semantic similarity to retrieve the closest existing ontology concepts, or uses Open World Assumption (OWA) reasoning to map them to valid parent classes.

- SHACL Structural & Contract Validation: 
    The mapped AST graph is converted into RDF triples and validated against code_shapes.ttl using pyshacl to enforce required parameter types, interface contracts, and module declarations.

- Z3 SMT Logical Verification: 
    For structurally valid code, the verifier extracts preconditions, postconditions, and numeric bounds from the graph problem definition and feeds them to the Z3 SMT solver to mathematically prove satisfiability (sat vs unsat).

- Closed-Loop Self-Correction Interceptor: 
    If SHACL returns structural violations or Z3 yields an unsat core (e.g., boundary condition failure), the exact error trace is sent back to gemini-3.5-flash-lite to automatically refactor the code until it satisfies all formal constraints.

- Knowledge Graph Enrichment & Vector Store Learning: 
    Once fully verified, the system generates new RDF triples representing the solution, links them to ex:Problem and ex:Solution nodes in GraphDB, and embeds the verified code into the Gemini vector space—effectively teaching the "Semantic Brain" a new verified capability for future RAG retrievals.

- Final Verified Pipeline Response: 
    The system returns the complete execution trace, pass/fail status for each phase, and the verified, provably correct Python artifact back to the user or frontend visualizer.
