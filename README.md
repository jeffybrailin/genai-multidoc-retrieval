### Name: Jeffy Brailin T
### Reg.No: 212223040076

## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
The challenge is to design and implement a multi-document retrieval agent capable of efficiently extracting and synthesizing information from a collection of quantum computing research papers. The agent should leverage LlamaIndex’s retrieval-augmented framework to build intelligent tools that can perform semantic search and summarization across multiple PDFs

### DESIGN STEPS:

#### STEP 1:
Data Collection and Preprocessing
#### STEP 2:
Index Construction with LlamaIndex & Query Handling and Response Generation
#### STEP 3:
Evaluation and Testing

### PROGRAM:

```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

import os, nest_asyncio
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
nest_asyncio.apply()

papers = [
    "22266_Accelerating_Inference_f.pdf",   
    "7839_Accelerating_Regression_T.pdf",   
    "25645_Quantum_Inspired_Image_E.pdf",   
]

from utils import get_doc_tools
from pathlib import Path

paper_to_tools = {}
for paper in papers:
    print(f"Getting tools for: {paper}")
    vtool, stool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools[paper] = [vtool, stool]

initial_tools = [t for p in papers for t in paper_to_tools[p]]
len(initial_tools)

from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

llm = OpenAI(model="gpt-3.5-turbo")  # keep same API key; upgrade model if you like

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools,
    llm=llm,
    verbose=True
)
agent = AgentRunner(agent_worker)

resp = agent.query(
    "From the three papers, list each paper's main goal in 2-3 bullets each."
)
print(str(resp))

resp = agent.query(
    "Compare evaluation setups: For Regression vs Image-encoding vs Inference-acceleration papers, "
    "what headline results or guarantees are claimed?"
)
print(str(resp))


from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

all_tools = [t for p in papers for t in paper_to_tools[p]]
obj_index = ObjectIndex.from_objects(all_tools, index_cls=VectorStoreIndex)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)

from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm,
    system_prompt=(
        "You are an agent designed to answer queries ONLY using the provided PDF tools. "
        "Do not rely on prior knowledge. Cite paper names in answers."
    ),
    verbose=True
)
agent = AgentRunner(agent_worker)


t(agent.query(
    "Summarize: (1) the quantum approach to accelerating multilayer NN inference; "
    "(2) the quantum framework for accelerating generalized regression tasks; "
    "(3) the quantum-inspired image encodings pipeline for financial time series. "
    "Then give a 6-row comparison table with: Paper | Core Idea | Data/Bench | Complexity/Claim | Empirical Result | Limitations."
))

print(agent.query(
    "For each paper, extract key assumptions (e.g., QRAM, access models, sparsity, phase encodings), "
    "and explain how these assumptions affect the stated speedups or accuracy."
))

print(agent.query(
    "Create a step-by-step reading guide for these three papers: must-read sections, figures/tables to focus on, "
    "and a 10-item glossary of terms appearing across the works."
))





```
### OUTPUT:

<img width="1299" height="452" alt="image" src="https://github.com/user-attachments/assets/b4d4aca4-6fb8-421c-9e7a-0be890676e51" />

<img width="851" height="488" alt="image" src="https://github.com/user-attachments/assets/3763615f-534d-4a5d-80ab-4ea358592738" />

### RESULT:
The system successfully retrieves and synthesizes relevant information from multiple documents, providing concise and relevant answers to the user's query. Performance is evaluated based on the accuracy, relevance, and coherence of the responses.
