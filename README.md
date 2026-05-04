# Lipophilicity
## Machhine Learning for Drug Design - Final Project
### Group: Alexandra Biddiscombe, Louie Daans, Viktória Kubišová

GitHub repository for the final project of Machine Learning for Drug Design, on the study of the lipophilicity (absorption) of molecules.

#### Group Deliverables
- Written report (one per group)
- Structured like a scientific paper: Abstract, Introduction, Materials & Methods, Results, Discussion/Conclusion, References
  **Hint:** Use a LaTeX or Word template from a scientific journal to guide formatting and structure
- Code repository (one per group)
- Include scripts, notebooks, training/test datasets and any supplementary files
- Should be reproducible and well-commented
  **Hint:** You can organize it like a GitHub repository
- Oral presentation/discussion (one per group, everyone has to present!)
- Cover the overall project (e.g., include insight into modeling decisions)
- Each student will have to go deeper into the parts they covered

Data:  https://tdcommons.ai/single_pred_tasks/adme/#lipophilicity-astrazeneca

#### Steps:
##### Literature search
- Selecting a few benchmarks available in literature
##### Data sources, selection and cleaning
- Training and test sets splitting strategies
- Data selection and cleaning as usual
##### Features selection
- Features generation and selection will be evaluated
##### Training and testing the ML approach
- The appropriate machine learning model should be selected
- The results should be discussed with available models in literature.
##### Deployment
- The model should work in a real-world environment providing
- Explainable Prediction
- Indication about the applicability domain

---

### Files in the repository and what they do:
- **README.md**: Markdown file which summarises project objectives and repository content;
- **Lipophilicity.ipynb**: Python Notebook containing the code for non-GNN regression models, with explanation for each step, following the processes learned in class;
- **GNN_models_randomsplit**: Python Notebook containing the code for GNN models, which compares a custom coded simple Graph model with convolutional layers (baseline), PyTorch's PyG GCN (Graph Convolutional Network) model, and PyTorch's PyG GIN (Graph Isomorphisms Network) model, using a random split of the initial data;
- **GNN_models_scaffoldsplit**: Python Notebook containing the code for the same GNN models, using a scaffold split of the initial data;
- **compare_models_testsets.ipynb**: A Python Notebook evaluating the 6 total resulting models (3 on one training split method, 3 on the other);
- **helpers.ipynb**: A collection of helper functions including code from the lectures, used in the GNN notebooks.

Directories:
- **Research papers**: Contains a selection of PDFs used during the literature search section;
- **models**: Contains results from non-GNN model creation;
- **saved_models_randomsplit**: Contains model data from GNN model creation using random data splitting;
- **saved_models_scaffoldsplit**: Contains model data from GNN model creation using scaffold data splitting; 

---

### Work distribution:
- **Alexandra Biddiscombe**: Data Exploration, Literature Study and Model Research, Report
- **Louie Daans**: Data Exploration, GNN Theory Research, GNN Modelling Decisions, Report
- **Viktória Kubišová**: Project Management, Data Exploration, Feature Selection and Analysis, Modelling Decisions, Report
