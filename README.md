# Clinical Fact Verification Using Retrieval-Augmented Verification (RAV)

## Overview
This repository contains the implementation for the project "Clinical Fact Verification Using Retrieval-Augmented Verification (RAV)". The project adapts the Retrieval-Augmented Verification (RAV) framework from Zheng et al. "Evidence Retrieval is Almost All You Need for Fact Verification" (ACL Anthology, 2024) to verify medical claims against authoritative clinical guidelines, such as UpToDate, StatPearls, and OpenEvidence.

The system consists of:
1. **Evidence Retrieval**: Extracting relevant evidence from large corpora.
2. **Claim Verification**: Classifying claims as `Supports`, `Contradicts`, or `Ambiguous` based on retrieved evidence.
3. **Dataset Curation**: Generating a custom dataset of claims categorized into supporting, contradicting, and ambiguous using pre-trained language models and curated medical articles.
4. **Evaluation Metrics**: Measuring the system’s performance using metrics like precision, recall, F1-score, and retrieval accuracy.

---

## Key Features
- **Hybrid Evidence Retrieval**: Utilizes bi-encoder retrieval for initial evidence filtering and cross-encoder ranking for refined evidence selection.
- **Custom Dataset Generation**: Generates nuanced claims from medical guidelines.
- **Transformer-Based Models**: Incorporates domain-specific models like PubMedBERT for evidence retrieval and claim verification.
- **Claim Categorization**: Classifies claims into `Supports`, `Contradicts`, and `Ambiguous` categories.

## Models Used
1. **Bi-Encoder Retriever**
   - Model: PubMedBERT (fine-tuned)
   - Purpose: Initial evidence filtering
2. **Cross-Encoder Ranker**
   - Model: PubMedBERT (fine-tuned)
   - Purpose: Refined evidence ranking
3. **Claim Classifier**
   - Model: Transformer-based classifier (PubMedBERT backbone)
   - Purpose: Classifying claims as `Supports`, `Contradicts`, or `Ambiguous`

---

## Evaluation Metrics
   - Precision
   - Recall
   - F1-Score
   - Accuracy

---

## Results
### Classification Metrics
| Class            | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| Supports         | 95%       | 94%    | 95%      |
| Contradicts      | 94%       | 96%    | 95%      |
| Ambiguous        | 97%       | 99%    | 98%      |

---

## Known Issues
- Evidence ranking requires improved handling of ambiguity.
- Chunking may result in context loss for long paragraphs.
- Limited scalability with full-length articles.

---

## Future Work
- **Enhanced Evidence Retrieval:** Implement hierarchical chunking and advanced context aggregation.
- **Real-Time Integration:** Develop a browser extension for clinicians.
- **Expanded Datasets:** Incorporate additional clinical sources to improve robustness.

---

## Contributors
- **Shubham Rateria** (NetID: sr2369, Cornell Tech)
- **William J. Reid** (NetID: wjr83, Cornell Tech)
- **Harshini Donepudi** (NetID: hsd39, Cornell Tech)

---

## Setup for langchain -> duckduckgo pipeline
## Setup 

```
conda env create -f environment.yml # install the requirements in a conda environment

export OPENAI_API_KEY="<key-here>"
```

## Generating Data

```
cd data

python data-gen-script-2.py
```

The currently generated dataset can be found at [data/csv/generated_claim_triplets_with_topics.csv](data/csv/generated_claim_triplets_with_topics.csv)

## Training and loading the RAV model

The file [rav-final.ipynb](rav-final.ipynb) contains the code for training the model and loading the trained model for inference. The file contains different sections which explain how to use them.