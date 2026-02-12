# REVIS: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models

This repository contains the official implementation of the paper **"REVIS: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models"**.

## Overview 

**REVIS** is a training-free framework designed to mitigate object hallucination in Large Vision-Language Models (LVLMs). By explicitly decoupling visual information from language priors via orthogonal projection and employing sparse intervention, REVIS effectively restores visual grounding during inference.

---
## Project Structure

```text
Revis/
├── data/                         # Datasets and assets
│   ├── OH_sampled_data_100.json  # Sampled data for vector extraction
|   ├── coco_pope_calibration_merged.jsonl  # Sampled data for layer selection 
├── calibration_results/          # Output directory for layer selection results
├── steering_utils/               # Core implementation modules
│   ├── data_loader.py            # Data loading and preprocessing
│   ├── llm_layers.py             # Tools for accessing and manipulating LLM layers
│   ├── pca.py                    # PCA analysis for visualization (e.g., t-SNE)
│   └── vector_calculator.py      # Implements Orthogonal Visual Vector Construction
├── utils/                        # Evaluation metrics and tools
│   ├── chair.py                  # CHAIR metric evaluation script
│   ├── mme.py                    # MME benchmark evaluation script
│   ├── mmvet_ori.py              # MM-Vet benchmark evaluation script
└── vector/                       # Pre-computed steering vectors
    ├── llava15_none_image.pt     # Steering vector for LLaVA-1.5
    ├── llava16_none_image.pt     # Steering vector for LLaVA-1.6
    └── qwen2.5vl_none_image.pt   # Steering vector for Qwen2.5-VL

```

---

## Preparation  
Set up the environment with:

```bash
pip install transformers >=4.53.0
```

---

## Usage
> **Note**: All results are based on **Hugging Face model versions** (via `transformers`) for consistency. We do not use model-specific codebases like the LLaVA GitHub repository.


The REVIS workflow consists of two primary stages:

1. **Vector Extraction**  
2. **Inference with Latent Steering**

### 1. Vector Extraction  
To compute your own steering vectors using custom data:

```bash
python main_all_visual_only.py compute
```
- Computes Vectors
- Applies Gram-Schmidt Orthogonalization
- Saves the purified vector to `vector/[model_name]_none_image.pt`

### 2. Inference with REVIS  
Apply sparse latent steering during generation:
```bash
# load Qwen2.5VL vector for steering
python main_all_visual_only.py run 
```

---

## Evaluation  

We evaluate REVIS using multiple standard benchmarks:


| Benchmark  | Task                                 | Metric                     |
|------------|--------------------------------------|----------------------------|
| **POPE**   | Object hallucination detection       | Accuracy, Precision, Recall |
| **CHAIR**  | Hallucination attribute analysis     | CHAIR Score                |
| **MME**    | Multi-modal perception & cognition   | Score                      |
| **MM-Vet** | Open-ended reasoning                 | Qwen3 judged score         |


---

## Key Components  

### 1. Vector Extraction (`steering_utils/`)

The core logic for constructing the steering vector lies in `steering_utils/vector_calculator.py`.
* It computes the **Raw Visual Vector** and **Language Prior Vector**.
* It performs the **Gram-Schmidt Orthogonalization** to synthesize the purified visual vector.

### 2. Pre-computed Vectors (`vector/`)
We provide pre-computed orthogonal steering vectors for popular LVLMs. You can directly load these `.pt` files to reproduce our results without re-running the extraction process:

* **LLaVA-1.5**
* **LLaVA-1.6**
* **Qwen2.5-VL**

