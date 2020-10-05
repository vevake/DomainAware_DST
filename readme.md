
# DA-DST : Domain-Aware Dialogue State Tracker for Multi-Domain Dialogue Systems

[<img src="https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png" width="10%">](https://pytorch.org/)[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[<img align="right" src="https://raw.githubusercontent.com/vevake/GSAT/master/imgs/unitn.png" width="20%">](https://www.unitn.it/)
[<img align="right" src="https://raw.githubusercontent.com/vevake/GSAT/master/imgs/fbk.png" width="15%">](https://www.fbk.eu/en/)

This is a PyTorch implementation of the paper: **[Domain-Aware Dialogue State Tracker for Multi-Domain Dialogue Systems](https://arxiv.org/abs/2001.07526).** by [Vevake Balaraman](https://scholar.google.it/citations?hl=it&user=GTtAXeIAAAAJ) and [Bernardo Magnini](https://scholar.google.it/citations?user=jnQE-4gAAAAJ&hl=it&oi=ao).

# Abstract

In task-oriented dialogue systems the dialogue state tracker (DST) component is responsible for predicting the state of the dialogue based on the dialogue history. Current DST approaches rely on a predefined domain ontology, a fact that limits their effective usage for large scale conversational agents, where the DST constantly needs to be interfaced with ever-increasing services and APIs. Focused towards overcoming this drawback, we propose a domain-aware dialogue state tracker, that is completely data-driven and it is modeled to predict for dynamic service schemas including zero-shot domains. Unlike approaches that propose separate models for prediction of intents, requested slots, slot status, categorical slots and non-categorical slots, we propose a single model in an end-to-end architecture. The proposed model also utilizes domain and slot information to extract both domain and slot specific representations from a given dialogue, and then uses such representations to predict the values of the corresponding slot in a given domain. Integrating this mechanism with a pretrained language models, our approach can effectively learn semantic relations and effectively perform zero-shot tracking for domains not present in training.

# Citation

The bibtex is below.

```text
@article{balaraman2020domainaware,
      title={Domain-Aware Dialogue State Tracker for Multi-Domain Dialogue Systems},
      author={Vevake Balaraman and Bernardo Magnini},
      year={2020},
      eprint={2001.07526},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

# Dataset

The [Schema Guided Dataset  (SGD)](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) is used for the experiments.

# Executing the program

1. Configure the directories and BERT model locations in ```config.py```

2. Create a schema dictionary for the dataset.

    ```python
    python create_schema_dict.py
    ```

3. create schema encodings

    ```python
    python encode_schema.py
    ```

4. Train the model

    ```python
    python train.py
    ```

5. Test the final model

    ```python
    python test.py
    ```

# Contact

Please feel free to contact me at balaraman@fbk.eu for any queries.
