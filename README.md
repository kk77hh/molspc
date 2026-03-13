# MolSPC: Chemical-Context-Aware Foundation Model for Molecular Generation and Optimization via Hierarchical Structure-Property Alignment
Peng Han, Ke Yan, Xiaochen Xie, Bin Liu*

## Abstract

Molecular generative design and multi-property optimization are pivotal in drug discovery, and recent advances in pretrained molecular foundation models have brought significant breakthroughs. However, existing pretrained models lack explicit chemical context, limiting their ability to capture nuanced structure-property relationships and guide targeted molecular modifications. Current approaches typically process molecules in isolation, neglecting the chemical deltas (subtle structural changes influencing property shifts) and failing to integrate key structural insights into generative workflows. To address these gaps, we introduce MolSPC (Molecular Structure-Property modeling with Contextual learning), a chemical-context-aware foundation model that unifies molecular understanding and generation through a hierarchical pretraining framework. First, our chemical self-awareness pretraining develops substructure-sensitive representations through self-supervised learning on large molecular corpora, enabling the model to intrinsically understand atomic environments. Second, cross-modal semantic grounding aligns molecular graphs with textual descriptors via contrastive learning and bidirectional molecule-text matching, creating a unified chemical semantics space. Third, our delta-aware co-learning framework processes structurally analogous molecule pairs to jointly predict property differences and structural modifications, establishing explicit structure-property transformation rules. For downstream tasks, MolSPC is fine-tuned via supervised learning and reinforcement learning, demonstrating superior performance in scaffold-based generation, multi-property optimization and protein-pocket-conditional design. Notably, MolSPC provides interpretable structure-activity relationships (SAR) insights by revealing how functional-group-level edits drive property modulation. Our work advances AI-driven molecular design by bridging contextual chemical awareness with generative precision, offering a paradigm for flexible, multi-objective drug discovery.

## Requirements

See `environment.yml`. Run the following command to create a new anaconda environment `molspc`: 


```bash
conda env create -f environment.yml
```


## Reproduce the results


**File preparation.** Firstly, please download the necessary datasets and checkpoints files from the link (https://huggingface.co/kk77hh/MolSPC/tree/main).


**Pretraining stage1 and stage2.** We achieved chemical self-awareness pretraining and cross-modal semantic grounding through the first two stages of pretraining, with the implementation approach referenced from the [molca](https://github.com/acharkq/MolCA) article. 


**Pretraining stage3.**

```bash
python data_pretrain_stage3.py
```

```bash
python stage2.py --root 'data/compare/train/'  --devices '0,1' --valid_root 'data/compare/valid/'  --filename "stage2" --stage2_path "all_checkpoints/pretrain_stage2.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 20 --mode pretrain --prompt '[START_I_SMILES]{}[END_I_SMILES]' --tune_gnn --llm_tune lora --inference_batch_size 2  --double True --batch_size 8
```

### Training the Model for Molecular Generation and Optimization

**Data processing.** 

```bash
python data_finetune_molopt.py 
```

**Fine-tune stage.** 

```bash
python stage2.py --root 'data/opt/train/'  --devices '0,1' --valid_root 'data/opt/valid/'  --filename "stage2" --stage2_path "all_checkpoints/pretrain_stage3.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 8 --mode pretrain --prompt '[START_I_SMILES]{}[END_I_SMILES]' --tune_gnn --llm_tune lora --inference_batch_size 2  --double True --batch_size 16 --init_checkpoint all_checkpoints/pretrain_stage3.ckpt
```


**Test**

```bash
python stage2.py --root 'data/opt/train/' --devices '0,' --valid_root 'data/opt/valid/' --filename 'stage2' --stage2_path 'all_checkpoints/stage2/checkpoint_molopt.ckpt' --opt_model 'facebook/galactica-1.3b' --max_epochs 20 --mode eval --prompt '[START_I_SMILES]{}[END_I_SMILES]' --tune_gnn --llm_tune lora --inference_batch_size 2 --double True --batch_size 1 --init_checkpoint all_checkpoints/stage2/checkpoint_molopt.ckpt
```