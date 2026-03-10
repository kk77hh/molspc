# MolSPC: Chemical-Context-Aware Foundation Model for Molecular Generation and Optimization via Hierarchical Structure-Property Alignment
Peng Han, Ke Yan, Bin Liu


## Requirements

See `environment.yml`. Run the following command to create a new anaconda environment `molspc`: 


```bash
conda env create -f environment.yml
```


## Reproduce the results


**File preparation**Firstly, please download the necessary datasets and checkpoints files from the link (https://huggingface.co/kk77hh/MolSPC/tree/main).


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