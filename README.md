# BEFT (Accepted to ACL 2026 Main Conference)
BEFT: Bias-Efficient Fine-Tuning of Language Models in Low-Data Regimes

![BEFT](./img/main.png)


# Environment 

```bash
pip install -r requirements.txt
```

# Our Key Finding:
> We provide a user-friendly and easy-to-run notebook, `tutorial.ipynb`, that offers a simple tutorial and visualizes importance rankings and the achieved accuracy of **b**<sub>v</sub>, **b**<sub>q</sub>, and **b**<sub>k</sub> by different bias-term investigation approaches in SST-2 low-data regime.

To get the importance ranking by **our bias-efficient approch** and **_Magnitude_ approach** for BERT<sub>BASE</sub> on low-data SST-2 dataset:

```bash
python run_BEFT.py 
       --task-name sst2\  
       --model-name bert-base-cased
```

To get the importance ranking by **_Fisher_ approach** for BERT<sub>BASE</sub> on low-data SST-2 dataset:

```bash
python run_BEFT.py 
       --task-name sst2\  
       --model-name bert-base-cased\
       --fisher True\
       --batch-size 8
```

To get the performance of fine-tuning among **b**<sub>v</sub>, **b**<sub>q</sub>, and **b**<sub>k</sub> for BERT<sub>BASE</sub> on low-data SST-2 dataset:

```bash
python run_BEFT.py 
       --task-name sst2\  
       --model-name bert-base-cased\
       --bias-terms-loop True
```
For Table 1 (**evaluating our bias-efficient approach across tasks and data regimes**) in our paper:
```bash
#<task>: ['sst2','rte','qqp','qnli','mnli','cola','mrpc','stsb']
#<model>:['bert-base-cased','bert-large-cased','roberta-base']
#<data>: ['low','medium','high'] 
python run_BEFT.py 
       --task-name <task>\  
       --model-name <model>\ 
       --data-regime <data>\ 
       --bias-terms-loop True 
```
For Table 2 (**efficiency and effectiveness**) in our paper:
```bash
#for our BEFT
python run_BEFT.py 
       --task-name rte\
       --model-name bert-base-cased\
       --bias-terms-loop True 

#for All biases
python run_BEFT.py 
       --task-name rte\
       --model-name bert-base-cased\
       --bias-terms-loop False

#for Rand uniform and All parameters
#<fine_tune>: ['rand_uniform','full_ft']
python run_BEFT.py 
       --fine-tune-type <fine_tune>\
       --task-name rte\
       --model-name bert-base-cased\
```
# Extending Our Key Finding without Requiring Any Post-Hoc Evaluation:
For Table 3 (**extending our key finding to different datasets and LLMs**) in our paper:
```bash
#<task>: ['sst2','rte','cb','wic']
#<model>:['bert-base-cased','bert-large-cased','roberta-base']
python run_BEFT.py 
       --task-name <task>\  
       --model-name <model>\ 
       --bias-terms-loop True 
```
Extending LoRA to BEFT (**injecting LoRA into **b**<sub>v</sub>, **b**<sub>q</sub>, and **b**<sub>k</sub>**) in Table 4:
```bash
python others/run_BEFT_LoRA.py 
       --task-name sst2\
       --model-name bert-base-cased\
       --bias-terms-loop True 
```
Extending DoRA, VeRA, and VeRA1D to BEFT are also included in `others/BEFT_LoRA_evaluator.py`.

# Generalize to Autoregressive LLMs without Requiring Any Post-Hoc Evaluation:
To generalize our findings to get the performance of **b**<sub>v</sub>, **b**<sub>q</sub>, and **b**<sub>k</sub> for OPT-1.3B on low-data RTE dataset:
```
bash autoregressive_llm/finetune_BEFT.sh
```

To get the performance of prefix-tuning and LoRA fine-tuning for OPT-1.3B on low-data RTE dataset:
```bash
# prefix-tuning
MODE=prefix bash autoregressive_llm/finetune.sh

# LoRA
MODE=lora bash autoregressive_llm/finetune.sh
```

To get the performance the ICL and Zero-Shot techniques for OPT-1.3B on low-data RTE dataset:
```bash
# In-context learning
bash autoregressive_llm/icl.sh 

# Zero-shot
bash autoregressive_llm/icl.sh --num_train 0
```
For Table 5 in our paper:
```bash
#<task>: ['SST2','RTE','CB','BoolQ','WSC','WIC','MultiRC','Copa','ReCoRD','SQuAD','DROP']
#<model>:['facebook/opt-1.3b','facebook/opt-6.7b']
#<mode>: ['prefix','lora']
#our BEFT
MODEL=<model> TASK=<task> bash autoregressive_llm/finetune_BEFT.sh
# prefix-tuning and LoRA
MODE=<mode> MODEL=<model> TASK=<task> bash autoregressive_llm/finetune.sh
# In-context learning
MODEL=<model> TASK=<task> bash autoregressive_llm/icl.sh 
# Zero-shot
MODEL=<model> TASK=<task> bash autoregressive_llm/icl.sh --num_train 0
```

# Adding Bias Terms to Bias-Free LLMs without Requiring Any Post-Hoc Evaluation:
To generalize our findings to bias-free LLMs with adding **b**<sub>v</sub> for LLaMA2-7B on low-data SST-2 dataset:
```bash
MODE=additive_bias MODEL=meta-llama/Llama-2-7b-hf TASK=SST2 bash autoregressive_llm/finetune_BEFT.sh
```
For Table 6 and Appendix C.7-Table 18 in our paper:
```bash
#<task>: ['SST2','Copa','SQuAD']
#<model>:['meta-llama/Llama-2-7b-hf','EleutherAI/gpt-j-6b','deepseek-ai/deepseek-coder-1.3b-base']
MODE=additive_bias MODEL=<model> TASK=<task> bash autoregressive_llm/finetune_BEFT.sh
```

# Citation
```
@article{huang2025beft,
  title={BEFT: Bias-Efficient Fine-Tuning of Language Models},
  author={Huang, Baichuan and Balashankar, Ananth and Aminifar, Amir},
  journal={arXiv preprint arXiv:2509.15974},
  year={2025}
}
```

> **Acknowledgement**: We acknowledge the following repositories [BitFit](https://github.com/benzakenelad/BitFit) and [MeZO](https://github.com/princeton-nlp/mezo).
