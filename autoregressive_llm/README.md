

## Installation

This code is tested on the following libraries:
```
torch==2.1.2
transformers==4.28.1
accelerate==0.33.0
```

## Usage

Use `run.py` for all functions (zero-shot/ICL/fine-tuning):
```bash
python run.py {ARGUMENTS}
```

Please read `run.py` for a complete list of arguments. We introduce some of the most important ones below. 
* `--additive_bias`: adding bias term to bias-free LLMs
* `--num_train`: Number of training examples. For ICL, this is the number of demonstrations.
* `--num_dev`: Number of validation examples.
* `--num_test`: Number of testing examples.
* `--model_name`: HuggingFace model name or path.
* `--task_name`: Task name.
* `--trainer`: can be `none` (zero-shot/ICL), `regular` (fine-tuning).
* `--train_as_classification`: turn this on for classification tasks (Cross Entropy over likelihood of each class' label words). Otherwise it is LM-style teacher forcing.
* `--prefix_tuning`: use prefix-tuning. 
* `--lora`: use LoRA.




