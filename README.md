# BEFT 
BEFT: Bias-Efficient Fine-Tuning Language Models

![BEFT](./img/main.png)


# Environment 

```
$ pip install -r requirements.txt
```


# Evaluation examples:

To get the importance ranking by our proposed bias-efficient metric on low-date SST-2 dataset:

```
python run_BEFT.py 
       --task-name sst2\  
       --model-name bert-base-cased\
       --bias-terms-loop False
```

To get the performance of fine-tuning different bias on low-date  SST-2 dataset:

```
python run_BEFT.py 
       --task-name sst2\  
       --model-name bert-base-cased\
       --bias-terms-loop True
```

