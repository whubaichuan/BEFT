# BEFT 
BEFT: Bias-Efficient Fine-Tuning Language Models

![BEFT](./img/main.png)


# Environment 

```
$ pip install -r requirements.txt
```


# Evaluation examples:

To get the importance ranking by **our proposed bias-efficient metric** and **_Magnitude_ metric** for BERT<sub>BASE</sub> on low-date SST-2 dataset:

```
python run_BEFT.py 
       --task-name sst2\  
       --model-name bert-base-cased
```

To get the importance ranking by **_Fisher_ metric** for BERT<sub>BASE</sub> on low-date SST-2 dataset:

```
python run_BEFT.py 
       --task-name sst2\  
       --model-name bert-base-cased\
       --fisher-metric True\
       --batch-size 8
```

To get the performance of fine-tuning among **b**<sub>v</sub>, **b**<sub>q</sub>, and **b**<sub>k</sub> for BERT<sub>BASE</sub> on low-date SST-2 dataset:

```
python run_BEFT.py 
       --task-name sst2\  
       --model-name bert-base-cased\
       --bias-terms-loop True
```

