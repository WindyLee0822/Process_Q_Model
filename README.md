<div align="center">

# Process Reward Model with Q-value Rankings

</div>


## Introduction
<div align="center">
<img src="figures/PQM.png" width="822px">
</div>

We present a new framework for PRM by framing it as a $Q$-value ranking problem, providing a theoretical basis for reward modeling that captures inter-dependencies among reasoning states.
We also show that prior classification-based PRM can be cast as a special case under our framework.
We validate its effectiveness through comprehensive experiments and ablation studies on a wide range of sampling policies, LLM backbones, and different test sets. 


## Reproduction

### Train
#### Main Experiments
To reproduce experimental results of main experiments, please run

```
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 trian_main.py
```

There are some arguments you could use,

```
--model-path <path of LLM backbone>
--dataset-path <path of the Math-Shepherd corpus>
--save-path <path to save checkpoints>
--loss-type <select from [rank,orm,mse,bce]>
--zeta <hyperparameter of our loss as in Eq.10 of our paper>
```

The choice [rank,orm,mse,bce] of loss-type refers to our comparative loss of PQM, outcome reward model (ORM), MSE loss and BCE loss. 

#### Ablation Experiments

To reproduce experimental results of main experiments, please run

```
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 train_ablation.py
```

There are some arguments you could use,

```
--model-path <path of LLM backbone>
--dataset-path <path of the Math-Shepherd corpus>
--save-path <path to save checkpoints>
--loss-type <select from [rank,theory-rank,ablate-rank]>
--zeta <hyperparameter of our loss as in Eq.10 of our paper>
```

The choice [rank,theory-rank,ablate-rank] of loss-type refers to the practical version as in Eq.10, the theoretical version as in Eq.9, the ablate version as in Eq.12.


### Evaluation

To obtain the Best-of-N results of PQM, please run

```
CUDA_VISIBLE_DEVICES=2,3 nohup deepspeed bon_eval_hf.py 
```

There are some arguments you could use

```
--backbone-path <the path of a LLM backbone to train your PQM>
--model-path <the checkpoint to be evaluated>
--data-name <select from [math,gsm8k], corresponding to MATH500 and GSM-Plus>
--data-file <the file of trajectories sampled by policy models>
--combine <1 when incorporating self-consistency, otherwise 0>
--baseline <1 when obtaining self-consistency and pass@k, otherwise 0>
--save-file <path to save your evaluation results>
```

### Checkpoints & Evaluation Data

We release the sampling corpus of three policies and PQM checkpoints on 🤗[huggingface](https://huggingface.co/Windy0822/PQM/)


## Citation
To be supplemented
