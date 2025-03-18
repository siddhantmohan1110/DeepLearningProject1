# ECE-GY 7123 Deep Learning Project 1 - Mean Square Terror
by Siddhant Mohan (sm12766@nyu.edu), Akshat Mishra (am15111@nyu.edu) and Jay Daftari (jd5829@nyu.edu).

This project helps train and test a modified version of ResNet having <5M params.

Forked and adapted from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).

Competed in this [Kaggle competition](https://www.kaggle.com/competitions/deep-learning-spring-2025-project-1/leaderboard) and achieved rank 21.

Train with
```
python3 train.py --exp_name exp1
```

Get CIFAR-10 test accuracy with
```
python3 test.py --exp_name exp1 --ckpt_name <ckpt_name>
```

Generate submission CSV for the no-label data with 
```
python3 test.py --exp_name exp1 --ckpt_name <ckpt_name> --nolabel 1
```

Carry out test-time augmentation with 
```
python3 test.py --exp_name exp1 --ckpt_name <ckpt_name> --tta 1
```

Also contains the experiment folder ```exp2_aug1``` whose results helped us achieve rank 21 in the competition. The folder consists of the accuracy and loss plot, the model summary, the best model file and the best submission.csv file.
