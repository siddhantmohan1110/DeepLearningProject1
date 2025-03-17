# ECE-GY 7123 Deep Learning Project 1 - Mean Square Terror
by and Siddhant Mohan (sm12766@nyu.edu), Akshat Mishra (am15111@nyu.edu) and Jay Daftari (jd5829@nyu.edu).

This project helps train and test a modified version of ResNet having <5M params.

A fork of [pytorch-cifar](url).

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
