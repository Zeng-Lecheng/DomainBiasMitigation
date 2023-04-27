This an improvement to the "Domain Independent Training" method in the paper. Clone this repo and follow the authors' instructions below. Run `independent_improve.ps1` (Powershell, Windows) or `independent_improve.sh` (Bash, Linux and macOS). Results goes to `record/cifar-s_domain_independent/update_in_turn` and `record/cifar-s_domain_independent/sum_loss_up`

How did we improve it? We implemented the basic idea in Domain Independent Training, which is to train one classifier for each domain, which is isolated to other domains. It is coded in `ResNetDuoOut` in `basenet.py`.I tried two ways in updating the network with duo outputs. One with lower bias is in master branch and the other is in `dev1_n`. 

----
# Effective Strategies for Bias Mitigation
Code for the CVPR paper:

[Towards Fairness in Visual Recognition: Effective Strategies for Bias Mitigation](https://arxiv.org/abs/1911.11834v2)

Zeyu Wang, Klint Qinami, Ioannis Christos Karakozis, Kyle Genova, Prem Nair, Kenji Hata, Olga Russakovsky

```
@inproceedings{wang2020fair,
author = {Zeyu Wang and Klint Qinami and Ioannis Karakozis and Kyle Genova and Prem Nair and Kenji Hata and Olga Russakovsky},
title = {Towards Fairness in Visual Recognition: Effective Strategies for Bias Mitigation},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2020}
}
```

## Requirements
* Python 3.6+
* PyTorch 1.0+
* h5py
* tensorboardX
* tqdm

## Data Preparation
First download and unzip the CIFAR-10 and CINIC-10 by running the script `download.sh`

Then manually download the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), put `Anno` into `data/celeba/Anno`, `Eval` into `data/celeba/Eval`, put all align and cropped images to `data/celeba/images`

Run the `preprocess_data.py` to generate data for all experiments (this step involves creating h5py file for CelebA images, so would take some time 1~2 hours)

## Run Experiments
To conduct experiments, run `main.py` with corresponding arguments (`experiment` specifies which experiment to run, `experiment_name` specifies a name to this experiment for saving the model and result). For example:

```
python main.py --experiment celeba_baseline --experiment_name e1 --random_seed 1
```

After running, the experiment result will be saved under `record/experiment/experiment_name`
