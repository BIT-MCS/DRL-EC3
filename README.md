# Edics
This is the code accompanying the paper: "[Energy-Efficient UAV Control for Effective and Fair Communication Coverage: A Deep Reinforcement Learning Approach](https://ieeexplore.ieee.org/document/8432464)", published in JSAC.

## :page_facing_up: Description
Unmanned aerial vehicles (UAVs) can be used to serve as aerial base stations to enhance both the coverage and performance of communication networks in various scenarios, such as emergency communications and network access for remote areas. Mobile UAVs can establish communication links for ground users to deliver packets. However, UAVs have limited communication ranges and energy resources. Particularly, for a large region, they cannot cover the entire area all the time or keep flying for a long time. It is thus challenging to control a group of UAVs to achieve certain communication coverage in a long run, while preserving their connectivity and minimizing their energy consumption. Toward this end, we propose to leverage emerging deep reinforcement learning (DRL) for UAV control and present a novel and highly energy-efficient DRL-based method, which we call DRL-based energy-efficient control for coverage and connectivity ($DRL-EC^3$ ). The proposed method 1) maximizes a novel energy efficiency function with joint consideration for communications coverage, fairness, energy consumption and connectivity; 2) learns the environment and its dynamics; and 3) makes decisions under the guidance of two powerful deep neural networks. We conduct extensive simulations for performance evaluation.

## :wrench: Installation
1. Clone repo
    ```bash
    git clone https://github.com/BIT-MCS/DRL-EC3.git
    cd DRL-EC3
    ```
2. Install dependent packages
    ```sh
    conda create -n mcs python==3.8
    conda activate mcs
    pip install tensorflow-gpu==1.15
    pip install -r requirements.txt
    ```


## :computer: Training

Train our solution
```bash
python experiments/train.py
```
## :checkered_flag: Testing

Test with the trained models 

```sh
python experiments/test.py --load-dir=your_model_path
```

Random test the env

```sh
python experiments/test_random.py
```

## :clap: Reference
- https://github.com/openai/maddpg


## :scroll: Acknowledgement

This work was supported in part by the National Natural Science Foundation of China under Grant 61772072 and in part by the National Key Research and Development Program of China under Grant 2018YFB1003701.
<br>
Corresponding author: Chi Harold Liu.

## :e-mail: Contact

If you have any question, please email `daizipeng@bit.edu.cn`.

## Paper
If you are interested in our work, please cite our paper as

```
@ARTICLE{liu2018energy,
    author={Liu, Chi Harold and Chen, Zheyu and Tang, Jian and Xu, Jie and Piao, Chengzhe},
    journal={IEEE Journal on Selected Areas in Communications (JSAC)},
    title={Energy-Efficient UAV Control for Effective and Fair Communication Coverage: A Deep Reinforcement Learning Approach},
    year={2018},
    volume={36},
    number={9},
    pages={2059-2070},
}
```
