

# The Unsupervised Reinforcement Learning Suite (URLS)

URLS aims to provide a set of unsupervised reinforcement learning algorithms and experiments for the purpose of researching the applicability of unsupervised reinforcement learning to a variety of paradigms. 

The codebase is based upon [URLB](https://github.com/rll-research/url_benchmark) and [ExORL](https://github.com/denisyarats/exorl). Further details are provided in the following papers:

- [URLB: Unsupervised Reinforcement Learning Benchmark](https://arxiv.org/abs/2110.15191)
- [Don't Change the Algorithm, Change the Data: Exploratory Data for Offline Reinforcement Learning](https://arxiv.org/abs/2201.13425)

URLS is intended as a successor to URLB allowing for an increased number of experiments and RL paradigms.

## Prerequisites

Install [MuJoCo](http://www.mujoco.org/) if it is not already the case:

* Download MuJoCo binaries [here](https://mujoco.org/download).
* Unzip the downloaded archive into `~/.mujoco/`.
* Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`.

Install the following libraries:
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 unzip
```

Install dependencies:
```sh
conda env create -f conda_env.yml
conda activate urls-env
```

****

## Workflow

We provide the following workflows:

### Unsupervised Reinforcement Learning
  **Pre-training**, learn from agents intrinsic reward on a specific domain
  ```sh
  python pretrain.py agent=UNSUPERVISED_AGENT domain=DOMAIN
  ```
  **Fine-tuning**, learn with the pre-trained agent on a specific, task specific reward is now used for the agent
  ```sh
  python finetune.py pretrained_agent=UNSUPERVISED_AGENT task=TASK snapshot_ts=TS obs_type=OBS_TYPE
  ```

### Offline Learning from Unsupervised Reinforcement Learning
  **Pre-training**, learn from agents intrinsic reward on a specific domain
  ```sh
  python pretrain.py agent=UNSUPERVISED_AGENT domain=DOMAIN
  ```
  **Sampling**, sampledemos from agent replay buffer on a specific domain
  ```sh
  python sampling.py agent=UNSUPERVISED_AGENT domain=DOMAIN samples=SAMPLES snapshot_ts=TS obs_type=OBS_TYPE
  ```
  **Offline-learning**, learn a policy using the offline data collected on the specific task.
  ```sh
  python train_offline.py agent=OFFLINE_AGENT expl_agent=UNSUPERVISED_AGENT task=TASK
  ```

****

## Unsupervised Agents
The following unsupervised reinforcement learning agents are available, replace `UNSUPERVISED_AGENT` with **Command**. 
For example to use DIAYN, set `UNSUPERVISED_AGENT` = `diayn`.

| Agent | Command | Type | Implementation Author(s) | Paper | Intrinsic Reward
|---|---|---|---|---|---|
| ICM | `icm` | Knowledge | Denis | [paper](https://arxiv.org/abs/1705.05363)| $\| \| g(\mathbf{z}_{t+1} \| \mathbf{z}_{t}, \mathbf{a}_{t}) - \mathbf{z}_{t+1} \| \| ^{2}$
| Disagreement| `disagreement`| Knowledge | Catherine | [paper](https://arxiv.org/abs/1906.04161) |  $Var\{ g_{i} (\mathbf{z}_{t+1} \| \mathbf{z}_{t}, \mathbf{a}_{t}) \}$
| RND | `rnd`| Knowledge | Kevin | [paper](https://arxiv.org/abs/1810.12894) | $\| \| g(\mathbf{z}_{t}, \mathbf{a}_{t}) - \tilde{g}(\mathbf{z}_{t}, \mathbf{a}_{t}) \| \| ^{2}_{2}$
| APT(ICM) | `icm_apt` | Data | Hao, Kimin | [paper](https://arxiv.org/abs/2103.04551)| $\sum_{j \in random} \log \| \| \mathbf{z}_{t} - \mathbf{z}_{j} \| \|$
| APT(Ind) | `ind_apt` | Data | Hao, Kimin | [paper](https://arxiv.org/abs/2103.04551)| $\sum_{j \in random} \log \| \| \mathbf{z}_{t} - \mathbf{z}_{j} \| \|$
| ProtoRL | `proto` | Data | Denis | [paper](https://arxiv.org/abs/2102.11271)| $\sum_{j \in random} \log \| \| \mathbf{z}_{t} - \mathbf{z}_{j} \| \|$
| DIAYN | `diayn` | Competence | Misha | [paper](https://arxiv.org/abs/1802.06070)| $\log q(\mathbf{w}\|\mathbf{z}) + const$
| APS | `aps` | Competence | Hao, Kimin | [paper](http://proceedings.mlr.press/v139/liu21b.html)| $r_{t}^{APT}(\mathbf{z}) + \log q(\mathbf{z} \| \mathbf{w})$
| SMM | `smm` | Competence | Albert | [paper](https://arxiv.org/abs/1906.05274) | $\log p^{*}(\mathbf{z}) - \log q_{\mathbf{w}}(\mathbf{z}) - \log p(\mathbf{w}) + \log d(\mathbf{w} \| \mathbf{z})$

****

## Offline Agents

The following 5 RL procedures are available to learn a policy offline from unsupervised data. Replace `OFFLINE_AGENT` with **Command**, for example to use behavioral cloning, set `OFFLINE_AGENT` = `bc`.

| Offline RL Procedure | Command | Paper |
|---|---|---|
| Behavior Cloning | `bc` |  [paper](https://proceedings.neurips.cc/paper/1988/file/812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf)|
| CQL | `cql` |  [paper](https://arxiv.org/pdf/2006.04779.pdf)|
| CRR | `crr` |[paper](https://arxiv.org/pdf/2006.15134.pdf)|
| TD3+BC | `td3_bc` | [paper](https://arxiv.org/pdf/2106.06860.pdf) |
| TD3 | `td3` | [paper](https://arxiv.org/pdf/1802.09477.pdf)|

**** 

## Environments

The following environments with specific domains and tasks are provided. We also provide a [wrapper](utils/wrappers/gym_wrapper.py) to convert Gym environments to DMC extended time-step types based on DeepMind's [acme wrapper](https://github.com/deepmind/acme/blob/master/acme/wrappers/gym_wrapper.py).

Environment Type | Domain | Task |
|---|---|---|
Deep Mind Control | `walker` | `stand`, `walk`, `run`, `flip` |
Deep Mind Control | `quadruped` | `walk`, `run`, `stand`, `jump` |
Deep Mind Control | `jaco` | `reach_top_left`, `reach_top_right`, `reach_bottom_left`, `reach_bottom_right`
Deep Mind Control | `cheetah` | `run` | `run_backward`
Gym Box2D | `BipedalWalker-v3` | `walk`
Gym Box2D | `CarRacing-v1` | `race`
Gym Classic Control | `MountainCarContinuous-v0` | `goal`
Safe Control | `SimplePointBot` | `goal`


## License
The majority of URLS including the ExORL & URLB based code is licensed under the MIT license, however portions of the project are available under separate license terms: DeepMind is licensed under the Apache 2.0 license.
