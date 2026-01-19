# gr-libs

`gr-libs` is a Python library for **Goal Recognition (GR)** in Markov Decision Processes (MDPs).  
It provides implementations of multiple online and offline GR algorithms built on top of the **Gymnasium API**.

All agents interact with Gym environments registered by the companion package **`gr-envs`**, which is a required dependency.  
See: https://github.com/MatanShamir1/gr_envs

<table> <tr> <td width="40%"> <img src="gifs/point_maze.gif" width="200"/> </td> <td> A simple example of an agent trajectory used for goal recognition. Detailed descriptions of all supported domains, environment instances, and algorithms are documented separately in [gr_libs/Domains_and_algorithms.md](gr_libs/Domains_and_algorithms.md).
 </td> </tr> </table>


## 📖 Table of Contents

- [📖 Table of Contents](#-table-of-contents)
- [⚙️ Setup](#-setup)
- [🧩 Configuration](#-configuration)
- [🚀 Running](#-running)
- [📈 Scaling Experiments](#-scaling-experiments)
- [🛠️ For Developers](#️-for-developers)
- [🌐 Related Links](#-related-links)
- [📄 License](#-license)
- [🤝 Acknowledgments](#acknowledgments)
---

## ⚙️ Setup

### Requirements
- Python **≥ 3.11**
- Linux (recommended)
- Windows users: use **Git Bash** or **WSL**. Microsoft Visual C++ Build Tools ≥ 14.0 is required

---

### 1. Create a Python Environment (recommended)

#### Linux / macOS
```sh
python3.12 -m venv gr_env
source gr_env/bin/activate
````

#### Windows (Git Bash)

```sh
py -3.12 -m venv gr_env
source gr_env/Scripts/activate
```

---

### 2. Upgrade Packaging Tools

```sh
pip install --upgrade pip setuptools wheel versioneer
```

---

### 3. Install `gr-libs`

`gr-libs` uses extras to install environment-specific dependencies.

```sh
pip install gr_libs[minigrid]
```

Install additional environment extras:

```sh
pip install gr_libs[maze]
pip install gr_libs[highway]
pip install gr_libs[panda]
```

---

### Editable Installation (developers)

```sh
git clone https://github.com/MatanShamir1/gr_libs.git
cd gr_libs
pip install -e .
```

---

## 🧩 Configuration

### Download Trained Agents and Caches

For a quick initial run, it is recommended to download the pretrained agents and caches using the CLI:

```sh
download-gr-libs-dataset
```

Specify a target directory if needed:

```sh
download-gr-libs-dataset --extract_to /path/to/data
```

This will create:

```
Grlib/trained_agents/
Grlib/gr_cache/
```

### Alternative Dataset Access

#### Run from source

```sh
git clone https://github.com/MatanShamir1/gr_libs.git
cd gr_libs
pip install gdown
python download_dataset.py
```

#### Docker (dataset included)

```sh
docker pull ghcr.io/matanshamir1/gr_image:latest
docker run -it ghcr.io/matanshamir1/gr_image:latest bash
```

Install `gr-libs` inside the container using the instructions above.

---

## 🚀 Running

`gr-libs` supports **Online Dynamic Goal Recognition (ODGR)** workflows.
You can either write a custom Python script or run experiments via configuration files.

---

### Method 1: Python Script

```python
from gr_libs import Graql
from gr_libs.environment._utils.utils import domain_to_env_property
from gr_libs.environment.environment import MINIGRID, QLEARNING
from gr_libs.metrics.metrics import stochastic_amplified_selection
from gr_libs.ml.tabular.tabular_q_learner import TabularQLearner
from gr_libs.ml.utils.format import random_subset_with_order


def run_graql_minigrid_tutorial():
    recognizer = Graql(domain_name="minigrid", env_name="MiniGrid-SimpleCrossingS13N4")

    # Graql doesn't have a domain learning phase, so we skip it

    recognizer.goals_adaptation_phase(
        dynamic_goals=[(11, 1), (11, 11), (1, 11)],
        dynamic_train_configs=[
            (QLEARNING, 100000) for _ in range(3)
        ],  # for expert sequence generation.
    )

    property_type = domain_to_env_property(MINIGRID)
    env_property = property_type("MiniGrid-SimpleCrossingS13N4")

    # TD3 is different from recognizer and expert algorithms, which are SAC #
    actor = TabularQLearner(
        domain_name="minigrid",
        problem_name="MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",
        env_prop=env_property,
        algorithm=QLEARNING,
        num_timesteps=100000,
    )
    actor.learn()
    # sample is generated stochastically to simulate suboptimal behavior, noise is added to the actions values #
    full_sequence = actor.generate_observation(
        action_selection_method=stochastic_amplified_selection,
        random_optimalism=True,  # the noise that's added to the actions
    )

    partial_sequence = random_subset_with_order(
        full_sequence, (int)(0.5 * len(full_sequence)), is_consecutive=False
    )
    closest_goal = recognizer.inference_phase(partial_sequence, (11, 1), 0.5)
    print(
        f"closest_goal returned by Graql: {closest_goal}\nactual goal actor aimed towards: (11, 1)"
    )
    return closest_goal, (11, 1)


if __name__ == "__main__":
    run_graql_minigrid_tutorial()
```

---

### Method 2: Configuration-Based Execution

Predefined ODGR tasks are defined in `consts.py`.

Run an experiment:

```sh
python odgr_executor.py \
  --recognizer ExpertBasedGraml \
  --domain minigrid \
  --env_name MiniGrid-SimpleCrossingS13N4 \
  --task L1
```

Add statistics collection:

```sh
--collect_stats
```

This produces:

* Summary `.pkl` and `.txt` files (accuracy, runtime)
* Trajectory visualizations (PNG / MP4)
* Pickled confidence or embedding outputs for offline analysis

---

## 📈 Scaling Experiments

Run many ODGR problems in parallel using `all_experiments.py`:

```sh
python gr_libs/all_experiments.py \
  --domains minigrid parking \
  --envs MiniGrid-SimpleCrossingS13N4 Parking-S-14-PC- \
  --tasks L1 L2 L3 L4 L5 \
  --recognizers ExpertBasedGraml Graql \
  --n 5
```

Results are aggregated into:

```
outputs/summaries/
```

Analysis and plotting tools are available in:

```
evaluation/
```

See `evaluation/README.md` for details.

---

## 🛠️ For Developers

For contributors:

```sh
pip install pre-commit
pre-commit install
```

CI, Docker images, and GitHub Actions are documented here:
[https://github.com/MatanShamir1/gr_libs/tree/main/CI](https://github.com/MatanShamir1/gr_libs/tree/main/CI)


---

## 🌐 Related Links

- **GR theoretical tutorial**: [A hub for multiple goal recognition resources](https://www.planrec.org/resources.html/)
- **GRasRL**: [Goal Recognition as Reinforcement Learning](https://arxiv.org/abs/2202.06356)
- **DRACO**: [Goal Recognition using Actor-Critic Optimization](https://arxiv.org/abs/2501.01463)
- **GRAML**: [Goal Recognition as Metric Learning](https://arxiv.org/abs/2505.03941)
- **GDGR**: [General Dynamic Goal Recognition using Goal-Conditioned and Meta Reinforcement Learning](https://arxiv.org/abs/2505.09737)


## 📄 License
This project is open source under the MIT License.

## 🤝 Acknowledgments
If you use SCP, please cite our technical report:
```bibtex
@misc{grlibs2025,
  title        = {gr-libs: A Python Library for Goal Recognition in Markov Decision Processes},
  author       = {Shamir, Matan and Nageris, Ben and Elhadad, Osher and Mirsky, Reuth},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/MatanShamir1/gr_libs}},
  note         = {Python library for online and offline dynamic goal recognition algorithms built on top of the Gymnasium API}
}

