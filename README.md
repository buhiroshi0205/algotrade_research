# algotrade_research


## Python Environment Requirements


Listed in `requirements.txt`. However for cuda-accelerated computing, manual installation of a cuda-supported pytorch first may be required before installing via requirements.txt.


## Prediction


Documentation is available as comments in the code.

`main.py` acts as a driver that runs the code in `train_models.py`.

`model_evaluation.py` is deprecated code that is currently not in use at all. With new evaluation code, it should be deleted.

`models/` contains the actual models used in prediction, and are imported by `train_models.py` as configured. `models/README.md` contains some more details.

### Training a model

`cd` to `prediction` directory first.

To run a single-threaded basic training: `python3 main.py`

To run multi-process (multi-GPU) training, change the `processes` variable to the desired number of processes/GPUs and run `python3 main.py`.

Hyperparameters and other configuration can be modified in the code, in `main.py/run()`.

### Visualizing results

We do not have an automatic tool for this yet.


## Reinforcement Learning (rl)


Documentation is available as comments in the code.

`dailyenv.py` contains the environment for RL, structured as an OpenAI Gym environment.

`run_baseline.py` contains code for running a baseline algorithm, which is SEPARATE from any reinforcement learning. Code in this file is not referenced by any other file.

`train_rl.py` actually trains the models. Currently the code is hardcoded to use A2C but should be developed to include other algorithms too like PPO. Keep in mind that the default hparams are different for A2C and PPO but the code doesn't consider that, so a drop-in replacement will not fully work.

`automl.py` uses the library Optuna for AutoML tuning of hyperparameters, referencing training code in `train_rl.py`.

### Training a model

`cd` to `rl` directory first.

To run a 1-trial basic training: `python3 train_rl.py` with `n_trials` < CPU core count (currently set to 1)

To run automl hparam tuning: `python3 automl.py` with `n_trials` < CPU core count (currently set to 20)

Hyperparameters and other configuration can be modified in the code, in `train_rl.py/run_once()` for basic training and in `automl.py/objective()` for automl.

### Visualizing results

Use tensorboard on directory `tensorboard_logs/{experiment_name}`, like `tensorboard --logdir tensorboard_logs/test`