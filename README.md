# algotrade_research


### Prediction


Documentation is available as comments in the code.

`main.py` acts as a driver that runs the code in `train_models.py`.

`model_evaluation.py` is deprecated code that is currently not in use at all. With new evaluation code, it should be deleted.

`models/` contains the actual models used in prediction, and are imported by `train_models.py` as configured. `models/README.md` contains some more details.


### Reinforcement Learning (rl)


Documentation is available as comments in the code.

`dailyenv.py` contains the environment for RL, structured as an OpenAI Gym environment.

`run_baseline.py` contains code for running a baseline algorithm, which is SEPARATE from any reinforcement learning. Code in this file is not referenced by any other file.

`train_rl.py` actually trains the models. Currently the code is hardcoded to use A2C but should be developed to include other algorithms too like PPO. Keep in mind that the default hparams are different for A2C and PPO but the code doesn't consider that, so a drop-in replacement will not fully work.

`automl.py` uses the library Optuna for AutoML tuning of hyperparameters, referencing training code in `train_rl.py`.