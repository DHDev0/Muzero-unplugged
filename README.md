# Muzero Unplugged

Pytorch Implementation of [Muzero Unplugged](https://arxiv.org/abs/2104.06294). Base on [Muzero](https://github.com/DHDev0/Muzero/) and incorporate new feature of muzero unplugged.

Work in progress...




Docker
------
 
Build image: (building time: 22 min , memory consumption: 8.75 GB)
~~~bash
docker build -t muzero .
~~~ 
(do not forget the ending dot)

Start container:
~~~bash
docker run --cpus 2 --gpus 1 -p 8888:8888 muzero
#or
docker run --cpus 2 --gpus 1 --memory 2000M -p 8888:8888 muzero
#or
docker run --cpus 2 --gpus 1 --memory 2000M -p 8888:8888 --storage-opt size=15g muzero
~~~ 

The docker run will start a jupyter lab on https://localhost:8888//lab?token=token (you need the token) with all the necessary dependency for cpu and gpu(Nvidia) compute.

Option meaning:  
--cpus 2 -> Number of allocated (2) cpu core  
--gpus 1 -> Number of allocated (1) gpu  
--storage-opt size=15gb -> Allocated storage capacity 15gb (not working with windows WSL)  
--memory 2000M -> Allocated RAM capacity of 2GB  
-p 8888:8888 -> open port 8888 for jupyter lab (default port of the Dockerfile)  

Stop the container:
~~~bash
docker stop $(docker ps -q --filter ancestor=muzero)
~~~ 

Delete the container:
~~~bash
docker rmi -f muzero
~~~ 




CLI
-----------


Construct your dataset through experimentation.
~~~bash 
python muzero_cli.py human_buffer config/name_config.json
~~~
Training :
~~~bash 
python muzero_cli.py train config/name_config.json
~~~  

Training with report
~~~bash
python muzero_cli.py train report config/name_config.json
~~~  

Inference (play game with specific model) :
~~~bash 
python muzero_cli.py train play config/name_config.json
~~~ 

Training and Inference :
~~~bash 
python muzero_cli.py train play config/name_config.json
~~~  

Benchmark model :
~~~bash
python muzero_cli.py benchmark config/name_config.json
~~~ 

Training + Report + Inference + Benchmark :
~~~python 
python muzero_cli.py train report play benchmark play config/name_config.json
~~~  

Features
========


Core Muzero feature:
* [x] Work for any Gymnasium environments/games. (any combination of continous or/and discrete action and observation space)
* [x] MLP network for game state observation. (Multilayer perceptron)
* [x] LSTM network for game state observation. (LSTM)
* [x] Transformer decoder for game state observation. (Transformer)
* [x] Residual network for RGB observation using render. (Resnet-v2 + MLP)
* [x] Residual LSTM network for RGB observation using render. (Resnet-v2 + LSTM)
* [x] MCTS with 0 simulation (use of prior) or any number of simulation.
* [x] Model weights automatically saved at best selfplay average reward.
* [x] Priority or Uniform for sampling in replay buffer.
* [X] Scale the loss using the importance sampling ratio.
* [x] Custom "Loss function" class to apply transformation and loss on label/prediction.
* [X] Load your pretrained model from tag number.
* [x] Single player mode.
* [x] Training / Inference report. (not live, end of training)
* [x] Single/Multi GPU or Single/Multi CPU for inference, training and self-play.
* [x] Support mix precision for training and inference.(torch_type: bfloat16,float16,float32,float64)
* [X] Pytorch gradient scaler for mix precision in training.
* [x] Tutorial with jupyter notebook.
* [x] Pretrained weights for cartpole. (you will find weight, report and config file)
* [x] Commented with link/page to the paper.
* [x] Support : Windows , Linux , MacOS.
* [X] Fix pytorch linear layer initialization. (refer to : https://tinyurl.com/ykrmcnce)




The Muzero Reanalyze new add-on features include:  
  
* [X] The ability to accommodate any number of players with the provision of player cycle information.
* [X] The incorporation of reanalyze buffer and reanalyze ratio functionality.
* [X] The capability to construct human play datasets through experimentation (CLI only).
* [X] The capability to load human play datasets into the Demonstration buffer or Replay buffer for training.
* [X] The ability to specify the amount of sampled action that MCTS should utilize.
* [X] The implementation of a priority scale on neural network and replay buffer priority.
* [X] Various options for bounding, saving, and deleting games from the reanalyze buffer.
* [X] The introduction of the reanalyze_fraction_mode, which allows for the statistical or  
quantitative switch between new game and reanalyze data with a ratio of reanalyze buffer vs replay buffer."

TODO:
* [ ] Support of [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) 0.27.0
* [ ] Hyperparameter search. (pseudo-code available in self_play.py)
* [ ] Unittest the codebase and assert argument bound.
* [ ] Training and deploy on cloud cluster using Kubeflow, Airflow or Ray for aws,gcp and azure.


Authors  
==========

- [Daniel Derycke](https://github.com/DHDev0)  

Subjects
========

Deep reinforcement learning


License 
=======

[GPL-3.0 license](https://www.gnu.org/licenses/quick-guide-gplv3.html)  

