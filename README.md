# Navigation_RLND 

This repository contains my submission for the Navigation Project within the Udacity Deep Reinforcement Learning Nanodegree:
<https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation>

To solve this environment three value based methods were impleemnted:

* Deep Q Networks (original paper <https://www.nature.com/articles/nature14236>)
* Double Deep Q Netowrks (original paper <https://arxiv.org/abs/1509.06461>)
* Prioritized Experience Replay (original paper <https://arxiv.org/abs/1511.05952>) 

## Environment

![image](https://github.com/catarina-p/Navigation_RLND/assets/92151172/8d66bb1b-b6bc-49c0-b8ea-3f7ca7d70fd4)


Collecting a yellow banana (resp. a blue banana) gives a reward of 1 (resp. -1). The state space is a continuous 37 dimensional space. The action space is discrete $\mathcal A =\{0,1,2,3\}$ with

* 0 - move forward
* 1 - move backward
* 2 - turn left
* 3 - turn right.
  
The task is episodic and the environment is considered solved when the agent get an average score of 13 over 100 consecutive episodes.

## Installing the environment and dependencies

* Download the environment:
  * Linux: **[download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)**
  * Mac OSX: **[download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)**
  * Windows (32-bit): **[download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)**
  * Windows (64-bit): **[download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)**
* Unzip
* In the code, import the UnityEnvironment as follow (the file_name should target the reader's own *Banana.exe*):

```python
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name="Banana_Windows_x86_64\Banana.exe")
```

* Python version used: python==3.6
  * Linux or Mac:
    ```python
    conda create --name drlnd python=3.6
    source activate drlnd
    pip install -r requirements.txt
    ```
  
  * Windows:
    ```python
    conda create --name drlnd python=3.6
    conda activate drlnd
    pip install -r requirements.txt
    ```

## Getting started
Please follow the BananasNavigation_project.ipynb notebook. Here you can either train the model on your mahine, or load the pretrained weights for every case.
