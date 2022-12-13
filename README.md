# Visualizing Decision Boundaries of DNNs

### Installation
In order to run the code, we recommend using "conda" to set up the environment. Use below mentioned steps to install all the required libraries
1. Clone the repository to your local sytem and move to the root directory
   ```
   git clone https://git.opendfki.de/payal.varshney/visualizing-decision-boundaries-of-dnns.git
   ```
   ```
   cd visualizing-decision-boundaries-of-dnns
   ```

2. Create and activate a conda environment with python >= 3.8
    ```
    conda create -n myvenv python==3.8
    ```
    ```
    conda activate myvenv
    ```
3. Install required libraries 
   ```
   pip install .  --use-feature=in-tree-build
   ```

### Run the Code
After installation, the code can be run using following command
  ```
  python decision_boundaries/main.py --dataset DATASET --attack ATTACK --abs_stepsize STEP_SIZE --epsilon EPSILON
  ```
  **DATASET**: Supported datasets are: "cifar10", "mnist", "fashionmnist", and "stl10"

  **ATTACK**: Supported attacks are: "pgd" and "carlini_wagner"

  **STEPSIZE**: Step size value for applying adversarial attacks 

  **EPSILON**: Epsilon value for applying adversarial attacks

### Reproduce the Results
In order to reproduce the reported results use the below mentioned commands:

Visualizations using PGD Attack
```
python decision_boundaries/main.py --dataset cifar10 --attack PGD --abs_stepsize 0.0078125 --epsilon 0.0625
```
```
python decision_boundaries/main.py --dataset stl10  --attack PGD --abs_stepsize 0.0078125 --epsilon 0.0625
```
```
python decision_boundaries/main.py --dataset mnist --attack PGD --abs_stepsize 0.0078125 --epsilon 0.0625
```
```
python decision_boundaries/main.py --dataset fashionmnist --attack PGD --abs_stepsize 0.0078125 --epsilon 0.0625
```

Visualizations using Carlini Wagner Attack
```
python decision_boundaries/main.py --dataset cifar10 --attack Carlini_Wagner --abs_stepsize 0.01 --epsilon 5
```
```
python decision_boundaries/main.py --dataset stl10 --attack Carlini_Wagner --abs_stepsize 0.01 --epsilon 5
```
```
python decision_boundaries/main.py --dataset mnist --attack Carlini_Wagner --abs_stepsize 0.01 --epsilon 5
```
```
python decision_boundaries/main.py --dataset fashionmnist --attack Carlini_Wagner --abs_stepsize 0.01 --epsilon 5
```

### Milestones
1. Set-up repo with:
  - Function that receives a model, DS, projection method $\eta$ (+extra args) and
  - Plots samples of the DS as $eta(x) \in R^2$
    - Add color border that indicates the predicted class
  - Plot vornonoi diagram based on the result of k-means cluster of projected points $\eta(x), x\in$ DS
2. Integrate adversarial attacks to the repo by coupling the "Foolbox" library
  - PGD and Carlini-Wagner attacks should be implemented (a boundary-attack would be nice too)
  - Plot points (individually) as they move with each iteration of the attack.
    - Add alpha to fade previous points
    - Add directional arrows based on the direction where the attack would shift the current sample (all in the projected space)
3. Geneate plots for different datasets: MNIST, FashionMNIST, Cifar10, TinyImagenet/STL10
  - Try with different networks (large and small) to see differences of the projected space
  - Plot samples based on a notion of difficulty (e.g., MMD or the gradient's norm)
