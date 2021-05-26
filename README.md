# Byzantine_AirComp

Code for our paper "Byzantine-Resilient Federated Machine Learning via Over-the-Air Computation" (https://arxiv.org/abs/2105.10883).

## Environment Configuration

At first, you should build your environment via **pip** or **conda**.

```
pip install -r requirements.txt
conda install --yes --file requirements.txt
```

## Quick Start

### Ideal Byzantine-Resilient FL
```
python MNIST_Air_weight.py --agg gm2

python MNIST_Air_weight.py --attack classflip --agg gm2 --K 50 --B 5

python MNIST_Air_weight.py --attack classflip --agg gm2 --K 50 --B 10
```
### AirComp-Aid Byzantine-Resilient FL
```
python MNIST_Air_weight.py --var 1e-2 --mark 0-50

python MNIST_Air_weight.py --attack classflip --var 1e-2 --K 50 --B 5

python MNIST_Air_weight.py --attack classflip --var 1e-2 --K 50 --B 10
```