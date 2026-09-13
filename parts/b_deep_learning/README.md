# B. Deep learning foundations


**Lab:** [karpathy/micrograd — autodiff reference implementation](https://github.com/karpathy/micrograd)

| Rung | Goal | Module |
| ---- | ---- | ------ |
| B1 | Scalar autodiff engine (micrograd-style) plus a tiny MLP trained on it | `b1_autodiff_engine.py` |
| B2 | MLP on MNIST — manual backprop, then repeat with PyTorch autograd and diff the gradients | `b2_mlp_mnist.py` |
| B3 | CNN on CIFAR-10 with a batchnorm/dropout ablation | `b3_cnn_cifar10.py` |
| B4 | ResNet block implementation, trained on CIFAR-10/100 | `b4_resnet_blocks.py` |
| B5 | Optimizer bake-off — SGD vs Adam vs AdamW, with LR-schedule comparison | `b5_optimizer_bakeoff.py` |

Run any rung with `ladder run B1`; `--help` lists its flags. Outputs land
in `artifacts/<rung>/<timestamp>/`.
