# E. Generative models


**Lab:** [lucidrains/denoising-diffusion-pytorch — diffusion reference](https://github.com/lucidrains/denoising-diffusion-pytorch)

| Rung | Goal | Module |
| ---- | ---- | ------ |
| E1 | VAE on MNIST/FashionMNIST | `e1_vae.py` |
| E2 | DCGAN on a small image dataset | `e2_dcgan.py` |
| E3 | DDPM diffusion model from scratch | `e3_ddpm.py` |
| E4 | Compare sample quality and training stability across VAE, GAN, and diffusion | `e4_generative_comparison.py` |

Run any rung with `ladder run E1`; `--help` lists its flags. Outputs land
in `artifacts/<rung>/<timestamp>/`.
