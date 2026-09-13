# Broader machine learning — reading list

Statistical foundations through geometric deep learning and ML systems: the
ground beneath the post-training and safety tracks. The rungs in
[ladder.md](ladder.md) are the hands-on counterpart — each part below lines up
with the part of the ladder that shares its subject.

Rendered source: [`source/ml-reading-list.html`](source/ml-reading-list.html).

## Anchor books

| | Book | Why |
| --- | --- | --- |
| modern | [Understanding Deep Learning](https://udlbook.github.io/udlbook/) — Simon J. D. Prince | MIT Press, 2023; free web/PDF. Rigorous and notation-heavy, a good fit for a maths background. |
| classical | [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) — Hastie, Tibshirani, Friedman | Springer, 2nd ed.; free PDF. The classical-statistics grounding under everything else. |
| probabilistic | [Machine Learning: A Probabilistic Perspective](https://probml.github.io/pml-book/) — Kevin P. Murphy | MIT Press, 2012, and the newer two-volume *Probabilistic Machine Learning*. Every method as inference under a probabilistic model — Bayesian foundations, graphical models, HMMs, Gaussian processes. |

## 1. Statistical learning foundations

- [A Few Useful Things to Know about Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) — Domingos. The informal survey of what actually matters in practice. *(paper)*
- [A Theory of the Learnable](https://web.mit.edu/6.435/www/Valiant84.pdf) — Valiant. Origin of PAC learning. *(paper)*
- [The Lack of A Priori Distinctions Between Learning Algorithms](https://direct.mit.edu/neco/article-abstract/8/7/1341/6016) — Wolpert. The No Free Lunch theorems. *(paper)*
- [Stanford CS229 — Machine Learning](https://cs229.stanford.edu/) — bias-variance, VC dimension, regularization from first principles. *(course)*

## 2. Classical ML algorithms → ladder part A

- [Support-Vector Networks](https://link.springer.com/article/10.1007/BF00994018) — Cortes & Vapnik, 1995. *(paper)*
- [Random Forests](https://link.springer.com/article/10.1023/A:1010933404324) — Breiman, 2001. *(paper)*
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) *(paper)*
- [Mathematics for Machine Learning](https://mml-book.github.io/) — Deisenroth, Faisal, Ong. Free PDF; linear algebra, PCA, optimization worked from scratch. *(book)*

## 3. Deep learning foundations → ladder part B

- [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) — Rumelhart, Hinton, Williams, 1986. *(paper)*
- [Deep Learning](https://www.deeplearningbook.org/) — Goodfellow, Bengio, Courville. Free web version. *(book)*
- [Batch Normalization](https://arxiv.org/abs/1502.03167) *(paper)*
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) *(paper)*
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) *(paper)*
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — ResNet. *(paper)*

## 4. Representation & self-supervised learning → ladder part D

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — word2vec. *(paper)*
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) *(paper)*
- [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) — SimCLR. *(paper)*
- [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) — MoCo. *(paper)*
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) — MAE. *(paper)*

## 5. Sequence models & attention → ladder part C

- [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — Hochreiter & Schmidhuber, 1997. *(paper)*
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) *(paper)*
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) *(paper)*
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) — ViT. *(paper)*
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP; line-by-line implementation walkthrough. *(course)*

## 6. Generative models → ladder part E

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE. *(paper)*
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) *(paper)*
- [Density Estimation Using Real NVP](https://arxiv.org/abs/1605.08803) — normalizing flows. *(paper)*
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM. *(paper)*
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) *(paper)*

## 7. Geometric & equivariant deep learning → ladder part F

*Matches the algebraic topology / equivariant cohomology background.*

- [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://geometricdeeplearning.com/) — Bronstein, Bruna, Cohen, Veličković. The unifying survey/proto-textbook. *(book)*
- [Group Equivariant Convolutional Networks](https://arxiv.org/abs/1602.07576) — Cohen & Welling. *(paper)*
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Kipf & Welling. *(paper)*
- [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) — GraphSAGE. *(paper)*

## 8. Scaling & ML systems → ladder part G

*Adjacent to ignite-catalog.*

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al. *(paper)*
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla. *(paper)*
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) *(paper)*
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) *(paper)*
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) *(paper)*

## 9. Companion courses

- [Stanford CS231n — CNNs for Visual Recognition](http://cs231n.stanford.edu/) *(course)*
- [Stanford CS224n — NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) *(course)*
- [fast.ai — Practical Deep Learning for Coders](https://course.fast.ai/) *(course)*
