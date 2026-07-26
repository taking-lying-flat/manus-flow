## Manus Flow

All image-model examples use CIFAR-10 through the shared `dataloader.py`.
The training split is downloaded automatically to `data/`; no dataset or data-path
argument is required.

For example:

```bash
python 2023_lipman_flow_matching/image_gen.py --help
python 2023_lipman_flow_matching/image_gen.py
```

## Citation

```bibtex
@article{Kingma2013AutoEncodingVB,
  title={Auto-Encoding Variational Bayes},
  author={Diederik P. Kingma and Max Welling},
  journal={CoRR},
  year={2013},
  volume={abs/1312.6114},
  url={https://api.semanticscholar.org/CorpusID:216078090}
}

@article{Dinh2014NICENI,
  title={NICE: Non-linear Independent Components Estimation},
  author={Laurent Dinh and David Krueger and Yoshua Bengio},
  journal={arXiv: Learning},
  year={2014},
  url={https://api.semanticscholar.org/CorpusID:13995862}
}

@article{SohlDickstein2015DeepUL,
  title={Deep Unsupervised Learning using Nonequilibrium Thermodynamics},
  author={Jascha Narain Sohl-Dickstein and Eric A. Weiss and Niru Maheswaranathan and Surya Ganguli},
  journal={ArXiv},
  year={2015},
  volume={abs/1503.03585},
  url={https://api.semanticscholar.org/CorpusID:14888175}
}

@article{JimenezRezende2015VariationalIW,
  title={Variational Inference with Normalizing Flows},
  author={Danilo Jimenez Rezende and Shakir Mohamed},
  journal={ArXiv},
  year={2015},
  volume={abs/1505.05770},
  url={https://api.semanticscholar.org/CorpusID:12554042}
}

@inproceedings{dinh2017density,
  title={Density Estimation using Real {NVP}},
  author={Dinh, Laurent and Sohl-Dickstein, Jascha and Bengio, Samy},
  booktitle={International Conference on Learning Representations},
  year={2017},
  url={https://openreview.net/forum?id=HkpbnH9lx}
}

@article{Kingma2018GlowGF,
  title={Glow: Generative Flow with Invertible 1x1 Convolutions},
  author={Diederik P. Kingma and Prafulla Dhariwal},
  journal={ArXiv},
  year={2018},
  volume={abs/1807.03039},
  url={https://api.semanticscholar.org/CorpusID:49657329}
}

@article{chen2018neural,
  title={Neural ordinary differential equations},
  author={Chen, Ricky TQ and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David K},
  journal={Advances in neural information processing systems},
  volume={31},
  year={2018}
}

@inproceedings{NEURIPS2019_3001ef25,
  author={Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  editor={H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
  pages={},
  publisher={Curran Associates, Inc.},
  title={Generative Modeling by Estimating Gradients of the Data Distribution},
  url={https://proceedings.neurips.cc/paper_files/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf},
  volume={32},
  year={2019}
}

@article{Liu2022FlowSA,
  title={Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow},
  author={Xingchao Liu and Chengyue Gong and Qiang Liu},
  journal={ArXiv},
  year={2022},
  volume={abs/2209.03003},
  url={https://api.semanticscholar.org/CorpusID:252111177}
}

@inproceedings{lipman2023flow,
  title={Flow Matching for Generative Modeling},
  author={Lipman, Yaron and Chen, Ricky T. Q. and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matt},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=PqvMRDCJT9t}
}
```
