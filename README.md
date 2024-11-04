# Data-driven multiscale modeling of subgrid parameterizations in climate models

This repository stores the code associated with our paper available
from [arXiv][arxiv] and appeared at an earlier stage at the [ICLR 2023
workshop][ccai] on Tackling Climate Change with Machine Learning.

The code in this repository can be used to recreate our experiments or
to modify our approach to work with additional systems. During our own
work we made use of the [Apptainer][apptainer] (or alternatively
[Singularity][singularity]) container system, and our definition file
[closure.def](closure.def) can be used to build a container including
JAX and required dependencies.

```console
$ apptainer build closure.sif closure.def
```

This produces a container image `closure.sif` which can be used to run
our software. For manual environment setup, dependencies are listed in
[requirements.txt](requirements.txt) and can be installed using
[pip][pip].

## Citing

If you make use of this software, please cite the associated paper:
```bibtex
@article{multiscaleclosure24,
  author={Karl Otness and Laure Zanna and Joan Bruna},
  title={Data-driven multiscale modeling of subgrid parameterizations in climate models},
  journal={arXiv Preprint},
  year={2024},
  url = {https://arxiv.org/abs/2303.17496}
}
```

## License

The software in this repository is made available under the terms of
the MIT license. See [LICENSE.txt](LICENSE.txt) for details.

[arxiv]: https://arxiv.org/abs/2303.17496
[ccai]: https://www.climatechange.ai/papers/iclr2023/60
[singularity]: https://sylabs.io/docs/
[apptainer]: https://apptainer.org/
[pip]: https://pip.pypa.io/
[pyqgjax]: https://github.com/karlotness/pyqg-jax
