# Data-driven multiscale modeling for correcting dynamical systems

[![Zenodo](https://zenodo.org/badge/983620273.svg)][zenodo]

This repository stores the code associated with [our paper][mlst] also
available on [arXiv][arxiv] and which appeared at an earlier stage at
the [ICLR 2023 workshop][ccai] on Tackling Climate Change with Machine
Learning where it won "Best ML Innovation".

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
@article{multiscaleclosure25,
  author={Karl Otness and Laure Zanna and Joan Bruna},
  title={Data-driven multiscale modeling for correcting dynamical systems},
  journal={Machine Learning: Science and Technology},
  year={2025},
  doi={10.1088/2632-2153/ae1a36}
}
```

## License

The software in this repository is made available under the terms of
the MIT license. See [LICENSE.txt](LICENSE.txt) for details.

[mlst]: https://doi.org/10.1088/2632-2153/ae1a36
[arxiv]: https://arxiv.org/abs/2303.17496
[ccai]: https://www.climatechange.ai/papers/iclr2023/60
[singularity]: https://sylabs.io/docs/
[apptainer]: https://apptainer.org/
[pip]: https://pip.pypa.io/
[pyqgjax]: https://github.com/karlotness/pyqg-jax
[zenodo]: https://doi.org/10.5281/zenodo.17645365
