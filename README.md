# Langevin algorithms for Markovian Neural Networks and Deep Stochastic control

Training a very deep neural network is a challenging task, as the deeper a neural network is, the more non-linear it is.
We compare the performances of various preconditioned Langevin algorithms with their non-Langevin counterparts for the training of neural networks of increasing depth. For shallow neural networks, Langevin algorithms do not lead to any improvement, however the deeper the network is and the greater are the gains provided by Langevin algorithms. Adding noise to the gradient descent allows to escape from local traps, which are more frequent for very deep neural networks.
Since the deepest layers of a network are the most non-linear ones, we introduce a new Langevin algorithm called Layer Langevin, which consists in adding Langevin noise only to the weights associated to the deepest layers.
We then prove the benefits of Langevin and Layer Langevin algorithms for the training of popular deep residual architectures for image classification.

In this repository we give the implementation of Langevin and Layer Langevin optimizers as instances of the TensorFlow <tt>tf.keras.optimizers.Optimizer</tt> base class and we compare Langevin and non-Langevin optimizers for the training of various image classification problems.

The machine learning library that is used is TensorFlow.




## Requirements

```setup
pip install tensorflow
pip install pandas
```

## Training example

```
python simulations.py
```



## Citation
Please cite our paper if it helps your research:

	@ARTICLE{2022arXiv221214718B,
		author = {{Bras}, Pierre},
			title = "{Langevin algorithms for very deep Neural Networks with application to image classification}",
		journal = {arXiv e-prints},
		keywords = {Computer Science - Machine Learning, Statistics - Machine Learning},
			year = 2022,
			month = dec,
			eid = {arXiv:2212.14718},
			pages = {arXiv:2212.14718},
	archivePrefix = {arXiv},
		eprint = {2212.14718},
	primaryClass = {cs.LG},
		adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv221214718B},
		adsnote = {Provided by the SAO/NASA Astrophysics Data System}
	}
