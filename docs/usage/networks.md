# Overview

In PiNN, a network refers to a specific architecture of atomic neural network.
PiNN networks are essentially Keras models, this means you can use networks for
some simple prediction tasks. For more complex tasks like predicting the
potential energy surface or the dipole moment requires defining a relation
between the network prediction (atomic energy, atomic charge) to the desired
properties (forces, dipole moments), use `pinn.model`.

## Using a network

Atomic neural networks (ANNs) in PiNN makes predictions on atoms. By default,
networks in PiNN makes one prediction per atom. A PiNN network can be used just
like a Keras model.

```Python
from pinn.networks import PiNet
pinet = PiNet()
prediction = pinet(tensors)
```


## Controlling output

Optionally, a network can output multi-dimensional or per-structure predictions.
The output is controlled by two parameters of PiNN networks: `out_units` and
`out_pool`.

PiNN networks output a tensor with the shape `[n_atoms]` if `out_units=1` and
`[n_atoms, out_units]` otherwise. Four `out_pool` options are available: "sum",
"max", "min" and "avg". Each outputs structure-wise predictions by reducing the
atomic contributions with the corresponding method.

## Preprocessing

Atomic neural networks often involves calculating the neighbor list of atomic
with certain cutoff, and computing atomic fingerprints. Different networks has
different preprocessing protocols and those operations are abstracted as the
preprocess method of atomic neural networks. The methods can be called like
network.preprocess, and can be used as a dataset operation.

```Python
from pinn.networks import PiNet
from pinn.io import write_tfrecord
write_tfrecord('ds_preprocessed.yml', ds.map(pinet.preprocess))
```


