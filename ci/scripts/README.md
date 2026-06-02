# CI scripts

This directory contains scripts to create an environment that is useable by neurodamus.

This stack includes MPI, parallel HDF5, the python dependencies, libsonatareport and neuron.

The goal is to make creating this environment reproducible to a degree and uniform across different installations.

These scripts can be used as building blocks to install the stack or pieces of the stack in different environments and different configurations.

The best way to understand this is to look both at the [Github Actions](https://github.com/openbraininstitute/neurodamus/blob/main/.github/workflows/simulation_test.yml) and also the [Docker build](https://github.com/openbraininstitute/neurodamus/blob/main/Dockerfile) to see how they can be used.

Each of the scripts follows the same pattern comma it is sourced and then one of the functions within it is executed.

It is assumed that "bash" is the shell that is being used.

The order of operations is important because there are some dependencies between them, consult the GitHub actions to see what the latest suggested order is.

Not every step is required to be execute, for instance `hdf5` might be installable by the system package manager so it does not need to be compiled.
