Simulation Inputs  
=================  

This page provides an overview of the nomenclature used for inputs in a Neurodamus circuit simulation. It assumes the reader is already familiar with the NEURON software and its terminology. Otherwise, here are the `NEURON docs <https://nrn.readthedocs.io/en/latest/>`_. For much more detailed information about the input files and the SONATA format, the reader can refer to the `SONATA docs <https://sonata-extension.readthedocs.io/en/latest/sonata_tech.html>`_.

``simulation_config.json``  
--------------------------  

This is the main SONATA file that controls the simulation settings. Key fields include:  

- What **reports** the simulator should generate.  
- The duration of the simulation.  

A **report** is one of the possible outputs of the simulation. It typically consists of a time trace of a specific value (e.g., voltage) recorded per neuron. Reports can also be generated per neuronal section.  

``circuit_config.json``  
-----------------------  

Defines how neurons are structured and connected. It can be seen as a graph of graphs, representing interactions between different neuron populations. This file references:  

- ``nodes.h5``: Contains neuron properties. Note that these *nodes* are unrelated to :class:`neurodamus.node.Node`, which is a low-level simulation orchestrator.  
- ``edges.h5``: Defines connections between neurons.  

A **population** is a group of neurons that interact within themselves and with other populations. NEURON does not have a concept of populations—this is specific to Neurodamus.  

Neurons in ``nodes.h5`` are primarily defined by their **Electrical Type (EType)** and **Morphology Type (MType)**.  

- **ETypes**: Ion channel distributions that have been reverse-engineered from specific firing patterns. Assigning an ion channel distribution (distributed mechanisms) to a population often involves genetic algorithms and trial-and-error. They are provided as hoc files, and the path to the folder containing all EType hoc files for a circuit is specified in ``circuit_config.json``. Individual neurons reference their assigned EType in ``nodes.h5``.  
- **MTypes**: Categories of neuron morphologies (graph structures). Some neurons have extensive branching, while others feature a single prominent axon. NEURON itself does not recognize MTypes, as it requires instantiated neurons in memory rather than generic morphology categories. However, Neurodamus uses MTypes to facilitate circuit development.  
- **Morphologies**: Specific instances of MTypes. During circuit building, multiple neuron structures (graphs of sections) are generated from MTypes and stored in `.asc` (NeuronLucida) or `.swc` files. These morphology files are referenced in ``nodes.h5``. ``circuit_config.json`` also specifies the directory containing these morphology files. Due to the complexity of generating many neurons, some morphologies may be reused.  

``node_sets.json``  
------------------  

This file specifies which neurons (by GID) are included in a particular simulation. It overrides the default selection provided in ``circuit_config.json``, which typically includes all neurons unless ``node_sets.json`` is supplied.  
