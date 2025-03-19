Neuro-Glia-Vasculature (NGV)
============================

.. image:: img/ngv.drawio.svg
   :alt: NGV Diagram
   :align: center

The NGV simulation requires three node populations, each stored in separate files: **neocortex_neurons**, **astrocytes**, and **vasculature**.

- **Neocortex_neurons** represent standard neurons.
- **Astrocytes** represent glial cells.
- The **vasculature** file is unique: although it is structured and treated as a node file by Neurodamus, each node (row in the table) actually refers to a vasculature segment. The vasculature itself is divided into morphology nodes (used internally for structure reconstruction), sections (optional), and segments. For more details, refer to the official documentation:  
  `https://sonata-extension.readthedocs.io/en/latest/sonata_tech.html#fields-for-vasculature-population-model-type-vasculature`.

The diagram also illustrates the various connections among the populations, as described by the edge files (one per connection type):

- **Neuron <-> Neuron connections** are stored in the `neocortex_neurons__chemical_synapse` file. These are simple synaptic connections (type: chemical).
- **Neuron <-> Astrocyte** connections are in the `neuroglial` edge file, with the connection type being `synapse_astrocyte`. This is a tripartite connection, added on top of an existing synapse (defined by the neuron <-> neuron edge file). It connects the existing synapse to an astrocyte segment, which can then modulate the synapse, thus giving the connection its tripartite nature.
- **Astrocyte <-> Astrocyte connections** are described in the `glialglial` edge file. These connections are gap junctions.
- **Astrocyte -> Vasculature** connections are in the `gliovascular` edge file, with the connection type `endfoot`. Here, the `vascCouplingB` mechanism is inserted to control the radii of the vasculature segments. This follows a master-slave relationship, where the astrocyte dictates the vasculature radii.
