Neuro-Glia-Vasculature (NGV)
============================

.. image:: img/ngv.drawio.svg
   :alt: NGV Diagram
   :align: center


The explaination here is loosely based on the `tiny_CI <https://github.com/BlueBrain/MultiscaleRun/tree/main/multiscale_run/templates/tiny_CI>`_ simulation example. The rest of the data for this simulation can be downloaded from here:

.. code-block:: bash

   wget https://github.com/BlueBrain/MultiscaleRun/releases/download/0.8.2/tiny_CI_neurodamus_release-v0.8.2.tar.gz


The NGV simulation requires three node populations, each stored in separate files: **neocortex_neurons**, **astrocytes**, and **vasculature**.

- **Neocortex_neurons** represent standard neocortex neurons.
- **Astrocytes** represent glial cells.
- The **vasculature** file is unique: although it is structured and treated as a node file by Neurodamus and the SONATA convention, each node (row in the table) actually refers to a vasculature segment. The vasculature itself is divided into morphology nodes (used internally for structure reconstruction), sections (optional), and segments. Currently, Neurodamus only reads the file and does not utilize this additional information. For more details, refer to the official documentation:  
  `https://sonata-extension.readthedocs.io/en/latest/sonata_tech.html#fields-for-vasculature-population-model-type-vasculature`_.

The diagram also illustrates the various connections among the populations, as described by the edge files (one per connection type):

- **Neuron <-> Neuron connections** are stored in the `neocortex_neurons__chemical_synapse` file. These are simple synaptic connections (type: chemical).
- **Neuron <-> Astrocyte** connections are defined in the `neuroglial` edge file, with the connection type `synapse_astrocyte`. These are tripartite connections that encapsulate an existing neuron-to-neuron synapse connection. A tripartite connection links an existing synapse to an astrocyte segment, enabling the astrocyte to modulate the synapse. This is achieved by placing a `GlutReceive` point process on an astrocyte segment that "spies" on a specific neuron-to-neuron synapse connection.

.. note::
  While in biological systems astrocytes can influence the spied synapse, this behavior is currently not implemented. In the current model, astrocytes cannot influence neuron synapse connections through a tripartite connection.

- **Astrocyte <-> Astrocyte connections** are described in the `glialglial` edge file. These connections are gap junctions. Currently not handled.
- **Astrocyte -> Vasculature** connections are in the `gliovascular` edge file, with the connection type `endfoot`. Here, the `vascCouplingB` mechanism is inserted to control the radii of the vasculature segments. This follows a master-slave relationship, where the astrocyte dictates the vasculature radii.

