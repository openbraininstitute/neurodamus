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
- The **vasculature** file is unique: although it is structured and treated as a node file by Neurodamus and the SONATA convention, each node (row in the table) actually refers to a vasculature segment. The vasculature itself is divided into morphology nodes (used internally for structure reconstruction), sections (optional), and segments. Currently, Neurodamus only reads the file and does not utilize this additional information. For more details, refer to the `official documentation <https://sonata-extension.readthedocs.io/en/latest/sonata_tech.html#fields-for-vasculature-population-model-type-vasculature>`_.

The diagram also illustrates the various connections among the populations, as described by the edge files (one per connection type):

- **Neuron <-> Neuron connections** are stored in the `neocortex_neurons__chemical_synapse` file. These are simple synaptic connections (type: chemical).
- **Neuron <-> Astrocyte** connections are defined in the `neuroglial` edge file, with the connection type `synapse_astrocyte`. These are tripartite connections that encapsulate an existing neuron-to-neuron synapse connection. A tripartite connection links an existing synapse to an astrocyte segment, enabling the astrocyte to modulate the synapse. This is achieved by placing a `GlutReceive` point process on an astrocyte segment that "spies" on a specific neuron-to-neuron synapse connection.

.. note::
  While in biological systems astrocytes can influence the spied synapse, this behavior is currently not implemented. In the current model, astrocytes cannot influence neuron synapse connections through a tripartite connection.

- **Astrocyte <-> Astrocyte connections** are described in the `glialglial` edge file. These connections are gap junctions. Currently not handled.
- **Astrocyte -> Vasculature** connections are in the `gliovascular` edge file, with the connection type `endfoot`. Here, the `vascCouplingB` mechanism is inserted to control the radii of the vasculature segments. This follows a master-slave relationship, where the astrocyte dictates the vasculature radii.

Astrocytes and Their Role in Blood Vessel Modulation
=====================================================

.. image:: img/tripartite.drawio.svg
   :alt: Tripartite Connection
   :align: center

Astrocytes serve as critical intermediaries between neuronal activity and vascular response, playing a regulatory role in controlling blood vessel radii. This function is enabled by their close anatomical and functional association with neurons, particularly within specialized regions known as *microdomains*—localized zones where an astrocyte interfaces with adjacent synapses and vasculature. Microdomains exhibit minimal overlap, effectively delineating the spatial domain of influence of individual astrocytes.

Tripartite Synapse Mechanism
----------------------------

The model begins with a network of neurons connected via stochastic synapses. When a presynaptic neuron fires, it sends a signal through the spike exchange to the appropriate synapse, which probabilistically decides whether to fire or fail within a `NET_RECEIVE` block. Although the stochastic behavior is implemented in the postsynaptic neuron's point process, it conceptually represents the probabilistic release of neurotransmitters by the presynaptic neuron.

If the synapse is part of a tripartite connection—so named because it involves a presynaptic neuron, a postsynaptic neuron, and an astrocyte—and it successfully fires, it sets an internal state variable ``Ustate = 1``. This triggers an additional signal back into the spike exchange system, which is received by one or more ``glutReceive`` point processes distributed across the astrocyte, as well as by a cumulative ``glutReceiveSoma`` process located in the soma section that collects signals sent to any of the astrocyte's ``glutReceive`` point processes. These mechanisms enable the astrocyte to monitor and respond to synaptic activity, forming the basis for its role in modulating blood vessel radii.

Astrocyte Structure and Dynamics
--------------------------------

Each astrocyte is composed of multiple sections:

- **Soma section**:

  - Contains three key components:

    - A `glutReceive` point process.
    - A `glutReceiveSoma` point process, which monitors recent synaptic activity across the entire astrocyte by counting signals received within the last millisecond. If no signals arrive during this period, the counter is reset, enabling real-time responsiveness.
    - A `cadifus` mechanism responsible for calcium diffusion, configured to observe both `glutReceive` and `glutReceiveSoma`.

  Due to the order of section initialization, the `cadifus` pointer is overwritten by the `glutReceiveSoma` reference when the soma is connected last. Consequently, the `glutReceive` in the soma is likely unused in practice.

- **Endfeet sections**:

  - Instantiated during astrocyte creation.
  - Each contains:
  
    - A ``glutReceive`` point process that counts all received glutamate signals.
    - A ``cadifus`` mechanism that diffuses calcium based on this counter and resets the glutamate count.
    - A ``vascCouplingB`` mechanism that modulates the blood vessel radius in response to local calcium concentration.

  - Each endfoot is instantiated as a new neuron section and added to the ``endfeet`` SectionList. They are not connected to each other but are connected to the ``source_node_id``.

    Contrary to the `documentation <https://sonata-extension.readthedocs.io/en/latest/sonata_tech.html#fields-for-endfoot-connection-type-edges>`_, the fields ``vasculature_section_id`` and ``vasculature_segment_id`` are not mandatory. In practice, they are unused by both the SONATA reader and Neurodamus.

    Only the ``source_node_id`` field is required, as the ``section_id`` and ``segment_id`` can be used to infer it. Including these additional fields may introduce inconsistencies.

Signal Processing and Vascular Modulation Chain
-----------------------------------------------

The sequence of interactions leading to blood vessel modulation is as follows:

1. A presynaptic neuron fires and sends a signal to the spike exchange.
2. The targeted synapse processes this signal through its `NET_RECEIVE` block and may fire.
3. If the synapse is part of a tripartite connection and it fires, `Ustate` is set to 1.
4. This event sends another signal to the spike exchange, which is collected by `glutReceive` and `glutReceiveSoma` point processes on the astrocyte.
5. The `glutReceive` processes track the total number of signals received over the simulation.
6. The `cadifus` mechanism diffuses calcium based on the glutamate signal counters and resets the glutamate count.
7. `glutReceiveSoma` in the soma tracks recent activity, resetting if inactive for a millisecond.
8. Endfeet sections use `vascCouplingB` to adjust blood vessel radii based on calcium levels.

Implementation Details
----------------------

All `glutReceive` objects are stored in a `glut_list` in `neurodamus.ngv.Astrocyte` to prevent garbage collection. The list ends with the `GlutReceiveSoma` instance, ensuring index alignment with section placement.

This architecture allows astrocytes to effectively translate synaptic activity into localized vascular responses, thereby linking neural signaling to blood flow regulation.





