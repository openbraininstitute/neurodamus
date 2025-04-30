Neuro-Glia-Vasculature (NGV)
============================

.. image:: img/ngv.drawio.svg
   :alt: NGV Diagram
   :align: center

The NGV simulation requires three node populations, each stored in separate files: **neocortex_neurons**, **astrocytes**, and **vasculature**.

- **Neocortex_neurons** represent standard neocortex neurons.
- **Astrocytes** represent glial cells.
- The **vasculature** file is unique: although it is structured and treated as a node file by Neurodamus and the SONATA convention, each node (row in the table) actually refers to a vasculature segment. The vasculature itself is divided into morphology nodes (used internally for structure reconstruction), sections (optional), and segments. Currently, Neurodamus only reads the file and does not utilize this additional information. For more details, refer to the official documentation:  
  `https://sonata-extension.readthedocs.io/en/latest/sonata_tech.html#fields-for-vasculature-population-model-type-vasculature`.

The diagram also illustrates the various connections among the populations, as described by the edge files (one per connection type):

- **Neuron <-> Neuron connections** are stored in the `neocortex_neurons__chemical_synapse` file. These are simple synaptic connections (type: chemical).
- **Neuron <-> Astrocyte** connections are defined in the `neuroglial` edge file, with the connection type `synapse_astrocyte`. These are tripartite connections that encapsulate an existing neuron-to-neuron synapse connection. A tripartite connection links an existing synapse to an astrocyte segment, enabling the astrocyte to modulate the synapse. This is achieved by placing a `GlutReceive` point process on an astrocyte segment that "spies" on a specific neuron-to-neuron synapse connection.

.. note::
  While in biological systems astrocytes can influence the spied synapse, this behavior is currently not implemented. In the current model, astrocytes cannot influence neuron synapse connections through a tripartite connection.

- **Astrocyte <-> Astrocyte connections** are described in the `glialglial` edge file. These connections are gap junctions. Currently not handled.
- **Astrocyte -> Vasculature** connections are in the `gliovascular` edge file, with the connection type `endfoot`. Here, the `vascCouplingB` mechanism is inserted to control the radii of the vasculature segments. This follows a master-slave relationship, where the astrocyte dictates the vasculature radii.

A Circuit Config
-----------------

It may be helpful to provide a sample of a typical `circuit_config.json` file. This is from `tiny_CI`, a synthetic model built to test multiscale run simulations. All the accessory data to run the simulation can be downloaded using the following command:

.. code-block:: bash

   wget https://github.com/BlueBrain/MultiscaleRun/releases/download/0.8.2/tiny_CI_neurodamus_release-v0.8.2.tar.gz

.. code-block:: json

  {
    "version": 2,
    "manifest": {
      "$BASE_DIR": "tiny_CI_neurodamus/build"
    },
    "node_sets_file": "$BASE_DIR/sonata/node_sets.json",
    "networks": {
      "nodes": [
        {
          "nodes_file": "$BASE_DIR/sonata/networks/nodes/neocortex_neurons/nodes.h5",
          "populations": {
            "neocortex_neurons": {
              "type": "biophysical",
              "biophysical_neuron_models_dir": "$BASE_DIR/../emodels/hoc",
              "spatial_segment_index_dir": "$BASE_DIR/sonata/networks/nodes/neocortex_neurons/spatial_segment_index",
              "provenance": {
                "bioname_dir": "$BASE_DIR/../bioname"
              },
              "alternate_morphologies": {
                "h5v1": "$BASE_DIR/morphologies/neurons",
                "neurolucida-asc": "$BASE_DIR/morphologies/neurons"
              }
            }
          }
        },
        {
          "nodes_file": "$BASE_DIR/sonata/networks/nodes/astrocytes/nodes.h5",
          "populations": {
            "astrocytes": {
              "type": "astrocyte",
              "alternate_morphologies": {
                "h5v1": "$BASE_DIR/morphologies/astrocytes/h5"
              },
              "microdomains_file": "$BASE_DIR/sonata/networks/nodes/astrocytes/microdomains.h5",
              "provenance": {
                "bioname_dir": "$BASE_DIR/../bioname"
              }
            }
          }
        },
        {
          "nodes_file": "$BASE_DIR/sonata/networks/nodes/vasculature/nodes.h5",
          "populations": {
            "vasculature": {
              "type": "vasculature",
              "vasculature_file": "$BASE_DIR/../atlas/vasculature.h5",
              "vasculature_mesh": "$BASE_DIR/../atlas/vasculature.obj",
              "provenance": {
                "bioname_dir": "$BASE_DIR/../bioname"
              }
            }
          }
        }
      ],
      "edges": [
        {
          "edges_file": "$BASE_DIR/sonata/networks/edges/functional/neocortex_neurons__chemical_synapse/edges.h5",
          "populations": {
            "neocortex_neurons__chemical_synapse": {
              "type": "chemical",
              "provenance": {
                "bioname_dir": "$BASE_DIR/../bioname"
              },
              "spatial_synapse_index_dir": "$BASE_DIR/sonata/networks/edges/functional/neocortex_neurons__chemical_synapse/spatial_synapse_index"
            }
          }
        },
        {
          "edges_file": "$BASE_DIR/sonata/networks/edges/neuroglial/edges.h5",
          "populations": {
            "neuroglial": {
              "type": "synapse_astrocyte",
              "provenance": {
                "bioname_dir": "$BASE_DIR/../bioname"
              }
            }
          }
        },
        {
          "edges_file": "$BASE_DIR/sonata/networks/edges/glialglial/edges.h5",
          "populations": {
            "glialglial": {
              "type": "glialglial",
              "provenance": {
                "bioname_dir": "$BASE_DIR/../bioname"
              }
            }
          }
        },
        {
          "edges_file": "$BASE_DIR/sonata/networks/edges/gliovascular/edges.h5",
          "populations": {
            "gliovascular": {
              "type": "endfoot",
              "endfeet_meshes_file": "$BASE_DIR/sonata/networks/edges/gliovascular/endfeet_meshes.h5",
              "provenance": {
                "bioname_dir": "$BASE_DIR/../bioname"
              }
            }
          }
        }
      ]
    }
  }

The rest of the simulation can be found `here <https://github.com/BlueBrain/MultiscaleRun/tree/main/multiscale_run/templates/tiny_CI>`_.