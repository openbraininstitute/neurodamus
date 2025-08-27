from neuron import coreneuron

from neurodamus.core import NeuronWrapper as Nd

Nd.cvode.cache_efficient(1)
coreneuron.enable = True
coreneuron.file_mode = True
coreneuron.sim_config = "sim.conf"

print(dir(coreneuron))

print(coreneuron.sim_config)

# # set build_path only if the user explicitly asked with --save
# # in this way we do not create 1_2.dat and time.dat if not needed
# if SimConfig.save:
#     coreneuron.save_path = self.build_path
# if SimConfig.restore:
#     coreneuron.restore_path = self.restore_path

# # Model is already written to disk by calling pc.nrncore_write()
# coreneuron.skip_write_model_to_disk = True
# coreneuron.model_path = f"{self.datadir}"
# Nd.pc.psolve(Nd.tstop)