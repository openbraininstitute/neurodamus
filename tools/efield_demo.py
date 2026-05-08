from neuron import h

h("{load_file(\"nrngui.hoc\")}")

# fake multiple ElectrodeSource items with variable count of fields
# initial demo to pass NEURON Vector objs to mod file

h("create soma")
h("access soma")
h( "insert extracellular" )

h( "print soma.L" )

obj = h.EFieldIntegrator(0.5)
h.setpointer( h.soma(0.5).extracellular._ref_e, 'e_ext', obj )

h.Vector()

# ElectrodeSource 1
delay = 10
dur = 50
ramp_up = 20
ramp_down = 30

es_xvec = h.Vector([1, 2, 3])
es_yvec = h.Vector([2, 4, 5])
es_zvec = h.Vector([4, 8, 7])
es_freqvec = h.Vector([10, 50, 100])
es_phasevec = h.Vector([10, -10, 0])

obj.add_electrode_source( delay, dur, ramp_up, ramp_down, es_xvec, es_yvec, es_zvec, es_freqvec, es_phasevec )

# ElectrodeSource 2 - support pending
'''
delay = 120
dur = 100
ramp_up = 0
ramp_down = 10

es_xvec = h.Vector([7, 5])
es_yvec = h.Vector([13, 1])
es_zvec = h.Vector([4, 7])
es_freqvec = h.Vector([50, 100])
es_phasevec = h.Vector([10, 0])

obj.add_electrode_source( delay, dur, ramp_up, ramp_down, es_xvec, es_yvec, es_zvec, es_freqvec, es_phasevec )
'''

# run sim
h.tstop = 200

h("stdinit()")

while h.t < h.tstop:
    h.fadvance()
h("print t")

h.quit()
