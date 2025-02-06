import settings

from neuron import h
from ring import Ring
from dump_cellstate import dump_cellstate

# initialize global variables
settings.init(gap_=False, nring_=1)
pc = settings.pc

h.load_file("cell.hoc")

ncell_A = 3
nbranch = [2, 2]
ncompart = [2, 2]
ncell_B = 2
popB_offset = 1000

# population A , gidstart=0
ring_A = Ring(ncell_A, nbranch, ncompart, gidstart=0, types=range(ncell_A))

# population B, gidstart=1000
ring_B = Ring(ncell_B, nbranch, ncompart, gidstart=popB_offset,
              types=list(range(ncell_B)))

# connect A gid 0 -> B gid 0
precell = pc.gid2cell(0)
postcell = pc.gid2cell(0+popB_offset)
synobj = postcell.synlist[1]
nc = pc.gid_connect(0, synobj)
ring_B.nclist.append(nc)

# dump cell states
# a) with nrn prcellstate C++
h.load_file('stdgui.hoc')
h.stdinit()
pc.prcellstate(1000, "t0")
pc.prcellstate(0, "t0")

# b) with previous prcellstate HOC
h.load_file('prcellstate.hoc')
h.prcellgid(0)
h.prcellgid(1000)

# c) with user python function
for gid in [0, 1000]:
    dump_cellstate(pc, h.CVode(), gid)
