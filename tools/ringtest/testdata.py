from neuron import h
from ring import Ring
import settings

# initialize global variables
settings.init(gap_=False, nring_=1)


# population A , gidstart=0

h.load_file("cell.hoc")
ncell_A = 3

nbranch = [2, 2]
ncompart = [2, 2]

ring_A = Ring(ncell_A, nbranch, ncompart, gidstart=0, types=range(ncell_A))

pc = settings.pc

for gid in range(ncell_A):
    cell = pc.gid2cell(gid)
    for sec in cell.all:
        for seg in sec:
            print(f"gid={gid} {seg}")

for nc in ring_A.nclist:
    print(f"{nc} {nc.precell()} -> {nc.postcell()}")

# population B, gidstart=1000
pop_offset = 1000
ncell_B = 2
ring_B = Ring(ncell_B, nbranch, ncompart, gidstart=pop_offset,
              types=list(range(ncell_B)))
for gid in range(pop_offset, pop_offset+ncell_B):
    cell = pc.gid2cell(gid)
    for sec in cell.all:
        for seg in sec:
            print(f"gid={gid} {seg}")

# connect A gid 0 -> B gid 0
precell = pc.gid2cell(0)
postcell = pc.gid2cell(0+pop_offset)
synobj = postcell.synlist[1]
nc = pc.gid_connect(0, synobj)
ring_B.nclist.append(nc)

for nc in ring_B.nclist:
    print(f"{nc} {nc.precell()} -> {nc.postcell()}")

# dump cell states
h.load_file('stdgui.hoc')
h.stdinit()
pc.prcellstate(1000, "t0")
pc.prcellstate(0, "t0")
