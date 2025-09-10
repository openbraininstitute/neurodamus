# from neurodamus.core.nodeset import _ranges_overlap

from libsonata import Selection

a = Selection([0, 1, 2, 3, 7, 9, 10])
b = Selection([0, 1])

for i in a.ranges:
    print(i)




