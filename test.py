# from neurodamus.core.nodeset import _ranges_overlap

from libsonata import Selection

a = Selection([0, 1, 2, 3])
b = Selection([0, 1])

c = a & b

print(a)
print(b)

d = b & a
print(c) # -> Selection([[0, 2]])
print(d) # -> Selection([[0, 4]])


