from pysph.solver.utils import load
data = load('ed_output/ed_1000.hdf5')
f = data['arrays']['fluid']
plt.axis('equal')
plt.scatter(f.x, f.y, c=f.p, marker='.')