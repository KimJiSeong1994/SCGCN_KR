class Utils :
    @staticmethod
    def Delaunay(points) :
        import numpy as np
        from tqdm import tqdm
        from scipy.spatial import Delaunay

        # points = [[s.x, s.y] for _, s in enumerate(landuse.geometry.centroid.reset_index(drop = True))]
        tri = Delaunay(points)
        PPE_ajd = np.zeros((len(points), len(points)))
        for i in tqdm(tri.simplices.tolist()) :
            PPE_ajd[i[0], i[1]] = 1
            PPE_ajd[i[0], i[2]] = 1
            PPE_ajd[i[1], i[2]] = 1

        return PPE_ajd


