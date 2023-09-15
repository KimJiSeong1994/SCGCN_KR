class Utils :
    @staticmethod
    def Delaunay(points) :
        import numpy as np
        from tqdm import tqdm
        from scipy.spatial import Delaunay
        tri = Delaunay(points) # [[s.x, s.y] for _, s in enumerate(landuse.geometry.centroid.reset_index(drop = True))]
        PPE_ajd = np.zeros((points.shape[0], points.shape[0]))
        for i in tqdm(tri.simplices.tolist()) :
            PPE_ajd[i[0], i[1]] = 1
            PPE_ajd[i[0], i[2]] = 1
            PPE_ajd[i[1], i[2]] = 1

        return PPE_ajd
