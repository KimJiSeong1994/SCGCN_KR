class Get_geoinfo :
    def __init__(self, place_nm : list) :
        import itertools
        import osmnx as ox
        import pandas as pd
        import geopandas as gpd

        self.area = ox.geocode_to_gdf(place_nm, which_result = 1)
        use_index = ['residential', 'commercial', 'industrial', 'retail']
        use_sub_index =  ['greenfield', 'forest', 'grass', 'meadow', 'recreation_ground', 'cemetery']
        use_leisure = ['park']

        landuse = ox.geometries_from_place(place_nm, {"landuse" : True})
        landuse = pd.concat([landuse[landuse["landuse"].isin(list(itertools.chain(*[use_index, use_sub_index])))], landuse[landuse["leisure"].isin(use_leisure)]])
        landuse.loc[(landuse["landuse"].isin(use_sub_index)) | (landuse["leisure"].isin(use_leisure)), "landuse"] = "OpenSpace"
        landuse = gpd.GeoDataFrame(landuse, geometry = "geometry", crs = 'EPSG:4326')

        self.landuse = landuse[landuse.geometry.type == 'Polygon'] # Remove incomplete polygon
        self._get = {'area' : self.area, 'landuses' : self.landuses}

    def __getitem__(self, item) :
        return self._get[item]
