
class Get_geoinfo :
    def __init__(self, place_nm : list) :
        import osmnx as ox
        import geopandas as gpd

        self.area = ox.geocode_to_gdf(place_nm, which_result = 1)
        buildings = ox.geometries_from_place(place_nm, {"building" : True})
        buildings = buildings.loc[buildings.geometry.type == 'Polygon'] # Remove incomplete polygon
        self.buildings = gpd.GeoDataFrame(buildings, geometry = 'geometry', crs = 'EPSG:4326')
        self._get = {'area' : self.area, 'buildings' : self.buildings}

    def __getitem__(self, item) :
        return self._get[item]

if __name__ == "__main__" :
    place_nm = ["Wuhan", "Hanyang", "Wuchang"] # "Jianghan"
    info = Get_geoinfo(place_nm)
    area, buildings = info['area'], info['buildings']