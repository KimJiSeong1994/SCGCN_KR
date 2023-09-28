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
        self._get = {'area' : self.area, 'landuses' : self.landuse}

    def __getitem__(self, item) :
        return self._get[item]

def Get_streetview(locations, i) :
    import io
    import requests
    from PIL import Image
    from src.config import config
    param = {'fov' : '120', 'heading' : '-45', 'pitch' : '30', 'locations' : f'{locations}'}
    url = f"https://maps.googleapis.com/maps/api/streetview?size=400x300&location={param['locations']}&fov={param['fov']}&heading={param['heading']}&pitch={param['pitch']}&key={config.GOOGLEMAP_API}"
    Image.open(io.BytesIO(requests.request("GET", url, headers ={}, data = {}).content)).save(f"{config.FILE_PATH}/street_view_{i}.png")

if __name__ == "__main__" :
    import os, shutil
    from tqdm import tqdm
    from src.config import config

    place_nm = ["seoul"]
    info = Get_geoinfo(place_nm)
    seoul_info = info["landuses"]
    seoul_info.to_pickle("./Reproduce/data/seoul_landuse.pkl")

    points = [[s.y, s.x] for _, s in enumerate(seoul_info.geometry.centroid.reset_index(drop = True))]
    points = [str(p[0]) + "," + str(p[1]) for p in points]
    _ = [Get_streetview(p, i) for p, i in tqdm(enumerate(points))]

    landuse_type = list(set(seoul_info['landuse'].values))
    landuse_dict = {t: seoul_info[seoul_info['landuse'].isin([t])].index.tolist() for t in landuse_type}
    _ = [os.makedirs(f'{config.FILE_PATH}/{t}') for t in landuse_type]

    file_list = sorted(os.listdir(config.FILE_PATH))[6:]
    file_list.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))
    for t in tqdm(landuse_type) :  # Move category name file
        for i in landuse_dict[t] :
            shutil.move(f"{config.FILE_PATH}/{file_list[i]}", f"{config.FILE_PATH}/{t}/{file_list[i]}")

