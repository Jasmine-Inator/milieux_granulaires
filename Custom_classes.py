class index_list:
    def _init_(self, index_list):
        self.indexes=index_list

    def ascoords(self, coords_list):
        return index_2_coords(index_list, coords_list)

class cluster(index_list):
    def _init_(self, index_list, coords_system):
        self.indexes=index_list
        self.coords=self.indexes.ascoords(coords_system)
        self.points=[sh.geometry.Point(point) for point in self.coords]
        self.polygon=sh.geometry.Polygon([(point.x, point.y) for point in self.points])

class coords_system:
    def _init_(self, coords_list)