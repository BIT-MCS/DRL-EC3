from .map import Map
from .flag import FLAGS
from PIL import Image
import time
import os

class MapM(Map):

    def __init__(self, log_path, width=80, height=80):
        super(MapM, self).__init__(width, height)
        self.__time = time.time()
        self.full_path = os.path.join(log_path, 'img_map')
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)

    def draw_wall(self, map):
        wall = FLAGS.wall_value
        width = FLAGS.wall_width
        for j in range(0, 80, 1):
            for i in range(80-width, 80, 1):
                self.draw_sqr(i, j, 1, 1, wall, map)
            for i in range(0, width, 1):
                self.draw_sqr(i, j, 1, 1, wall, map)
        for i in range(0, 80, 1):
            for j in range(0, width, 1):
                self.draw_sqr(i, j, 1, 1, wall, map)
            for j in range(80-width, 80, 1):
                self.draw_sqr(i, j, 1, 1, wall, map)

    def get_value(self, x, y, map):
        x, y = self.__trans(x, y)
        super(MapM, self).get_value(x, y, map)

    def __trans(self, x, y):
        return int(4 * x + FLAGS.wall_width*2), int(y * 4 + FLAGS.wall_width*2)

    def draw_obstacle(self, x, y, width, height, map):
        # self.clear_cell(x, y, map)
        x, y = self.__trans(x, y)
        self.draw_sqr(x, y, width * 4, height * 4, FLAGS.wall_value, map)

    def draw_chargestation(self, x, y, map):
        self.clear_cell(x, y, map)
        x, y = self.__trans(x, y)
        self.draw_sqr(x, y + 1, 4, 2, 1, map)
        self.draw_sqr(x + 1, y, 2, 4, 1, map)

    # xy transpose occur
    def draw_point(self, x, y, value, map):
        x, y = self.__trans(x, y)
        self.draw_sqr(x, y, 2, 2, value, map)

    def clear_point(self, x, y, map):
        x, y = self.__trans(x, y)
        self.draw_sqr(x, y, 2, 2, 0, map)

    def clear_uav(self, x, y, map):
        self.clear_cell(x, y, map)

    def draw_UAV(self, x, y, value, map):
        x = -1 if x < -1 else FLAGS.map_x if x > FLAGS.map_x else x
        y = -1 if y < -1 else FLAGS.map_y if y > FLAGS.map_y else y
        self.clear_cell(x, y, map)
        x, y = self.__trans(x, y)
        # self.draw_sqr(x, y + 1, 4, 2, value, map)
        # self.draw_sqr(x + 1, y, 2, 4, value, map)
        # value = self.get_value(x, y)
        self.draw_sqr(x, y, 4, 4, value, map)
        # self.draw_sqr(x, y, 4, 4, value)

    def clear_cell(self, x, y, map):
        x, y = self.__trans(x, y)
        self.draw_sqr(x, y, 4, 4, 0, map)

    def draw_goal(self, x, y, map):
        # x, y = self.__trans(x, y)
        # value = self.get_value(x + 2, y + 2, map)
        # self.draw_sqr(x, y, 4, 4, 1, map)
        # self.draw_sqr(x + 2, y + 2, 2, 2, value, map)
        pass

    def draw_FillStation(self,x,y,value,map):
        x, y = self.__trans(x, y)
        self.draw_sqr(x, y, 2, 2,value,map)


    def save_as_png(self, map, ip=None):
        img = Image.fromarray(map * 255)
        img = img.convert('L')
        # img.show()
        if ip is None:
            name = time.time() - self.__time
        else:
            name = str(ip)
        img.save(os.path.join(self.full_path, str(name)), 'png')

