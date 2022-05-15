import numpy as np

class Map(object):
    def __init__(self, width, height):
        # self.__map = np.zeros((width, height))
        self.__width = width
        self.__height = height

    # @property
    # def map(self):
    #     return self.__map
    @property
    def width(self):
        return self.__width
    @property
    def height(self):
        return self.__height

    def get_value(self, x, y, map):
        return map[x][y]

    def draw_sqr(self, x, y, width, height, value, map):
        assert 0 <= x < self.__width and 0 <= y < self.__height, 'the position ({0}, {1}) is not correct.'.format(x, y)
        for i in range(x, x + width, 1):
            for j in range(y, y + height, 1):
                map[i][j] = value


