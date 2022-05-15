class Setting(object):
    def __init__(self, log):
        self.V = {
            'MAP_X': 16,
            'MAP_Y': 16,
            'MAX_VALUE': 1.,
            'MIN_VALUE': 0.,
            'OBSTACLE': [   # todo：OBSTACLE
                [0, 3, 1, 1],
                [2, 9, 2, 1],
                [1, 3, 1, 2],
                [2, 15, 2, 1],
                [2, 0, 1, 1],
                [4, 4, 1, 1],
                [5, 4, 1,3],
                [5, 11, 1, 3],
                [10, 0, 3, 1],
                [10, 1, 1, 1],
                [10, 5, 1, 3],
                [8, 10, 3, 1],
                [9, 15, 1, 1],
                [13, 6, 1, 2],
                [13, 13, 1, 2],
                [12, 15, 4, 1],
                [15, 10, 1, 1]
            ],
            'CHANNEL': 3,

            'NUM_UAV': 2,  # TODO:无人机个数
            'INIT_POSITION': (0, 8, 8),
            'MAX_ENERGY': 50.,  # TODO: 初始能量
            'NUM_ACTION': 2,  # 2
            'SAFE_ENERGY_RATE': 0.2,
            'RANGE': 1.1,   # TODO：采集范围
            'MAXDISTANCE': 1.,
            'COLLECTION_PROPORTION': 0.2,  # c speed   # TODO： 采集速度
            'FILL_PROPORTION': 0.2,  # fill speed  # TODO：充电速度

            'WALL_REWARD': -1.,
            'VISIT': 1. / 1000.,
            'DATA_REWARD': 1.,
            'FILL_REWARD': 1.,
            'ALPHA': 1.,
            'BETA': 0.1,
            'EPSILON': 1e-4,
            'NORMALIZE': .1,
            'FACTOR': 0.1,
        }
        self.LOG = log
        self.time = log.time

    def log(self):
        self.LOG.log(self.V)
