class Setting(object):
    def __init__(self, log):
        self.V = {
            'MAP_X': 16,
            'MAP_Y': 16,
            'MAX_VALUE': 1.,
            'MIN_VALUE': 0.,
            'OBSTACLE': [
                # [0, 4, 1, 1],
                # [0, 9, 1, 1],
                # [0, 10, 2, 1],
                # [2, 2, 2, 1],
                # [5, 13, 1, 1],
                # [6, 12, 2, 1],
                # [10, 5, 3, 1],
                # [11, 5, 1, 3],
                # [10, 13, 1, 2],
                # [11, 13, 2, 1],
                # [12, 0, 1, 2],
                # [12, 5, 1, 1],
                # [12, 7, 1, 1],
                # [15, 11, 1, 1]
            ],
            'CHANNEL': 3,

            'NUM_UAV': 5,
            'INIT_POSITION': (0, 8, 8),
            'MAX_ENERGY': 50.,  # must face the time of lack
            'NUM_ACTION': 2,  # 2
            'SAFE_ENERGY_RATE': 0.2,
            'RANGE': 1.1,
            'MAXDISTANCE': 4.,
            'COLLECTION_PROPORTION': 0.2,  # c speed
            'FILL_PROPORTION': 0.2,  # fill speed

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
