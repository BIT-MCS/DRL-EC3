from experiments.crazy_env.tsp_env_setting import Setting
from experiments.image.mapM import MapM
import os
import copy
from os.path import join as pjoin
import numpy as np
import time
import cv2
import math
from gym import spaces


def mypjoin(path1, path2, paths=None):
    full_path = pjoin(path1, path2)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    if paths is not None:
        full_path = pjoin(full_path, paths)
        if not os.path.exists(full_path):
            os.mkdir(full_path)
    return full_path


def myint(a):
    # return int(np.ceil(a))
    return int(np.floor(a))


class Env(object):
    def __init__(self, log):
        # self.tr = tracker.SummaryTracker()
        self.sg = Setting(log)
        self.sg.log()

        # 6-19 00:42
        self.maxaction = 0
        self.minaction = 0
        #

        self.log_dir = log.full_path
        # self.log_dir = mypjoin('.', self.sg.time)
        # basis
        self.mapx = self.sg.V['MAP_X']  # 16
        self.mapy = self.sg.V['MAP_Y']  # 16
        self.map = MapM(self.log_dir)  # [80,80]
        self.channel = self.sg.V['CHANNEL']  # 3
        self.image_data = None
        self.image_position = None
        self.safe_energy_rate = self.sg.V['SAFE_ENERGY_RATE']  # 0.1

        # num of uavs
        self.n = self.sg.V['NUM_UAV']

        # [[80.80,3]]
        # Box用于实现连续数据构成的空间，其中包含两组参数：空间内数据范围（上限和下限），以及空间维度的大小
        self.observation_space = [spaces.Box(low=-1, high=1, shape=(self.map.width, self.map.height, self.channel)) for
                                  i in range(self.n)]

        # [[2]]
        # TODO:去掉了action-state(<0,>0),只留下 delta x, delta y
        self.action_space = [spaces.Box(low=-1, high=1, shape=(self.sg.V['NUM_ACTION'],)) for i in range(self.n)]

        self.maxenergy = self.sg.V['MAX_ENERGY']  # 100
        self.crange = self.sg.V['RANGE']  # 1.1
        self.maxdistance = self.sg.V['MAXDISTANCE']  # 1.0
        self.cspeed = np.float16(self.sg.V['COLLECTION_PROPORTION'])  # 0.2
        self.fspeed = np.float16(self.sg.V['FILL_PROPORTION'])  # 0.1
        self.alpha = self.sg.V['ALPHA']  # 1.0
        self.beta = self.sg.V['BETA']  # 0.1
        self.track = 1. / 1000.

        # ---- 6-8 14:48 add factor
        self.factor = self.sg.V['FACTOR']
        self.epsilon = self.sg.V['EPSILON']
        self.normalize = self.sg.V['NORMALIZE']

        # mapob [16,16]
        self.mapob = np.zeros((self.mapx, self.mapy)).astype(np.int8)

        """
        Initial Obstacles
        """
        # obstacles
        self.OB = 1
        obs = self.sg.V['OBSTACLE']

        # draw obstacles in mapob[16,16], the obstacle is 1, others is 0
        for i in obs:
            for x in range(i[0], i[0] + i[2], 1):
                for y in range(i[1], i[1] + i[3], 1):
                    self.mapob[x][y] = self.OB
        # reward
        self.pwall = self.sg.V['WALL_REWARD']  # -1

        """
        Initial POI(data)
        """
        # POI [256,3]  3->[x,y,value]
        test = [[1.5454101562e-01, 2.2583007812e-02, 6.5332031250e-01],
                [2.1936035156e-01, 2.1618652344e-01, 8.2568359375e-01],
                [3.3813476562e-01, 4.4738769531e-02, 6.6406250000e-02],
                [6.5478515625e-01, 6.5429687500e-01, 8.7280273438e-02],
                [6.9970703125e-01, 7.5000000000e-01, 4.6923828125e-01],
                [3.2177734375e-01, 4.9145507812e-01, 8.8769531250e-01],
                [6.0595703125e-01, 8.5449218750e-01, 1.0772705078e-01],
                [7.1679687500e-01, 1.1370849609e-01, 5.3759765625e-01],
                [7.3046875000e-01, 9.5800781250e-01, 3.6157226562e-01],
                [9.7656250000e-01, 4.9365234375e-01, 2.5732421875e-01],
                [1.4416503906e-01, 7.8320312500e-01, 7.1679687500e-01],
                [7.1435546875e-01, 2.1618652344e-01, 4.7070312500e-01],
                [1.3830566406e-01, 6.8310546875e-01, 6.7675781250e-01],
                [6.2304687500e-01, 1.4045715332e-02, 4.3017578125e-01],
                [9.2919921875e-01, 9.7460937500e-01, 5.6494140625e-01],
                [9.5996093750e-01, 3.4423828125e-02, 1.2927246094e-01],
                [5.4443359375e-01, 7.9199218750e-01, 3.7622070312e-01],
                [4.6777343750e-01, 5.4394531250e-01, 7.2753906250e-01],
                [4.7558593750e-01, 7.0898437500e-01, 7.6562500000e-01],
                [8.5205078125e-01, 4.8364257812e-01, 3.9965820312e-01],
                [7.1240234375e-01, 1.6027832031e-01, 5.7421875000e-01],
                [4.7460937500e-01, 9.8937988281e-02, 3.8500976562e-01],
                [6.1914062500e-01, 1.2841796875e-01, 1.4758300781e-01],
                [6.7773437500e-01, 5.8593750000e-02, 5.6689453125e-01],
                [5.2099609375e-01, 1.2927246094e-01, 1.6943359375e-01],
                [3.0737304688e-01, 9.3066406250e-01, 9.1845703125e-01],
                [1.7565917969e-01, 9.7802734375e-01, 4.3847656250e-01],
                [4.1040039062e-01, 8.9794921875e-01, 2.6123046875e-01],
                [6.5234375000e-01, 6.9580078125e-01, 6.5429687500e-01],
                [9.8046875000e-01, 4.0161132812e-01, 5.4003906250e-01],
                [6.2597656250e-01, 7.5244140625e-01, 8.1640625000e-01],
                [5.6762695312e-02, 7.7734375000e-01, 2.2973632812e-01],
                [9.0380859375e-01, 6.3720703125e-01, 8.8183593750e-01],
                [5.9326171875e-01, 5.8740234375e-01, 7.3339843750e-01],
                [2.6318359375e-01, 6.7480468750e-01, 3.6206054688e-01],
                [2.6245117188e-01, 5.3613281250e-01, 3.1201171875e-01],
                [5.5468750000e-01, 3.2397460938e-01, 5.8496093750e-01],
                [9.3896484375e-01, 6.6601562500e-01, 2.0996093750e-02],
                [1.3537597656e-01, 2.8100585938e-01, 1.8847656250e-01],
                [9.5507812500e-01, 8.2421875000e-01, 6.2890625000e-01],
                [4.3505859375e-01, 9.8046875000e-01, 7.4169921875e-01],
                [4.8559570312e-01, 4.9853515625e-01, 2.4414062500e-01],
                [6.8457031250e-01, 2.5073242188e-01, 4.5385742188e-01],
                [5.1025390625e-01, 8.9990234375e-01, 6.6601562500e-01],
                [6.6992187500e-01, 6.2011718750e-01, 6.6552734375e-01],
                [5.0292968750e-02, 8.3496093750e-01, 6.7968750000e-01],
                [7.8808593750e-01, 1.5332031250e-01, 9.0429687500e-01],
                [8.2128906250e-01, 7.9833984375e-01, 4.6142578125e-01],
                [3.0059814453e-02, 7.8125000000e-01, 4.9951171875e-01],
                [1.9006347656e-01, 7.3144531250e-01, 4.3994140625e-01],
                [8.3544921875e-01, 4.3237304688e-01, 8.6279296875e-01],
                [7.3437500000e-01, 9.9548339844e-02, 1.8688964844e-01],
                [2.6074218750e-01, 9.1699218750e-01, 5.9814453125e-01],
                [8.1689453125e-01, 1.9482421875e-01, 9.2675781250e-01],
                [8.7500000000e-01, 2.7221679688e-01, 7.4707031250e-01],
                [7.4121093750e-01, 6.7529296875e-01, 9.1601562500e-01],
                [9.3066406250e-01, 6.2207031250e-01, 8.2568359375e-01],
                [5.1220703125e-01, 1.7529296875e-01, 1.3122558594e-01],
                [8.9794921875e-01, 3.0053710938e-01, 8.1591796875e-01],
                [2.6953125000e-01, 6.9824218750e-01, 1.1224365234e-01],
                [7.1386718750e-01, 6.3134765625e-01, 1.3537597656e-01],
                [6.8066406250e-01, 6.5673828125e-01, 5.0195312500e-01],
                [5.4248046875e-01, 1.5234375000e-01, 1.6955566406e-01],
                [5.7568359375e-01, 1.5124511719e-01, 8.9599609375e-01],
                [1.7065429688e-01, 8.4411621094e-02, 2.5708007812e-01],
                [8.6474609375e-01, 2.2229003906e-01, 9.2675781250e-01],
                [9.3701171875e-01, 5.1849365234e-02, 3.6474609375e-01],
                [8.1298828125e-01, 7.8564453125e-01, 6.2402343750e-01],
                [4.1503906250e-01, 5.9423828125e-01, 5.0537109375e-01],
                [3.4179687500e-01, 4.7802734375e-01, 8.8818359375e-01],
                [3.9306640625e-01, 5.1074218750e-01, 3.0981445312e-01],
                [8.0566406250e-01, 1.6113281250e-01, 4.4848632812e-01],
                [8.8134765625e-02, 9.7705078125e-01, 8.5742187500e-01],
                [2.1984863281e-01, 7.5048828125e-01, 5.2978515625e-01],
                [8.5839843750e-01, 8.5058593750e-01, 4.6582031250e-01],
                [6.6259765625e-01, 6.6992187500e-01, 6.4404296875e-01],
                [8.7500000000e-01, 9.2138671875e-01, 3.1982421875e-01],
                [4.5800781250e-01, 5.3076171875e-01, 3.9868164062e-01],
                [5.2148437500e-01, 9.7705078125e-01, 8.2617187500e-01],
                [2.3986816406e-01, 5.0488281250e-01, 6.6650390625e-01]]

        # DATA shape:256*3
        self.DATAs = np.reshape(test, (-1, 3)).astype(np.float16)

        # # #TODO:调点
        # dx = [-0.2, -0.2, -0.2, 0, 0, 0, 0.2, 0.2, 0.2]
        # dy = [-0.2, 0, 0.2, -0.2, 0, 0.2, -0.2, 0, 0.2]
        # # replace the POI in obstacle position with the POI out of obstacle position
        # for index in range(self.DATAs.shape[0]):
        #     need_adjust = True
        #     while need_adjust:
        #         need_adjust = False
        #         for i in range(len(dx)):
        #             if self.mapob[min(myint(self.DATAs[index][0] * self.mapx + dx[i]), self.mapx - 1)][
        #                 min(myint(self.DATAs[index][1] * self.mapy + dy[i]), self.mapy - 1)] == self.OB:
        #                 need_adjust = True
        #                 break
        #         if need_adjust is True:
        #             self.DATAs[index] = np.random.rand(3).astype(np.float16)
        #
        # for i, poi_i in enumerate(self.DATAs):
        #     if i == 0:
        #         print("[[%.10e,%.10e,%.10e]," % (poi_i[0], poi_i[1], poi_i[2]))
        #     elif i == len(self.DATAs) - 1:
        #         print("[%.10e,%.10e,%.10e]]\n" % (poi_i[0], poi_i[1], poi_i[2]))
        #     else:
        #         print("[%.10e,%.10e,%.10e]," % (poi_i[0], poi_i[1], poi_i[2]))

        # POI data value [256]
        self._mapmatrix = copy.copy(self.DATAs[:, 2])

        # POI data Position  [256,2]
        self.datas = self.DATAs[:, 0:2] * self.mapx

        # sum of all POI data values
        self.totaldata = np.sum(self.DATAs[:, 2])
        log.log(self.DATAs)

        """
        Initial Fill Station
        """
        # TODO:加入加油站的有限油量
        station = [
            [0.1875, 0.8125, 50],
            [0.625, 0.8125, 50],
            [0.5, 0.5, 50],
            [0.375, 0.125, 50],
            [0.875, 0.25, 50]
        ]

        self.FILL = np.reshape(station, (-1, 3)).astype(np.float16)

        # Fill Station Position  [5,2]
        self.fills = self.FILL[:, 0:2] * self.mapx

        # Fill Station remain energy [5]
        self.fills_energy_remain = copy.copy(self.FILL[:, 2])

        # sum of all FIll Station remain energy
        self.total_fills_energy_remain = np.sum(self.FILL[:, 2])

        log.log(self.FILL)

        """
        Initial image information
        """
        # [80,80]
        self._image_data = np.zeros((self.map.width, self.map.height)).astype(np.float16)

        # [n,80,80]
        self._image_position = np.zeros((self.sg.V['NUM_UAV'], self.map.width, self.map.height)).astype(np.float16)

        # [80,80]
        self._image_access = np.zeros((self.map.width, self.map.height)).astype(np.float16)

        # empty wall
        # draw walls in the border of the map (self._image_data)
        # the value of the wall is -1
        # the width of the wall is 4, which can be customized in image/flag.py
        # after adding four wall borders, the shape of the map is still [80,80]
        self.map.draw_wall(self._image_data)

        # PoI
        # draw PoIs in the map (self._image_data)
        # the position of PoI is [x*4+8,y*4+8] of the [80,80] map,
        # where x,y->[0~15]
        # the PoI's size is [2,2] in [80,80] map
        # the value of PoI in the map is the actual value of PoI (self._mapmatrix[i])
        # PoI value->(0~1)
        for i, position in enumerate(self.datas):
            self.map.draw_point(position[0], position[1], self._mapmatrix[i], self._image_data)
        for obstacle in self.sg.V['OBSTACLE']:
            self.map.draw_obstacle(obstacle[0], obstacle[1], obstacle[2], obstacle[3], self._image_data)

        for i_n in range(self.n):
            # layer 2
            self.map.draw_UAV(self.sg.V['INIT_POSITION'][1], self.sg.V['INIT_POSITION'][2], 1.,
                              self._image_position[i_n])
            for i, position in enumerate(self.fills):
                self.map.draw_FillStation(position[0], position[1], self.fills_energy_remain[i],
                                          self._image_position[i_n])

        # 无人机随机颜色
        self.uav_render_color = []
        for i in range(self.n):
            self.uav_render_color.append(np.random.randint(low=0, high=255, size=3, dtype=np.uint8))

        self.pow_list = []

    def reset(self):
        # initialize data map
        # tr = tracker.SummaryTracker()
        self.mapmatrix = copy.copy(self._mapmatrix)
        self.fills_energy_remain = copy.copy(self.FILL[:, 2])

        # record data access times(per 0.001 default)
        self.maptrack = np.zeros(self.mapmatrix.shape)
        # ----
        # initialize state(get POI/filling) and positions of uavs
        self.uav = [list(self.sg.V['INIT_POSITION']) for i in range(self.n)]
        self.eff = [0.] * self.n
        self.count = 0
        self.zero = 0

        self.trace = [[] for i in range(self.n)]
        self.energytrace = [[] for i in range(self.n)]
        # initialize remaining energy
        self.energy = np.ones(self.n).astype(np.float64) * self.maxenergy
        # initialize indicators
        self.collection = np.zeros(self.n).astype(np.float16)
        # energy use
        self.use_energy = np.zeros(self.n).astype(np.float16)
        # energy fill
        self.fill_energy = np.zeros(self.n).astype(np.float16)
        # energy max
        self.max_energy_array = np.array([self.maxenergy] * self.n).astype(np.float16)

        # walls
        self.walls = np.zeros(self.n).astype(np.int16)

        # time
        self.time_ = 0

        # initialize images
        self.state = self.__init_image()
        return self.__get_state()

    def __init_image(self):
        self.image_data = copy.copy(self._image_data)
        self.image_access = copy.copy(self._image_access)
        self.image_position = copy.copy(self._image_position)
        self.image_track = np.zeros(self.image_position.shape)
        # ----
        state = []
        for i in range(self.n):
            image = np.zeros((self.map.width, self.map.height, self.channel)).astype(np.float16)
            for width in range(image.shape[0]):
                for height in range(image.shape[1]):
                    # god view
                    image[width][height][0] = self.image_data[width][height]
                    image[width][height][1] = self.image_position[i][width][height]
                    image[width][height][2] = self.image_access[width][height]
            state.append(image)
        return state

    def __draw_image(self, clear_uav, update_point, update_station, update_track):
        # update 3 channels
        for n in range(self.n):
            for i, value in update_point:
                self.map.draw_point(self.datas[i][0], self.datas[i][1], value, self.state[n][:, :, 0])
            for i, value in update_station:
                self.map.draw_point(self.fills[i][0], self.fills[i][1], value, self.state[n][:, :, 1])
            self.map.clear_uav(clear_uav[n][1], clear_uav[n][2], self.state[n][:, :, 1])
            self.map.draw_UAV(self.uav[n][1], self.uav[n][2], self.energy[n] / self.maxenergy, self.state[n][:, :, 1])

            # ---- draw track
            for i, value in update_track:
                self.map.draw_point(self.datas[i][0], self.datas[i][1], value, self.state[n][:, :, 2])

    def __get_state(self):
        return copy.deepcopy(self.state)

    # TODO: penalty加移动penalty,有待商榷
    def __get_reward(self, value, energy, distance, penalty, fairness, fairness_):
        factor0 = value / (self.factor * distance + self.alpha * value + self.epsilon)
        factor1 = energy / self.maxenergy / (self.factor * distance + self.epsilon)
        reward = factor0 + factor1
        if value == 0 and energy == 0:  # 浪费生命的一步
            return penalty - self.normalize * distance
        else:
            return reward * fairness_ + penalty

    def __get_fairness(self, values):
        square_of_sum = np.square(np.sum(values))
        sum_of_square = np.sum(np.square(values))
        if sum_of_square == 0:
            return 0.
        jain_fairness_index = square_of_sum / sum_of_square / float(len(values))
        return jain_fairness_index

    def __get_eff1(self, value, distance):
        return value / (distance + self.alpha * value + self.epsilon)

    def __cusume_energy1(self, uav, value, distance):
        # distance-0.1, alpha-1.0
        if (self.factor * distance + self.alpha * value < self.energy[uav]):
            self.energy[uav] -= (self.factor * distance + self.alpha * value)
            self.use_energy[uav] += (self.factor * distance + self.alpha * value)
        else:
            self.use_energy[uav] += self.energy[uav]
            distance = self.energy[uav] / self.factor
            self.energy[uav] = 0

    def __fill_energy1(self, uav):
        # fspeed-0.1
        if self.energy[uav] + self.fspeed * self.maxenergy <= self.maxenergy:
            self.fill_energy[uav] += self.fspeed * self.maxenergy
            self.energy[uav] += self.fspeed * self.maxenergy
        else:
            self.fill_energy[uav] += self.maxenergy - self.energy[uav]
            self.energy[uav] = self.maxenergy

    def step(self, actions, indicator=None):
        # actions = actions.reshape((2, 3))
        self.count += 1
        action = copy.deepcopy(actions)
        # 6-20 00:43
        if np.max(action) > self.maxaction:
            self.maxaction = np.max(action)
            # print(self.maxaction)
        if np.min(action) < self.minaction:
            self.minaction = np.min(action)
            # print(self.minaction)

        action = np.clip(action, -1e3, 1e3)

        normalize = self.normalize

        # TODO:梯度爆炸问题不可小觑,
        # 遇到nan直接卡掉
        for i in range(self.n):
            for ii in action[i]:
                if np.isnan(ii):
                    print('Nan. What can I do? do!')
                    while True:
                        pass

        reward = [0] * self.n
        self.tmp_value = [0] * self.n
        self.tmp_energy = [0] * self.n
        self.tmp_distance = [0] * self.n
        self.tmp_penalty = [0] * self.n
        self.dn = [False] * self.n  # no energy UAV
        update_points = []  # Updated PoI remained data
        update_stations = []  # Updated Station remained energy
        update_tracks = []  # Updated PoI access times
        clear_uav = copy.copy(self.uav)
        new_positions = []
        c_f = self.__get_fairness(self.maptrack)

        # update positions of UAVs
        for i in range(self.n):
            self.trace[i].append(self.uav[i])
            self.energytrace[i].append(self.energy[i] / self.maxenergy)

            # distance is from action(x,y), which is a kind of offset,[minaction,maxaction]
            distance = np.sqrt(np.power(action[i][0], 2) + np.power(action[i][1], 2))
            data = 0.0
            value = 0.0
            energy = 0.0
            penalty = 0.0

            # think about distance and energy
            # 1.normal and enough energy
            # 2.so small
            # 3.so large(>maxdistance) enough energy
            # 4.so large(>energy)
            if distance <= self.maxdistance and self.energy[i] >= self.factor * distance:
                new_x = self.uav[i][1] + action[i][0]
                new_y = self.uav[i][2] + action[i][1]
            else:
                maxdistance = self.maxdistance if self.maxdistance <= self.energy[i] else \
                    self.energy[i]
                # distance>=0.001
                if distance <= self.epsilon:
                    distance = self.epsilon
                    print("very small.")
                new_x = self.uav[i][1] + maxdistance * action[i][0] / distance
                new_y = self.uav[i][2] + maxdistance * action[i][1] / distance
                distance = maxdistance

            self.__cusume_energy1(i, 0, distance)

            # penalty!!
            # update position
            # if normal, save update
            # if reach OB or WALL, give negative reward, save original positon
            dx = new_x - self.uav[i][1]
            dy = new_y - self.uav[i][2]
            # TODO：简单的防夸张跳墙
            if 0 <= new_x < self.mapx and 0 <= new_y < self.mapy and self.mapob[myint(new_x)][
                myint(new_y)] != self.OB and \
                    self.mapob[myint(self.uav[i][1] + (dx / 2))][myint(self.uav[i][2] + (dy / 2))] != self.OB and \
                    self.mapob[myint(self.uav[i][1] + (dx / 3))][myint(self.uav[i][2] + (dy / 3))] != self.OB and \
                    self.mapob[myint(self.uav[i][1] + (2 * dx / 3))][
                        myint(self.uav[i][2] + (2 * dy / 3))] != self.OB and \
                    self.mapob[myint(self.uav[i][1] + (dx / 4))][myint(self.uav[i][2] + (dy / 4))] != self.OB and \
                    self.mapob[myint(self.uav[i][1] + (3 * dx / 4))][myint(self.uav[i][2] + (3 * dy / 4))] != self.OB:
                new_positions.append([0, new_x, new_y])
            else:
                new_positions.append([0, self.uav[i][1], self.uav[i][2]])
                penalty += normalize * self.pwall
                self.walls[i] += 1

            # TODO:加完了会有惊喜的哈哈哈！！！
            if self.energy[i] < self.safe_energy_rate * self.maxenergy:
                penalty += -1. * distance * normalize

            # TODO:先看能否加油
            # calculate distances between UAV and FillStation points
            _pos = np.repeat([new_positions[-1][1:]], [self.fills.shape[0]], axis=0)  # just repeat(On)  NB!
            _minus = self.fills - _pos
            _power = np.power(_minus, 2)
            _dis = np.sum(_power, axis=1)
            __exists_FS = 0
            tmp = self.energy[i]
            for index, dis in enumerate(_dis):
                # sensing Fill Station(crange=1.1)
                if np.sqrt(dis) <= self.crange:
                    __exists_FS = 1
                    # uodate poi data
                    if self.fills_energy_remain[index] > 0:
                        # TODO:加油站的信息更新
                        if self.fspeed * self.maxenergy <= self.fills_energy_remain[index]:
                            if self.energy[i] + self.fspeed * self.maxenergy <= self.maxenergy:
                                self.fill_energy[i] += self.fspeed * self.maxenergy
                                self.fills_energy_remain[index] -= self.fspeed * self.maxenergy
                                self.energy[i] += self.fspeed * self.maxenergy
                            else:
                                self.fill_energy[i] += self.maxenergy - self.energy[i]
                                self.fills_energy_remain[index] -= (self.maxenergy - self.energy[i])
                                self.energy[i] = self.maxenergy
                        else:
                            if self.energy[i] + self.fills_energy_remain[index] <= self.maxenergy:
                                self.fill_energy[i] += self.fills_energy_remain[index]
                                self.energy[i] += self.fills_energy_remain[index]
                                self.fills_energy_remain[index] = 0
                            else:
                                self.fill_energy[i] += self.maxenergy - self.energy[i]
                                self.fills_energy_remain[index] -= (self.maxenergy - self.energy[i])
                                self.energy[i] = self.maxenergy
                        update_stations.append([index, self.fills_energy_remain[index]])
                    break

            # 若在加油站范围内则加油,若不在任何一个加油站范围内,则采集数据
            if __exists_FS == 1:
                new_positions[-1][0] = -1  # 状态标识符置为-1
                if indicator is not None:
                    indicator[i] = -1
                # fill energy!!
                energy = self.energy[i] - tmp


            else:
                new_positions[-1][0] = 1  # 状态标识符置为1
                if indicator is not None:
                    indicator[i] = 1
                # calculate distances between UAV and data points
                _pos = np.repeat([new_positions[-1][1:]], [self.datas.shape[0]], axis=0)  # just repeat(On)  NB!
                _minus = self.datas - _pos
                _power = np.power(_minus, 2)
                _dis = np.sum(_power, axis=1)
                for index, dis in enumerate(_dis):
                    # sensing PoI(crange=1.1)
                    if np.sqrt(dis) <= self.crange:
                        self.maptrack[index] += self.track
                        update_tracks.append([index, self.maptrack[index]])  # update poi access times

                        # uodate poi data
                        if self.mapmatrix[index] > 0:
                            # cspeed just like a perceptage of consuming a special POI
                            data += self._mapmatrix[index] * self.cspeed
                            self.mapmatrix[index] -= self._mapmatrix[index] * self.cspeed
                            if self.mapmatrix[index] < 0:
                                self.mapmatrix[index] = 0.
                            update_points.append([index, self.mapmatrix[index]])

                # update info (collected data)
                # use energy to get POI(consume energy of UAVs, per alpha 1.0 default)
                value = data if self.energy[i] >= data * self.alpha else self.energy[i]
                self.__cusume_energy1(i, value, 0.)  # 采集数据

            # calculate fairness
            c_f_ = self.__get_fairness(self.maptrack)

            # reward
            reward[i] += self.__get_reward(value, energy, distance, penalty, c_f, c_f_)

            # TODO:debug
            self.tmp_value[i] = value
            self.tmp_energy[i] = energy
            self.tmp_distance[i] = distance
            self.tmp_penalty[i] = penalty

            # ----
            c_f = c_f_

            # efficiency
            self.eff[i] += self.__get_eff1(value, distance)
            self.collection[i] += value

            # mark no energy UAVs
            if self.energy[i] <= self.epsilon * self.maxenergy:
                self.dn[i] = True

        self.uav = new_positions
        t = time.time()
        self.__draw_image(clear_uav, update_points, update_stations, update_tracks)
        self.time_ += time.time() - t

        # TODO:放大reward  为什么要人为砍梯度?
        self.reward = list(np.clip(np.array(reward) / normalize, -2., 2.))
        # self.reward = list(np.array(reward) / normalize)

        info = None
        state = self.__get_state()
        for r in self.reward:
            if np.isnan(r):
                print('Rerward Nan')
                while True:
                    pass

        # TODO:不提前结束，给予一些的躺尸的经历,最极端的就是所有无人机一起躺尸，但是TDerror可能会有问题吧
        # done = True
        # for d in self.dn:
        #     if d is False:
        #         done = False
        #         break
        #     else:
        #         continue

        done = False
        return state, self.reward, done, info, indicator

    def render(self):
        global power_list
        observ = list(self.__get_state())
        observ = np.array(observ)
        observ = observ.transpose((0, 2, 1, 3))
        observ_0 = observ[np.random.randint(low=0, high=self.n), :, :, 0]
        observ_1 = observ[np.random.randint(low=0, high=self.n), :, :, 2]

        img_0 = np.zeros([80, 80, 3], dtype=np.uint8)
        self.draw_convert(observ_0, img_0, max(self._mapmatrix), color=np.asarray([0, 255, 0]))

        max_visit_val = max(np.max(observ_1), self.sg.V['VISIT'] * 20)
        img_1 = np.zeros([80, 80, 3], dtype=np.uint8)
        self.draw_convert(observ_1, img_1, max_visit_val, color=np.asarray([0, 255, 0]))

        for i in range(self.n):
            power_list = self.draw_convert(observ[i, :, :, 1], img_0, self.maxenergy, color=self.uav_render_color[i],
                                           is_power=True)

        img = np.hstack([img_0, img_1])
        img = cv2.resize(img, (800, 400))

        for p in power_list:
            cv2.circle(img, (p[1] * 5, p[0] * 5), 25, (0, 0, 255))

        img = cv2.flip(img, 0, dst=None)

        cv2.imshow('show', img)
        cv2.waitKey(1)

    def draw_convert(self, observ, img, max_val, color, is_power=False):
        for i in range(80):
            for j in range(80):

                if observ[j, i] < 0 and is_power == False:
                    img[j, i, 0] = np.uint8(255)
                elif observ[j, i] < 0 and is_power == True:
                    img[j, i, 2] = np.uint8(255)
                    self.pow_list.append((j, i))
                elif observ[j, i] > 0 and is_power == True:
                    img[j, i, :] = np.uint8(color * observ[j, i])
                elif observ[j, i] > 0 and is_power == False:
                    img[j, i, :] = np.uint8(color * observ[j, i] / max_val)

        if len(self.pow_list) > 0:
            return self.pow_list

    # TODO:MAYBE NOT USEFUL NOW!!!
    @property
    def leftrewards(self):
        return np.sum(self.mapmatrix) / self.totaldata

    @property
    def efficiency(self):
        return np.sum(self.collection / self.totaldata) * self.collection_fairness / (np.sum(self.normal_use_energy))

    @property
    def normal_use_energy(self):
        tmp = list(np.array(self.use_energy) / (self.max_energy_array))
        # for i in range(len(tmp)):
        #     if tmp[i] > 1.0:
        #         tmp[i] = 1.0

        return tmp

    @property
    def fairness(self):
        square_of_sum = np.square(np.sum(self.mapmatrix[:]))
        sum_of_square = np.sum(np.square(self.mapmatrix[:]))
        fairness = square_of_sum / sum_of_square / float(len(self.mapmatrix))
        return fairness

    @property
    def collection_fairness(self):
        collection = self._mapmatrix - self.mapmatrix
        square_of_sum = np.square(np.sum(collection))
        sum_of_square = np.sum(np.square(collection))
        fairness = square_of_sum / sum_of_square / float(len(collection))
        return fairness

    @property
    def normal_collection_fairness(self):
        collection = self._mapmatrix - self.mapmatrix
        for index, i in enumerate(collection):
            collection[index] = i / self._mapmatrix[index]
        square_of_sum = np.square(np.sum(collection))
        sum_of_square = np.sum(np.square(collection))
        fairness = square_of_sum / sum_of_square / float(len(collection))
        return fairness
