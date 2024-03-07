from env0.env_setting0 import Setting
from image.mapM import MapM
import os
import copy
from os.path import join as pjoin
import numpy as np
import time
# from pympler import tracker
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


        # self.log_dir = log.full_path
        print(log.full_path)
        self.log_dir = mypjoin('.', self.sg.time)
        # basis
        self.mapx = self.sg.V['MAP_X']
        self.mapy = self.sg.V['MAP_Y']
        self.map = MapM(self.log_dir)
        self.channel = self.sg.V['CHANNLE']
        self.image_data = None
        self.image_position = None

        # uavs
        self.n = self.sg.V['NUM_UAV']
        self.observation_space = [spaces.Box(low=-1, high=1, shape=(self.map.width, self.map.height, self.channel)) for
                                  i in range(self.n)]
        self.action_space = [spaces.Box(low=-1, high=1, shape=(self.sg.V['NUM_ACTION'],)) for i in range(self.n)]
        self.maxenergy = self.sg.V['MAX_ENERGY']
        self.crange = self.sg.V['RANGE']
        self.maxdistance = self.sg.V['MAXDISTANCE']
        self.cspeed = np.float16(self.sg.V['COLLECTION_PROPORTION'])
        self.alpha = self.sg.V['ALPHA']
        self.track = 1. / 1000.
        # ---- 6-8 14:48 add factor
        self.factor = self.sg.V['FACTOR']
        # ----
        # self.beta = self.sg.V['BETA']
        self.epsilon = self.sg.V['EPSILON']
        self.normalize = self.sg.V['NORMALIZE']
        # obstacles
        self.OB = 1
        self.mapob = np.zeros((self.mapx, self.mapy)).astype(np.int8)
        obs = self.sg.V['OBSTACLE']
        for i in obs:
            for x in range(i[0], i[0] + i[2], 1):
                for y in range(i[1], i[1] + i[3], 1):
                    self.mapob[x][y] = self.OB
        # reward
        self.pwall = self.sg.V['WALL_REWARD']
        self.rdata = self.sg.V['DATA_REWARD']
        self.pstep = self.sg.V['WASTE_STEP']

        test = [[7.33642578e-02, 2.69531250e-01, 2.81494141e-01],
                [8.95019531e-01, 9.73632812e-01, 2.92480469e-01],
                [5.00000000e-01, 2.69775391e-01, 5.78613281e-01],
                [2.56591797e-01, 9.40917969e-01, 1.86279297e-01],
                [5.30273438e-01, 4.57031250e-01, 3.90380859e-01],
                [3.30322266e-01, 3.19580078e-01, 2.03979492e-01],
                [1.23413086e-01, 6.88476562e-02, 7.76367188e-01],
                [1.02767944e-02, 9.99023438e-01, 8.34472656e-01],
                [6.66015625e-01, 6.83593750e-01, 2.68310547e-01],
                [8.75488281e-01, 5.19042969e-01, 9.21386719e-01],
                [6.89941406e-01, 1.40991211e-01, 2.58300781e-01],
                [4.75585938e-01, 9.25781250e-01, 6.91894531e-01],
                [6.16455078e-02, 2.71911621e-02, 3.36914062e-01],
                [6.19506836e-02, 8.25683594e-01, 3.84033203e-01],
                [6.01562500e-01, 1.92749023e-01, 5.58593750e-01],
                [6.74804688e-01, 7.32910156e-01, 3.62060547e-01],
                [5.38574219e-01, 6.76757812e-01, 1.43356323e-02],
                [8.76953125e-01, 7.98339844e-01, 4.41650391e-01],
                [8.70605469e-01, 2.06176758e-01, 4.76074219e-01],
                [5.47363281e-01, 2.02026367e-01, 6.69921875e-01],
                [8.03222656e-01, 2.40234375e-01, 5.81542969e-01],
                [9.87792969e-01, 1.49917603e-02, 9.50336456e-02],
                [4.52148438e-01, 1.66625977e-01, 9.69238281e-01],
                [1.27441406e-01, 7.05337524e-03, 1.44531250e-01],
                [3.00537109e-01, 8.23730469e-01, 8.95996094e-01],
                [3.09326172e-01, 9.25292969e-01, 8.89160156e-01],
                [4.48486328e-01, 4.97802734e-01, 2.10937500e-01],
                [3.92089844e-01, 5.18188477e-02, 3.69140625e-01],
                [7.15942383e-02, 7.82714844e-01, 3.21044922e-01],
                [8.25683594e-01, 6.38671875e-01, 3.76220703e-01],
                [8.36914062e-01, 7.09472656e-01, 2.81250000e-01],
                [9.45312500e-01, 8.15917969e-01, 7.89550781e-01],
                [2.42065430e-01, 5.84472656e-01, 9.02832031e-01],
                [9.34082031e-01, 1.72241211e-01, 3.70849609e-01],
                [3.54736328e-01, 4.90234375e-01, 9.87304688e-01],
                [3.50585938e-01, 1.26586914e-01, 3.50585938e-01],
                [3.67675781e-01, 4.57763672e-01, 2.13378906e-01],
                [3.50097656e-01, 5.02441406e-01, 3.24951172e-01],
                [9.12597656e-01, 8.50219727e-02, 2.12280273e-01],
                [8.03222656e-01, 5.37597656e-01, 1.59423828e-01],
                [9.46289062e-01, 6.02050781e-01, 2.34985352e-01],
                [3.37646484e-01, 9.49096680e-03, 9.97070312e-01],
                [3.85498047e-01, 5.94726562e-01, 1.81427002e-02],
                [9.70214844e-01, 7.64648438e-01, 4.93652344e-01],
                [6.06933594e-01, 1.90673828e-01, 5.24902344e-02],
                [7.98339844e-02, 4.51660156e-01, 6.46484375e-01],
                [6.64550781e-01, 4.71679688e-01, 8.91113281e-01],
                [1.95770264e-02, 8.50585938e-01, 8.79394531e-01],
                [6.83593750e-01, 3.07006836e-02, 9.87304688e-01],
                [9.70703125e-01, 7.96508789e-02, 6.67480469e-01],
                [7.52929688e-01, 1.40380859e-01, 2.69775391e-01],
                [5.69824219e-01, 8.32031250e-01, 8.64257812e-01],
                [6.37817383e-02, 1.97143555e-01, 2.38769531e-01],
                [7.71484375e-01, 2.25708008e-01, 8.58764648e-02],
                [7.44628906e-01, 5.01953125e-01, 4.51904297e-01],
                [9.37988281e-01, 6.22558594e-01, 8.70117188e-01],
                [5.26855469e-01, 6.06933594e-01, 4.72656250e-01],
                [8.54980469e-01, 8.50585938e-01, 8.12500000e-01],
                [6.50878906e-01, 6.57714844e-01, 2.10449219e-01],
                [8.08593750e-01, 7.40722656e-01, 2.78808594e-01],
                [9.27734375e-02, 5.13183594e-01, 5.23925781e-01],
                [3.51806641e-01, 9.97070312e-01, 5.85327148e-02],
                [4.28955078e-01, 8.54003906e-01, 5.02929688e-01],
                [6.23046875e-01, 6.21582031e-01, 5.11718750e-01],
                [1.46362305e-01, 3.22265625e-02, 2.14721680e-01],
                [2.77587891e-01, 5.43945312e-01, 4.75097656e-01],
                [6.47460938e-01, 6.01074219e-01, 2.17651367e-01],
                [7.26074219e-01, 3.05175781e-01, 2.01904297e-01],
                [9.89257812e-01, 3.91113281e-01, 5.16967773e-02],
                [1.44531250e-01, 3.56201172e-01, 6.53320312e-01],
                [3.28613281e-01, 2.26898193e-02, 5.79589844e-01],
                [1.28295898e-01, 2.81982422e-01, 5.55664062e-01],
                [7.54394531e-01, 3.95751953e-01, 1.58203125e-01],
                [5.08499146e-03, 9.56420898e-02, 1.45751953e-01],
                [2.80517578e-01, 2.45483398e-01, 9.68261719e-01],
                [6.14257812e-01, 5.58105469e-01, 9.96093750e-01],
                [8.23242188e-01, 9.62890625e-01, 5.07812500e-01],
                [3.18603516e-01, 5.28808594e-01, 6.60156250e-01],
                [7.75390625e-01, 4.06494141e-01, 9.83886719e-01],
                [8.04199219e-01, 2.16674805e-01, 6.63085938e-01],
                [6.67114258e-02, 5.15441895e-02, 7.69531250e-01],
                [4.19433594e-01, 3.65478516e-01, 6.98730469e-01],
                [4.94140625e-01, 8.50219727e-02, 6.94824219e-01],
                [6.36230469e-01, 1.24145508e-01, 5.36132812e-01],
                [6.64062500e-01, 4.07958984e-01, 6.56250000e-01],
                [7.74902344e-01, 8.01269531e-01, 7.56835938e-01],
                [3.46679688e-01, 5.14648438e-01, 1.89575195e-01],
                [2.05322266e-01, 3.16894531e-01, 3.40576172e-01],
                [5.24902344e-01, 6.85501099e-03, 2.66601562e-01],
                [2.33276367e-01, 7.70996094e-01, 9.94140625e-01],
                [5.28808594e-01, 3.14208984e-01, 8.60839844e-01],
                [1.37084961e-01, 9.77050781e-01, 6.05957031e-01],
                [7.54882812e-01, 9.38964844e-01, 9.48242188e-01],
                [5.38085938e-01, 3.28674316e-02, 6.04003906e-01],
                [8.00292969e-01, 9.02832031e-01, 4.09912109e-01],
                [6.88964844e-01, 1.45996094e-01, 5.25390625e-01],
                [9.54589844e-01, 4.43359375e-01, 3.99780273e-02],
                [7.43408203e-02, 7.23632812e-01, 9.20410156e-01],
                [7.80273438e-01, 6.44042969e-01, 1.51596069e-02],
                [5.42480469e-01, 8.46557617e-02, 4.81262207e-02],
                [2.50732422e-01, 5.39550781e-01, 6.81152344e-01],
                [1.36230469e-01, 4.08630371e-02, 7.01660156e-01],
                [6.25488281e-01, 9.60449219e-01, 4.90722656e-02],
                [3.14208984e-01, 5.43945312e-01, 4.24072266e-01],
                [2.53173828e-01, 9.02832031e-01, 4.40917969e-01],
                [1.02783203e-01, 4.94628906e-01, 1.53198242e-01],
                [2.78076172e-01, 2.03735352e-01, 2.65625000e-01],
                [8.38867188e-01, 5.37597656e-01, 3.07617188e-02],
                [4.91638184e-02, 5.19531250e-01, 6.07910156e-01],
                [4.47509766e-01, 1.69372559e-03, 4.80712891e-01],
                [5.84472656e-01, 3.07861328e-01, 8.44238281e-01],
                [9.79492188e-01, 8.03222656e-01, 7.20703125e-01],
                [1.40014648e-01, 5.45898438e-01, 7.52929688e-01],
                [8.14941406e-01, 1.98608398e-01, 6.80175781e-01],
                [7.00195312e-01, 7.81250000e-03, 2.94189453e-01],
                [7.82226562e-01, 6.59179688e-01, 9.01855469e-01],
                [5.02929688e-01, 5.54687500e-01, 9.96582031e-01],
                [3.27636719e-01, 5.03417969e-01, 2.96142578e-01],
                [7.20214844e-01, 2.51220703e-01, 8.77441406e-01],
                [9.37988281e-01, 3.42529297e-01, 2.68554688e-01],
                [9.46777344e-01, 8.13476562e-01, 4.30664062e-01],
                [1.79565430e-01, 7.67135620e-03, 8.56933594e-01],
                [6.37695312e-01, 9.58496094e-01, 9.68750000e-01],
                [4.66552734e-01, 2.22778320e-01, 5.78613281e-01],
                [9.94628906e-01, 3.46679688e-01, 3.96728516e-01],
                [7.42187500e-01, 6.86523438e-01, 4.22851562e-01],
                [4.24072266e-01, 3.41552734e-01, 6.34277344e-01],
                [7.58300781e-01, 4.00146484e-01, 9.92675781e-01],
                [1.00463867e-01, 7.52441406e-01, 8.96972656e-01],
                [2.21435547e-01, 4.53948975e-03, 7.00683594e-01],
                [1.59179688e-01, 3.56933594e-01, 3.71093750e-01],
                [9.75585938e-01, 7.82226562e-01, 1.44287109e-01],
                [1.97021484e-01, 2.02178955e-02, 2.55859375e-01],
                [2.22473145e-02, 4.82177734e-01, 7.26074219e-01],
                [8.56445312e-01, 3.43505859e-01, 4.72167969e-01],
                [9.41894531e-01, 4.56054688e-01, 1.66503906e-01],
                [2.03247070e-01, 8.99414062e-01, 9.60937500e-01],
                [7.68066406e-01, 6.76757812e-01, 5.68359375e-01],
                [5.50781250e-01, 4.30175781e-01, 4.83398438e-01],
                [8.86230469e-01, 4.58007812e-01, 1.02050781e-01],
                [7.96875000e-01, 2.96142578e-01, 7.21191406e-01],
                [2.00683594e-01, 7.11914062e-01, 2.91748047e-01],
                [4.01611328e-01, 5.70800781e-01, 8.74023438e-01],
                [9.42382812e-01, 9.16992188e-01, 6.76757812e-01],
                [4.33593750e-01, 1.40258789e-01, 8.36914062e-01],
                [3.09814453e-01, 2.40966797e-01, 5.59082031e-01],
                [5.71289062e-01, 8.16894531e-01, 8.78417969e-01],
                [5.39550781e-01, 4.43847656e-01, 9.95117188e-01],
                [2.49145508e-01, 8.94042969e-01, 8.72070312e-01],
                [5.73730469e-01, 4.72412109e-01, 6.66992188e-01],
                [6.44531250e-01, 6.50390625e-01, 9.58984375e-01],
                [4.36035156e-01, 9.52148438e-01, 8.22265625e-01],
                [1.97265625e-01, 4.88525391e-01, 1.81152344e-01],
                [6.44531250e-01, 9.41406250e-01, 2.84912109e-01],
                [9.19433594e-01, 4.14306641e-01, 1.68457031e-01],
                [8.89160156e-01, 8.33496094e-01, 8.52050781e-01],
                [5.00488281e-01, 2.39013672e-01, 1.78588867e-01],
                [9.31152344e-01, 7.51953125e-01, 6.45019531e-01],
                [9.85839844e-01, 8.89648438e-01, 7.42675781e-01],
                [7.87109375e-01, 6.16699219e-01, 2.25952148e-01],
                [9.09179688e-01, 1.62597656e-01, 7.76855469e-01],
                [9.66796875e-01, 2.04833984e-01, 9.41772461e-02],
                [3.28613281e-01, 7.63183594e-01, 1.40380859e-01],
                [6.14746094e-01, 8.26660156e-01, 3.44238281e-01],
                [1.36962891e-01, 5.34179688e-01, 2.84271240e-02],
                [3.02978516e-01, 9.28710938e-01, 3.76464844e-01],
                [6.22070312e-01, 8.33007812e-01, 8.69628906e-01],
                [7.18261719e-01, 2.03369141e-01, 9.60449219e-01],
                [2.69775391e-01, 6.86035156e-01, 7.48046875e-01],
                [4.23095703e-01, 8.95996094e-01, 5.03921509e-03],
                [8.61328125e-01, 3.87939453e-01, 4.07714844e-01],
                [1.15234375e-01, 8.13964844e-01, 5.51269531e-01],
                [5.08300781e-01, 3.02490234e-01, 3.99169922e-01],
                [5.12695312e-02, 5.16601562e-01, 7.35351562e-01],
                [7.18261719e-01, 9.92187500e-01, 9.40429688e-01],
                [9.06738281e-01, 9.29687500e-01, 3.18603516e-01],
                [7.37792969e-01, 7.26074219e-01, 8.36425781e-01],
                [1.18103027e-02, 4.26330566e-02, 6.30859375e-01],
                [7.88085938e-01, 5.09277344e-01, 1.24755859e-01],
                [5.54687500e-01, 2.45239258e-01, 9.16748047e-02],
                [9.44335938e-01, 4.91943359e-01, 3.21777344e-01],
                [9.37988281e-01, 4.94628906e-01, 1.44775391e-01],
                [4.56787109e-01, 8.32519531e-01, 2.20092773e-01],
                [3.61816406e-01, 7.36816406e-01, 5.19042969e-01],
                [7.36328125e-01, 7.96386719e-01, 7.63671875e-01],
                [2.84271240e-02, 8.34960938e-01, 2.22656250e-01],
                [1.61254883e-01, 2.37915039e-01, 7.39257812e-01],
                [7.20214844e-01, 6.17187500e-01, 7.19238281e-01],
                [6.13769531e-01, 8.34472656e-01, 3.78906250e-01],
                [7.03613281e-01, 3.82995605e-03, 1.10839844e-01],
                [1.11572266e-01, 8.80859375e-01, 9.24804688e-01],
                [2.37792969e-01, 5.97656250e-01, 7.20703125e-01],
                [5.69335938e-01, 3.29833984e-01, 2.56591797e-01],
                [6.97265625e-01, 5.82031250e-01, 5.42480469e-01],
                [9.76074219e-01, 8.06152344e-01, 3.95019531e-01],
                [2.91015625e-01, 6.47949219e-01, 7.50488281e-01],
                [2.36328125e-01, 1.01562500e-01, 6.90429688e-01],
                [3.54614258e-02, 3.41796875e-01, 1.89331055e-01],
                [4.40917969e-01, 8.55468750e-01, 6.16210938e-01],
                [9.13085938e-01, 1.22619629e-01, 3.65905762e-02],
                [1.09436035e-01, 8.11035156e-01, 6.42700195e-02],
                [5.84960938e-01, 6.36718750e-01, 5.49316406e-01],
                [1.78588867e-01, 3.01269531e-01, 4.47021484e-01],
                [1.80541992e-01, 8.08593750e-01, 5.20996094e-01],
                [6.27929688e-01, 4.38476562e-01, 4.32617188e-01],
                [9.81445312e-01, 5.37109375e-01, 7.40234375e-01],
                [2.02270508e-01, 8.41308594e-01, 6.25976562e-01],
                [8.53027344e-01, 6.74316406e-01, 9.16137695e-02],
                [5.69335938e-01, 9.65820312e-01, 2.50976562e-01],
                [2.87109375e-01, 5.35156250e-01, 7.79785156e-01],
                [3.71093750e-01, 6.81640625e-01, 9.40429688e-01],
                [9.13574219e-01, 7.00683594e-01, 1.02905273e-01],
                [5.00000000e-01, 5.58593750e-01, 6.46484375e-01],
                [6.87500000e-01, 1.38671875e-01, 7.20214844e-01],
                [2.05535889e-02, 1.13403320e-01, 8.75488281e-01],
                [2.29003906e-01, 9.48730469e-01, 6.71386719e-01],
                [2.24243164e-01, 6.50878906e-01, 3.56933594e-01],
                [1.61499023e-01, 5.75561523e-02, 5.55175781e-01],
                [1.93725586e-01, 9.69726562e-01, 4.04785156e-01],
                [8.55468750e-01, 3.57666016e-02, 4.82666016e-01],
                [1.29089355e-02, 7.27050781e-01, 6.25976562e-01],
                [5.21972656e-01, 7.37792969e-01, 5.96191406e-01],
                [4.75341797e-01, 2.03735352e-01, 2.97851562e-01],
                [8.20800781e-01, 1.22802734e-01, 1.60644531e-01],
                [2.80761719e-01, 1.07543945e-01, 5.24902344e-01],
                [2.34130859e-01, 6.90429688e-01, 4.27734375e-01],
                [9.66796875e-01, 1.17111206e-02, 7.00195312e-01],
                [8.13964844e-01, 3.52539062e-01, 8.86230469e-01],
                [2.95654297e-01, 6.84204102e-02, 5.57617188e-01],
                [3.02490234e-01, 6.45996094e-01, 4.21630859e-01],
                [6.57226562e-01, 9.38476562e-01, 1.52832031e-01],
                [7.59765625e-01, 8.78417969e-01, 7.83691406e-01],
                [6.04980469e-01, 6.04980469e-01, 3.24951172e-01],
                [5.87890625e-01, 2.76367188e-01, 2.40722656e-01],
                [5.23437500e-01, 9.62890625e-01, 2.18994141e-01],
                [9.40551758e-02, 7.78808594e-01, 4.52880859e-01],
                [9.13574219e-01, 5.56640625e-01, 8.66210938e-01],
                [3.29589844e-01, 2.63519287e-02, 5.52246094e-01],
                [4.27246094e-01, 1.13586426e-01, 5.00488281e-01],
                [9.97558594e-01, 1.75781250e-02, 1.34887695e-01],
                [7.00683594e-01, 1.01074219e-01, 2.36572266e-01],
                [5.35156250e-01, 8.13476562e-01, 1.81762695e-01],
                [3.52294922e-01, 2.31445312e-01, 4.08935547e-01],
                [7.76855469e-01, 5.40527344e-01, 4.71191406e-01],
                [2.14355469e-01, 8.48632812e-01, 6.66992188e-01],
                [7.00195312e-01, 1.85424805e-01, 1.27807617e-01],
                [1.04187012e-01, 8.95996094e-01, 8.00292969e-01],
                [5.69152832e-02, 8.06640625e-01, 1.57348633e-01],
                [1.58935547e-01, 5.29296875e-01, 8.51562500e-01],
                [2.71728516e-01, 6.44531250e-01, 4.10888672e-01],
                [7.23144531e-01, 9.79980469e-01, 1.96289062e-01],
                [9.05761719e-01, 7.03125000e-01, 1.17340088e-02],
                [3.24951172e-01, 3.52783203e-01, 9.37500000e-01],
                [1.04980469e-01, 4.91943359e-01, 9.11132812e-01],
                [5.57327271e-03, 2.04467773e-01, 4.24316406e-01],
                [9.47265625e-01, 9.57519531e-01, 3.58886719e-01]]

        self.DATAs = np.reshape(test, (-1, 3)).astype(np.float16)
        for index in range(self.DATAs.shape[0]):
            while self.mapob[myint(self.DATAs[index][0] * self.mapx)][
                myint(self.DATAs[index][1] * self.mapy)] == self.OB:
                self.DATAs[index] = np.random.rand(3).astype(np.float16)
        self._mapmatrix = copy.copy(self.DATAs[:, 2])
        self.datas = self.DATAs[:, 0:2] * self.mapx
        self.totaldata = np.sum(self.DATAs[:, 2])
        log.log(self.DATAs)

        self._image_data = np.zeros((self.map.width, self.map.height)).astype(np.float16)
        self._image_position = np.zeros((self.sg.V['NUM_UAV'], self.map.width, self.map.height)).astype(np.float16)
        self.map.draw_wall(self._image_data)
        for i, position in enumerate(self.datas):
            self.map.draw_point(position[0], position[1], self._mapmatrix[i], self._image_data)
        for obstacle in self.sg.V['OBSTACLE']:
            self.map.draw_obstacle(obstacle[0], obstacle[1], obstacle[2], obstacle[3], self._image_data)
        for i_n in range(self.n):
            # layer 1
            self.map.draw_UAV(self.sg.V['INIT_POSITION'][0], self.sg.V['INIT_POSITION'][1], 1.,
                              self._image_position[i_n])
        # self.tr.print_diff()

    def reset(self):
        # initialize data map
        # tr = tracker.SummaryTracker()
        self.mapmatrix = copy.copy(self._mapmatrix)
        # ---- original
        # self.maptrack = np.zeros(self.mapmatrix.shape)
        # ---- new 6-7-11-28
        # self.maptrack = np.ones(self.mapmatrix.shape) * self.track
        # ---- 18:43
        self.maptrack = np.zeros(self.mapmatrix.shape)
        # ----
        # initialize positions of uavs
        self.uav = [list(self.sg.V['INIT_POSITION']) for i in range(self.n)]
        self.eff = [0.] * self.n
        self.count = 0
        self.zero = 0

        self.trace = [[] for i in range(self.n)]
        # initialize remaining energy
        self.energy = np.ones(self.n).astype(np.float64) * self.maxenergy
        # initialize indicators
        self.collection = np.zeros(self.n).astype(np.float16)
        self.walls = np.zeros(self.n).astype(np.int16)

        # time
        self.time_ = 0

        # initialize images
        self.state = self.__init_image()
        # print(self.fairness)
        # image = [np.reshape(np.array([self.image_data, self.image_position[i]]), (self.map.width, self.map.height, self.channel)) for i in range(self.n)]
        # tr.print_diff()
        return self.__get_state()

    def __init_image(self):
        self.image_data = copy.copy(self._image_data)
        self.image_position = copy.copy(self._image_position)
        # ---- or
        # self.image_track = np.zeros(self.image_position.shape)
        # ---- new 6-7-11-28
        # self.image_track = np.ones(self.image_data.shape) * self.track
        # ---- 18:43
        self.image_track = np.zeros(self.image_position.shape)
        # ----
        state = []
        for i in range(self.n):
            image = np.zeros((self.map.width, self.map.height, self.channel)).astype(np.float16)
            for width in range(image.shape[0]):
                for height in range(image.shape[1]):
                    image[width][height][0] = self.image_data[width][height]
                    image[width][height][1] = self.image_position[i][width][height]
                    # ---- new 6-7-11-28
                    # image[width][height][2] = self.image_track[width][height]
                    # ---- end new
            state.append(image)
        return state

    def __draw_image(self, clear_uav, update_point, update_track):
        for n in range(self.n):
            for i, value in update_point:
                self.map.draw_point(self.datas[i][0], self.datas[i][1], value, self.state[n][:, :, 0])
            self.map.clear_uav(clear_uav[n][0], clear_uav[n][1], self.state[n][:, :, 1])
            self.map.draw_UAV(self.uav[n][0], self.uav[n][1], self.energy[n] / self.maxenergy, self.state[n][:, :, 1])
            # ---- draw track
            for i, value in update_track:
                # ---- or
                self.map.draw_point(self.datas[i][0], self.datas[i][1], value, self.state[n][:, :, 2])
                # ---- new 6-7 16:15
                # self.map.draw_point(self.datas[i][0], self.datas[i][1], -1. * value, self.state[n][:,:,2])
                # ---- end new

    def __get_state(self):
        return copy.deepcopy(self.state)

    def __get_reward(self, value, distance):
        return value

    def __get_reward0(self, value, distance):
        return value * self.rdata / (distance + 0.01)

    def __get_reward1(self, value, distance):
        alpha = self.alpha
        return value * self.rdata / (distance + alpha * value + 0.01)

    def __get_reward2(self, value, distance):
        belta = 0.1  # * np.power(np.e, value)
        return (value * self.rdata + belta) / (distance + self.alpha * value + 0.01)

    def __get_reward3(self, value, distance):
        belta = 0.1 * np.power(np.e, value)
        return (value * self.rdata + belta) / (distance + self.alpha * value + 0.01)

    def __get_reward4(self, value, distance):
        if value != 0:
            factor0 = value * self.rdata / (distance + self.alpha * value + self.epsilon)
            # jain's fairness index
            square_of_sum = np.square(np.sum(self.mapmatrix[:]))
            sum_of_square = np.sum(np.square(self.mapmatrix[:]))
            jain_fairness_index = square_of_sum / sum_of_square / float(len(self.mapmatrix))
            return factor0 * jain_fairness_index
        else:
            return self.epsilon / (distance + self.epsilon)

    def __get_reward5(self, value, distance, mapmatrix=None):
        if mapmatrix is None:
            if value != 0:
                # print(value)
                factor0 = value / (distance + self.alpha * value + self.epsilon)
                # jain's fairness index
                square_of_sum = np.square(np.sum(self.mapmatrix[:]))
                sum_of_square = np.sum(np.square(self.mapmatrix[:]))
                jain_fairness_index = square_of_sum / sum_of_square / float(len(self.mapmatrix))
                return factor0 * jain_fairness_index
            else:
                return - 1. * self.normalize * distance
        else:
            if value != 0:
                factor0 = value / (distance + self.alpha * value + self.epsilon)
                # jain's fairness index
                square_of_sum = np.square(np.sum(mapmatrix[:]))
                sum_of_square = np.sum(np.square(mapmatrix[:]))
                jain_fairness_index = square_of_sum / sum_of_square / float(len(mapmatrix))
                return factor0 * jain_fairness_index
            else:
                return - 1. * self.normalize * distance

    def __get_reward6(self, value, distance, mapmatrix=None):
        if mapmatrix is None:
            if value != 0:
                # print(value)
                factor0 = value / (distance + self.alpha * value + self.epsilon)
                # jain's fairness index
                collection = self._mapmatrix - self.mapmatrix
                square_of_sum = np.square(np.sum(collection))
                sum_of_square = np.sum(np.square(collection))
                jain_fairness_index = square_of_sum / sum_of_square / float(len(collection))
                return factor0 * jain_fairness_index
            else:
                return -1. * self.normalize * distance
        else:
            if value != 0:
                factor0 = value / (distance + self.alpha * value + self.epsilon)
                # jain's fairness index
                collection = self._mapmatrix - mapmatrix
                square_of_sum = np.square(np.sum(collection))
                sum_of_square = np.sum(np.square(collection))
                jain_fairness_index = square_of_sum / sum_of_square / float(len(collection))
                return factor0 * jain_fairness_index
            else:
                return -1. * self.normalize * distance

    def __get_reward7(self, value, distance, fairness, fairness_):
        if value != 0:
            factor0 = value / (distance + self.alpha * value + self.epsilon)
            delta_fairness = fairness_ - fairness
            # print(delta_fairness)
            return factor0 * delta_fairness
        else:
            return -1. * self.normalize * distance

    def __get_reward8(self, value, distance, fairness, fairness_):
        if value != 0:
            factor0 = value / (distance + self.alpha * value + self.epsilon)
            delta_fairness = fairness_ - fairness
            # print(delta_fairness)
            return factor0 * delta_fairness
        else:
            return self.normalize * self.pstep

    def __get_reward9(self, value, distance, fairness, fairness_):
        if value != 0:
            # ---- or
            # factor0 = value / (distance + self.alpha * value + self.epsilon)
            # ---- 6-8 14:48
            factor0 = value / (self.factor * distance + self.alpha * value + self.epsilon)
            # ----
            # delta_fairness = np.fabs(fairness_ - fairness)
            # print(delta_fairness)
            return factor0 * fairness_
        else:
            return -1. * self.normalize * distance

    def __get_fairness(self, values):
        square_of_sum = np.square(np.sum(values))
        sum_of_square = np.sum(np.square(values))
        if sum_of_square == 0:
            return 0.
        jain_fairness_index = square_of_sum / sum_of_square / float(len(values))
        return jain_fairness_index

    def __get_eff(self, value, distance):
        return value / self.maxenergy

    def __get_eff1(self, value, distance):
        return value / (distance + self.alpha * value + self.epsilon)

    def __cusume_energy0(self, uav, value, distance):
        self.energy[uav] -= distance

    def __cusume_energy1(self, uav, value, distance, energy=None):
        if energy is None:
            # ---- or
            # self.erengy[uav] -= (distance + value * self.alpha)
            # ---- 6-8 14:48
            self.energy[uav] -= (self.factor * distance + self.alpha * value)
            # ----
        else:
            # ---- or
            # energy[uav] -= (distance + value * self.alpha)
            # ---- 14:48
            energy[uav] -= (self.factor * distance + self.alpha * value)
            # ----

    # ---- or
    # def step(self, action):
    #     self.count += 1
    #     actions = copy.deepcopy(action)
    #     normalize = self.normalize
    #     for i in range(self.n):
    #         for ii in actions[i]:
    #             if np.isnan(ii):
    #                 print('Nan')
    #                 while True:
    #                     pass
    # ---- 6-8 10:57
    def step(self, actions,indicator=None):
        self.count += 1
        action = copy.deepcopy(actions)
        # 6-20 00:43
        if np.max(action) > self.maxaction:
            self.maxaction = np.max(action)
            # print(self.maxaction)
        if np.min(action) < self.minaction:
            self.minaction = np.min(action)
            # print(self.minaction)
        action = np.clip(action, -1e3, 1e3)  #

        normalize = self.normalize

        #TODO:梯度爆炸问题不可小觑,
        # 遇到nan直接卡掉
        for i in range(self.n):
            for ii in action[i]:
                if np.isnan(ii):
                    print('Nan')
                    while True:
                        pass

        reward = [0] * self.n
        self.dn = [False] * self.n  # no energy UAV
        update_points = []
        update_tracks = []
        clear_uav = copy.copy(self.uav)
        new_positions = []
        c_f = self.__get_fairness(self.maptrack)
        # update positions of UAVs
        for i in range(self.n):
            self.trace[i].append(self.uav[i])
            distance = np.sqrt(np.power(action[i][0], 2) + np.power(action[i][1], 2))
            data = 0

            if distance <= self.maxdistance and self.energy[i] >= distance:
                new_x = self.uav[i][0] + action[i][0]
                new_y = self.uav[i][1] + action[i][1]
            else:
                maxdistance = self.maxdistance if self.maxdistance <= self.energy[i] else self.energy[i]
                if distance <= self.epsilon:
                    distance = self.epsilon
                    print("very small.")
                new_x = self.uav[i][0] + maxdistance * action[i][0] / distance
                new_y = self.uav[i][1] + maxdistance * action[i][1] / distance
                distance = maxdistance
            if distance <= self.epsilon:
                self.zero += 1
            self.__cusume_energy1(i, 0, distance)
            if 0 <= new_x < self.mapx and 0 <= new_y < self.mapy and self.mapob[myint(new_x)][myint(new_y)] != self.OB:
                new_positions.append([new_x, new_y])
            else:
                new_positions.append([self.uav[i][0], self.uav[i][1]])
                reward[i] += normalize * self.pwall
                self.walls[i] += 1
            # calculate distances between UAV and data points
            _pos = np.repeat([new_positions[-1]], [self.datas.shape[0]], axis=0)
            _minus = self.datas - _pos
            _power = np.power(_minus, 2)
            _dis = np.sum(_power, axis=1)
            for index, dis in enumerate(_dis):
                if np.sqrt(dis) <= self.crange:
                    self.maptrack[index] += self.track
                    update_tracks.append([index, self.maptrack[index]])
                    if self.mapmatrix[index] > 0:
                        data += self._mapmatrix[index] * self.cspeed
                        self.mapmatrix[index] -= self._mapmatrix[index] * self.cspeed
                        if self.mapmatrix[index] < 0:
                            self.mapmatrix[index] = 0.
                        update_points.append([index, self.mapmatrix[index]])
            # update info
            value = data if self.energy[i] >= data * self.alpha else self.energy[i]
            self.__cusume_energy1(i, value, 0.)
            c_f_ = self.__get_fairness(self.maptrack)
            # ---- 6-7
            # 11:32
            # reward[i] += self.__get_reward9(value, distance, c_f, c_f_)
            # ---- 11:44
            # reward[i] += self.__get_reward7(value, distance, c_f, c_f_)
            # ---- 18:43
            reward[i] += self.__get_reward9(value, distance, c_f, c_f_)
            # ----
            c_f = c_f_
            self.eff[i] += self.__get_eff1(value, distance)
            self.collection[i] += value
            if self.energy[i] <= self.epsilon * self.maxenergy:
                self.dn[i] = True
        self.uav = new_positions
        t = time.time()
        self.__draw_image(clear_uav, update_points, update_tracks)
        self.time_ += time.time() - t
        # ---- or
        reward = list(np.clip(np.array(reward) / normalize, -2., 1.))
        # ---- new 18:43
        # reward = list(np.clip(np.array(reward) / normalize, -1., 1.))
        # ---- end new
        info = None
        state = self.__get_state()
        for r in reward:
            if np.isnan(r):
                print('Rerward Nan')
                while True:
                    pass
        return state, reward, sum(self.dn), info,indicator

    def render(self):
        print('coding...')

    @property
    def leftrewards(self):
        return np.sum(self.mapmatrix) / self.totaldata

    @property
    def efficiency(self):
        return np.sum(self.collection / self.totaldata) / (
                    self.n - np.sum(self.normal_energy)) * self.collection_fairness

    @property
    def normal_energy(self):
        return list(np.array(self.energy) / self.maxenergy)

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
