from experiments.crazy_env.env_setting3 import Setting
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
        test = [[3.9233398438e-01, 8.8378906250e-01, 5.8251953125e-01],
                [2.9370117188e-01, 6.7773437500e-01, 1.2719726562e-01],
                [9.3603515625e-01, 1.3024902344e-01, 1.7419433594e-01],
                [9.6826171875e-01, 3.6767578125e-01, 8.5742187500e-01],
                [5.8837890625e-01, 7.7978515625e-01, 8.9306640625e-01],
                [1.4114379883e-02, 8.3056640625e-01, 2.5146484375e-01],
                [6.1132812500e-01, 7.3828125000e-01, 3.6743164062e-01],
                [5.6054687500e-01, 5.3173828125e-01, 1.7956542969e-01],
                [5.7275390625e-01, 2.4987792969e-01, 2.9248046875e-01],
                [2.0858764648e-02, 5.3320312500e-01, 8.9746093750e-01],
                [4.8657226562e-01, 4.5043945312e-01, 8.4277343750e-01],
                [2.3101806641e-02, 8.2861328125e-01, 7.8027343750e-01],
                [4.5849609375e-01, 5.0585937500e-01, 1.9912719727e-02],
                [7.4755859375e-01, 3.9941406250e-01, 7.5146484375e-01],
                [2.6440429688e-01, 9.3322753906e-02, 9.6969604492e-03],
                [1.6479492188e-01, 8.2958984375e-01, 3.9770507812e-01],
                [3.9892578125e-01, 8.9062500000e-01, 1.8579101562e-01],
                [5.8300781250e-01, 7.8320312500e-01, 1.6748046875e-01],
                [8.8378906250e-01, 5.1953125000e-01, 9.1113281250e-01],
                [2.3034667969e-01, 8.8330078125e-01, 1.1303710938e-01],
                [4.8681640625e-01, 3.0834960938e-01, 3.4423828125e-01],
                [1.7724609375e-01, 8.8134765625e-01, 3.2324218750e-01],
                [5.2246093750e-01, 2.0507812500e-01, 4.6850585938e-01],
                [6.2548828125e-01, 8.9697265625e-01, 5.4248046875e-01],
                [9.4238281250e-01, 1.9445800781e-01, 2.4328613281e-01],
                [6.0595703125e-01, 7.8515625000e-01, 6.2065124512e-03],
                [5.3564453125e-01, 5.4150390625e-01, 4.9243164062e-01],
                [9.9218750000e-01, 5.0585937500e-01, 8.4960937500e-01],
                [7.3046875000e-01, 1.6955566406e-01, 4.1528320312e-01],
                [9.5849609375e-01, 2.5585937500e-01, 9.2822265625e-01],
                [9.2529296875e-01, 3.3813476562e-02, 8.0712890625e-01],
                [6.6528320312e-02, 7.2167968750e-01, 9.2187500000e-01],
                [4.8876953125e-01, 5.5664062500e-01, 7.8515625000e-01],
                [8.2946777344e-02, 8.2324218750e-01, 7.4902343750e-01],
                [2.9394531250e-01, 8.5986328125e-01, 3.7646484375e-01],
                [1.1816406250e-01, 7.2558593750e-01, 3.7280273438e-01],
                [7.9638671875e-01, 3.3325195312e-01, 8.3349609375e-01],
                [3.5552978516e-02, 6.8994140625e-01, 6.8603515625e-01],
                [4.9414062500e-01, 9.4335937500e-01, 5.7861328125e-01],
                [8.3593750000e-01, 5.7470703125e-01, 5.6347656250e-01],
                [7.8955078125e-01, 2.2985839844e-01, 9.2822265625e-01],
                [6.0351562500e-01, 8.8439941406e-02, 9.9511718750e-01],
                [4.2529296875e-01, 7.5048828125e-01, 3.8452148438e-01],
                [4.3212890625e-01, 6.5429687500e-01, 8.1884765625e-01],
                [4.8974609375e-01, 1.6333007812e-01, 1.3037109375e-01],
                [8.9062500000e-01, 5.6005859375e-01, 6.3720703125e-01],
                [2.3815917969e-01, 8.8623046875e-01, 9.8486328125e-01],
                [3.0737304688e-01, 5.9375000000e-01, 7.6660156250e-01],
                [4.1992187500e-01, 5.0195312500e-01, 2.1374511719e-01],
                [4.4784545898e-03, 5.2343750000e-01, 1.8859863281e-01],
                [1.5759277344e-01, 5.0439453125e-01, 9.4787597656e-02],
                [7.1289062500e-01, 7.9003906250e-01, 3.5064697266e-02],
                [9.6728515625e-01, 2.0126342773e-02, 6.8798828125e-01],
                [8.2373046875e-01, 2.1704101562e-01, 9.9609375000e-02],
                [7.8027343750e-01, 4.1064453125e-01, 4.6655273438e-01],
                [1.8762207031e-01, 7.8027343750e-01, 3.4643554688e-01],
                [6.0302734375e-01, 9.0332031250e-01, 2.5634765625e-01],
                [5.8203125000e-01, 3.0981445312e-01, 2.7685546875e-01],
                [6.9726562500e-01, 5.4833984375e-01, 6.8298339844e-02],
                [4.0454101562e-01, 7.9150390625e-01, 5.3906250000e-01],
                [2.6171875000e-01, 5.3955078125e-01, 2.1594238281e-01],
                [2.4243164062e-01, 7.2021484375e-01, 5.0439453125e-01],
                [3.1030273438e-01, 9.0869140625e-01, 9.4189453125e-01],
                [5.9033203125e-01, 4.0435791016e-02, 3.8403320312e-01],
                [4.3847656250e-01, 8.4082031250e-01, 1.7822265625e-01],
                [6.0205078125e-01, 5.8935546875e-01, 8.8684082031e-02],
                [5.7128906250e-01, 5.2685546875e-01, 8.5888671875e-01],
                [9.6044921875e-01, 5.3417968750e-01, 5.8593750000e-01],
                [4.2700195312e-01, 8.3056640625e-01, 4.6142578125e-01],
                [7.1826171875e-01, 2.3205566406e-01, 8.7402343750e-01],
                [7.3828125000e-01, 8.9746093750e-01, 2.1252441406e-01],
                [5.3222656250e-02, 5.8154296875e-01, 5.6884765625e-01],
                [1.0000000000e+00, 2.0764160156e-01, 4.7485351562e-01],
                [6.3476562500e-01, 9.1943359375e-01, 4.5312500000e-01],
                [9.2529296875e-02, 8.5644531250e-01, 6.8750000000e-01],
                [7.2998046875e-01, 5.1025390625e-01, 6.3671875000e-01],
                [5.1025390625e-01, 9.3896484375e-01, 7.2656250000e-01],
                [2.1789550781e-01, 3.2055664062e-01, 2.9956054688e-01],
                [1.9775390625e-01, 8.9843750000e-01, 1.4266967773e-02],
                [1.6650390625e-01, 2.4133300781e-01, 2.7514648438e-01],
                [7.9248046875e-01, 3.5009765625e-01, 6.8261718750e-01],
                [5.6396484375e-01, 5.1611328125e-01, 4.6215820312e-01],
                [5.7763671875e-01, 1.1627197266e-01, 4.1625976562e-02],
                [4.6118164062e-01, 2.2570800781e-01, 6.6503906250e-01],
                [1.7150878906e-01, 8.2275390625e-01, 9.9414062500e-01],
                [3.5375976562e-01, 9.1162109375e-01, 8.1054687500e-01],
                [4.3304443359e-02, 1.5466308594e-01, 8.6364746094e-02],
                [2.8613281250e-01, 9.9511718750e-01, 6.2597656250e-01],
                [1.1926269531e-01, 4.9169921875e-01, 5.5664062500e-01],
                [9.8876953125e-01, 8.2568359375e-01, 2.8662109375e-01],
                [8.8085937500e-01, 2.6025390625e-01, 3.6254882812e-02],
                [1.5234375000e-01, 3.2348632812e-01, 9.8583984375e-01],
                [5.1464843750e-01, 4.5166015625e-02, 5.6103515625e-01],
                [7.6708984375e-01, 6.6162109375e-01, 6.1767578125e-01],
                [6.1718750000e-01, 7.4609375000e-01, 9.0185546875e-01],
                [4.5483398438e-01, 2.3818969727e-02, 9.9707031250e-01],
                [8.2910156250e-01, 3.3984375000e-01, 5.9783935547e-02],
                [2.1194458008e-02, 6.1816406250e-01, 1.4379882812e-01],
                [4.0551757812e-01, 7.1289062500e-01, 8.2666015625e-01],
                [7.7734375000e-01, 3.2543945312e-01, 5.8398437500e-01],
                [6.6601562500e-01, 9.8925781250e-01, 9.6923828125e-01],
                [2.7709960938e-01, 3.2958984375e-01, 7.4267578125e-01],
                [2.7661132812e-01, 8.9746093750e-01, 8.4716796875e-01],
                [9.2138671875e-01, 8.2128906250e-01, 8.5742187500e-01],
                [2.5585937500e-01, 5.5511474609e-02, 5.1123046875e-01],
                [5.3125000000e-01, 4.2431640625e-01, 9.3017578125e-01],
                [5.8886718750e-01, 7.8369140625e-01, 3.0014038086e-02],
                [9.5458984375e-01, 7.6953125000e-01, 6.0400390625e-01],
                [7.3779296875e-01, 5.7714843750e-01, 6.7773437500e-01],
                [1.6616821289e-02, 3.7744140625e-01, 4.0917968750e-01],
                [9.5361328125e-01, 4.9731445312e-01, 9.0484619141e-03],
                [7.6806640625e-01, 7.4658203125e-01, 8.7792968750e-01],
                [7.9150390625e-01, 3.1298828125e-01, 2.5927734375e-01],
                [6.6210937500e-01, 9.2285156250e-01, 5.5371093750e-01],
                [9.8535156250e-01, 3.9624023438e-01, 2.7148437500e-01],
                [8.1640625000e-01, 7.5830078125e-01, 4.4311523438e-01],
                [7.4157714844e-02, 5.4833984375e-01, 3.3178710938e-01],
                [5.9863281250e-01, 5.8837890625e-01, 3.1762695312e-01],
                [4.8461914062e-01, 4.6752929688e-02, 4.5410156250e-01],
                [8.9257812500e-01, 5.8691406250e-01, 8.4375000000e-01],
                [2.1270751953e-02, 7.5683593750e-01, 8.7939453125e-01],
                [7.6171875000e-01, 7.6660156250e-01, 5.2490234375e-01],
                [6.1181640625e-01, 3.6718750000e-01, 9.5410156250e-01],
                [3.2153320312e-01, 4.9414062500e-01, 4.8168945312e-01],
                [7.5927734375e-01, 2.9760742188e-01, 4.8645019531e-02],
                [2.0507812500e-01, 1.6860961914e-02, 9.1650390625e-01],
                [4.9047851562e-01, 3.1396484375e-01, 7.7294921875e-01],
                [3.2836914062e-01, 1.1798095703e-01, 2.4993896484e-02],
                [5.4248046875e-01, 5.4736328125e-01, 6.6992187500e-01],
                [9.6679687500e-01, 2.6611328125e-01, 4.0600585938e-01],
                [4.0625000000e-01, 8.3251953125e-01, 6.7968750000e-01],
                [5.3027343750e-01, 3.7890625000e-01, 7.9492187500e-01],
                [4.0527343750e-02, 1.0119628906e-01, 4.9072265625e-01],
                [5.7568359375e-01, 5.4492187500e-01, 4.8632812500e-01],
                [3.6621093750e-01, 9.2333984375e-01, 1.2487792969e-01],
                [8.0664062500e-01, 2.7172851562e-01, 2.6391601562e-01],
                [2.9296875000e-01, 1.6857910156e-01, 7.1289062500e-01],
                [3.2177734375e-01, 5.7617187500e-01, 9.9072265625e-01],
                [1.0443115234e-01, 9.2822265625e-01, 6.9091796875e-01],
                [8.5678100586e-03, 5.9277343750e-01, 8.0761718750e-01],
                [3.2592773438e-01, 5.5664062500e-01, 9.8291015625e-01],
                [4.3066406250e-01, 8.5644531250e-01, 4.7753906250e-01],
                [9.9487304688e-02, 3.3251953125e-01, 5.7861328125e-01],
                [5.9619140625e-01, 5.7226562500e-01, 9.8291015625e-01],
                [9.6386718750e-01, 4.0844726562e-01, 4.9414062500e-01],
                [8.5693359375e-02, 4.3334960938e-01, 8.1591796875e-01],
                [7.6855468750e-01, 4.2138671875e-01, 2.4548339844e-01],
                [1.5563964844e-01, 8.6425781250e-01, 4.7338867188e-01],
                [7.2558593750e-01, 7.5341796875e-01, 7.1093750000e-01],
                [3.7727355957e-03, 2.7343750000e-01, 4.3823242188e-01],
                [1.0491943359e-01, 6.4111328125e-01, 3.0444335938e-01],
                [1.0858154297e-01, 7.0068359375e-01, 9.1796875000e-01],
                [8.0810546875e-01, 7.3583984375e-01, 6.4160156250e-01],
                [3.0273437500e-01, 7.5134277344e-02, 9.2138671875e-01],
                [5.5126953125e-01, 5.0097656250e-01, 6.2548828125e-01],
                [4.9023437500e-01, 3.1372070312e-01, 4.2846679688e-01],
                [9.5410156250e-01, 2.0593261719e-01, 9.0771484375e-01],
                [9.9218750000e-01, 5.9423828125e-01, 3.8305664062e-01],
                [7.3779296875e-01, 5.5517578125e-01, 8.3105468750e-01],
                [7.6220703125e-01, 4.3188476562e-01, 7.0495605469e-02],
                [1.3391113281e-01, 8.8476562500e-01, 7.2656250000e-01],
                [4.5214843750e-01, 3.7158203125e-01, 8.2324218750e-01],
                [9.5703125000e-01, 7.7441406250e-01, 9.0454101562e-02],
                [2.2753906250e-01, 3.6206054688e-01, 5.6347656250e-01],
                [4.6191406250e-01, 6.0693359375e-01, 2.7490234375e-01],
                [2.8149414062e-01, 9.7351074219e-02, 5.5175781250e-01],
                [3.6181640625e-01, 6.1621093750e-01, 2.7929687500e-01],
                [2.8173828125e-01, 6.0595703125e-01, 8.1738281250e-01],
                [4.7241210938e-01, 3.9038085938e-01, 8.7890625000e-01],
                [2.5659179688e-01, 3.4082031250e-01, 1.0223388672e-01],
                [7.4658203125e-01, 2.0495605469e-01, 8.7548828125e-01],
                [6.1474609375e-01, 7.6562500000e-01, 7.6513671875e-01],
                [4.5117187500e-01, 6.2841796875e-01, 4.6875000000e-01],
                [5.4718017578e-02, 9.6826171875e-01, 1.4208984375e-01],
                [4.1528320312e-01, 6.2451171875e-01, 8.5693359375e-01],
                [2.1606445312e-01, 1.9763183594e-01, 2.5268554688e-01],
                [5.8593750000e-01, 8.0444335938e-02, 3.9233398438e-01],
                [8.3007812500e-01, 8.3984375000e-02, 7.3828125000e-01],
                [7.2070312500e-01, 8.4667968750e-01, 4.3457031250e-01],
                [5.9033203125e-01, 9.1943359375e-01, 7.8271484375e-01],
                [9.2431640625e-01, 2.8540039062e-01, 9.9609375000e-01],
                [9.7558593750e-01, 3.4350585938e-01, 9.9511718750e-01],
                [2.5927734375e-01, 1.2939453125e-01, 8.0761718750e-01],
                [6.7236328125e-01, 9.0478515625e-01, 8.1103515625e-01],
                [5.1171875000e-01, 1.7944335938e-01, 1.8273925781e-01],
                [8.0859375000e-01, 6.4746093750e-01, 6.5368652344e-02],
                [7.4658203125e-01, 2.5244140625e-01, 8.8134765625e-01],
                [9.3212890625e-01, 2.3107910156e-01, 8.4570312500e-01],
                [6.0386657715e-03, 7.2851562500e-01, 2.7929687500e-01],
                [7.6416015625e-01, 5.0927734375e-01, 9.0136718750e-01],
                [8.7500000000e-01, 5.8789062500e-01, 4.1821289062e-01],
                [2.2766113281e-01, 2.2912597656e-01, 9.2626953125e-01],
                [1.5673828125e-01, 8.3105468750e-01, 3.9208984375e-01],
                [3.2177734375e-01, 5.0097656250e-01, 3.4545898438e-01],
                [4.5947265625e-01, 1.0040283203e-01, 7.8125000000e-01],
                [4.2449951172e-02, 3.6791992188e-01, 7.0556640625e-01],
                [7.5866699219e-02, 7.4658203125e-01, 3.9672851562e-01],
                [6.7565917969e-02, 7.1484375000e-01, 1.6098022461e-02],
                [7.1484375000e-01, 6.8750000000e-01, 6.6894531250e-01],
                [3.0664062500e-01, 4.6606445312e-01, 5.6494140625e-01],
                [2.6049804688e-01, 1.9201660156e-01, 2.0056152344e-01],
                [5.0872802734e-02, 3.3422851562e-01, 1.2408447266e-01],
                [5.2825927734e-02, 5.7714843750e-01, 1.1419677734e-01],
                [9.5263671875e-01, 1.0235595703e-01, 5.2343750000e-01],
                [9.2089843750e-01, 6.5087890625e-01, 3.5351562500e-01],
                [1.1315917969e-01, 4.6166992188e-01, 2.0593261719e-01],
                [1.0797119141e-01, 6.5576171875e-01, 1.5551757812e-01],
                [9.2089843750e-01, 8.6181640625e-01, 6.7578125000e-01],
                [9.0087890625e-01, 5.3906250000e-01, 1.4245605469e-01],
                [7.0524215698e-04, 4.7192382812e-01, 7.1716308594e-02],
                [6.8017578125e-01, 2.0251464844e-01, 3.8598632812e-01],
                [8.3984375000e-02, 7.1679687500e-01, 1.4501953125e-01],
                [7.7734375000e-01, 5.4296875000e-01, 7.2509765625e-01],
                [2.2607421875e-01, 4.7875976562e-01, 8.9013671875e-01],
                [6.3378906250e-01, 5.8935546875e-01, 5.6835937500e-01],
                [5.5007934570e-03, 7.5439453125e-02, 6.4013671875e-01],
                [2.3962402344e-01, 8.8964843750e-01, 5.3027343750e-01],
                [4.5971679688e-01, 2.4780273438e-01, 4.8583984375e-01],
                [6.9677734375e-01, 9.9853515625e-01, 9.9169921875e-01],
                [4.8803710938e-01, 5.8056640625e-01, 1.8933105469e-01],
                [1.3513183594e-01, 6.5527343750e-01, 5.9619140625e-01],
                [1.1431884766e-01, 6.8017578125e-01, 7.9727172852e-03],
                [9.3310546875e-01, 7.1484375000e-01, 8.2080078125e-01],
                [2.0593261719e-01, 7.6123046875e-01, 7.4951171875e-01],
                [4.8168945312e-01, 6.7480468750e-01, 4.6948242188e-01],
                [9.1015625000e-01, 5.7763671875e-01, 8.7841796875e-01],
                [1.4184570312e-01, 3.4765625000e-01, 7.1777343750e-01],
                [4.0252685547e-02, 3.6572265625e-01, 5.0683593750e-01],
                [9.0234375000e-01, 6.4843750000e-01, 7.9541015625e-01],
                [4.7558593750e-01, 1.7468261719e-01, 7.3632812500e-01],
                [7.5976562500e-01, 1.9836425781e-01, 5.0830078125e-01],
                [7.6464843750e-01, 6.5771484375e-01, 9.8681640625e-01],
                [1.8908691406e-01, 2.1398925781e-01, 7.6708984375e-01],
                [3.5424804688e-01, 6.6455078125e-01, 3.5229492188e-01],
                [6.1132812500e-01, 2.2412109375e-01, 3.7158203125e-01],
                [7.3535156250e-01, 3.2348632812e-01, 8.3923339844e-02],
                [3.4594726562e-01, 6.6796875000e-01, 6.4013671875e-01],
                [6.6162109375e-01, 7.5097656250e-01, 5.9765625000e-01],
                [7.7539062500e-01, 8.9306640625e-01, 7.4560546875e-01],
                [9.9218750000e-01, 4.0893554688e-01, 9.7802734375e-01],
                [7.3876953125e-01, 5.7568359375e-01, 5.2929687500e-01],
                [1.7639160156e-01, 7.0458984375e-01, 7.2167968750e-01],
                [4.6533203125e-01, 7.2363281250e-01, 9.7363281250e-01],
                [5.4199218750e-01, 3.5107421875e-01, 3.3618164062e-01],
                [3.8745117188e-01, 4.8168945312e-01, 8.1689453125e-01],
                [1.6159057617e-02, 7.4218750000e-01, 2.5537109375e-01],
                [2.0007324219e-01, 9.0185546875e-01, 9.1845703125e-01],
                [1.6320800781e-01, 8.9550781250e-01, 4.6655273438e-01],
                [5.7763671875e-01, 1.5380859375e-01, 4.4409179688e-01],
                [2.6382446289e-02, 2.5772094727e-02, 9.6630859375e-01],
                [9.6337890625e-01, 1.6333007812e-01, 6.4404296875e-01],
                [5.3417968750e-01, 1.4575195312e-01, 2.9467773438e-01],
                [2.7197265625e-01, 1.8444824219e-01, 6.9873046875e-01],
                [4.9438476562e-01, 9.8583984375e-01, 4.5507812500e-01],
                [7.7441406250e-01, 3.0004882812e-01, 3.2275390625e-01],
                [6.5966796875e-01, 9.0966796875e-01, 7.4890136719e-02]]

        # DATA shape:256*3
        self.DATAs = np.reshape(test, (-1, 3)).astype(np.float16)

        # #TODO:调点
        # dx = [-0.2, -0.2, -0.2, 0, 0, 0, 0.2, 0.2, 0.2]
        # dy = [-0.2, 0, 0.2, -0.2, 0, 0.2, -0.2, 0, 0.2]
        # # replace the POI in obstacle position with the POI out of obstacle position
        # for index in range(self.DATAs.shape[0]):
        #     need_adjust=True
        #     while need_adjust:
        #         need_adjust=False
        #         for i in range(len(dx)):
        #             if self.mapob[min(myint(self.DATAs[index][0] * self.mapx + dx[i]), self.mapx - 1)][
        #                 min(myint(self.DATAs[index][1] * self.mapy + dy[i]), self.mapy - 1)] == self.OB:
        #                 need_adjust=True
        #                 break
        #         if need_adjust is True:
        #             self.DATAs[index] = np.random.rand(3).astype(np.float16)
        #
        # for i, poi_i in enumerate(self.DATAs):
        #     if i == 0:
        #         print("[[%.10e,%.10e,%.10e]," % (poi_i[0], poi_i[1], poi_i[2]))
        #     elif i == 255:
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
            [0.1875, 0.8125],
            [0.625, 0.8125],
            [0.5, 0.5],
            [0.375, 0.125],
            [0.875, 0.25]
        ]

        self.FILL = np.reshape(station, (-1, 2)).astype(np.float16)

        # Fill Station Position  [5,2]
        self.fills = self.FILL[:, 0:2] * self.mapx

        # # Fill Station remain energy [5]
        # self.fills_energy_remain = copy.copy(self.FILL[:, 2])
        #
        # # sum of all FIll Station remain energy
        # self.total_fills_energy_remain = np.sum(self.FILL[:, 2])

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
                self.map.draw_FillStation(position[0], position[1], self._image_position[i_n])

        # 无人机随机颜色
        self.uav_render_color = []
        for i in range(self.n):
            self.uav_render_color.append(np.random.randint(low=0, high=255, size=3, dtype=np.uint8))

        self.pow_list = []

    def reset(self):
        # initialize data map
        # tr = tracker.SummaryTracker()
        self.mapmatrix = copy.copy(self._mapmatrix)
        # self.fills_energy_remain = copy.copy(self.FILL[:, 2])

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

            for i, position in enumerate(self.fills):
                self.map.draw_FillStation(position[0], position[1], self.state[n][:, :, 1])

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
        factor1 = energy / (self.maxenergy * self.fspeed)
        reward = factor0 + factor1
        if value == 0 and energy == 0:  # 浪费生命的一步
            return penalty - self.normalize * 1.0
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

        return distance

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
                penalty += -1. * normalize

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
                    self.__fill_energy1(i)
                    # # uodate poi data
                    # if self.fills_energy_remain[index] > 0:
                    #     # TODO:加油站的信息更新
                    #     if self.fspeed * self.maxenergy <= self.fills_energy_remain[index]:
                    #         if self.energy[i] + self.fspeed * self.maxenergy <= self.maxenergy:
                    #             self.fill_energy[i] += self.fspeed * self.maxenergy
                    #             self.fills_energy_remain[index] -= self.fspeed * self.maxenergy
                    #             self.energy[i] += self.fspeed * self.maxenergy
                    #         else:
                    #             self.fill_energy[i] += self.maxenergy - self.energy[i]
                    #             self.fills_energy_remain[index] -= (self.maxenergy - self.energy[i])
                    #             self.energy[i] = self.maxenergy
                    #     else:
                    #         if self.energy[i] + self.fills_energy_remain[index] <= self.maxenergy:
                    #             self.fill_energy[i] += self.fills_energy_remain[index]
                    #             self.energy[i] += self.fills_energy_remain[index]
                    #             self.fills_energy_remain[index] = 0
                    #         else:
                    #             self.fill_energy[i] += self.maxenergy - self.energy[i]
                    #             self.fills_energy_remain[index] -= (self.maxenergy - self.energy[i])
                    #             self.energy[i] = self.maxenergy
                    #     update_stations.append([index, self.fills_energy_remain[index]])
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

        # for p in power_list:
        #     cv2.circle(img, (p[1] * 5, p[0] * 5), 25, (0, 0, 255))

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
