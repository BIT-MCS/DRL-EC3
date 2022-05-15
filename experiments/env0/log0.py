import time
import os
import matplotlib.pyplot as plt
import numpy as np

class Log(object):
    def __init__(self):
        self.time = str(time.strftime("%Y/%m-%d/%H-%M-%S", time.localtime()))
        self.full_path = os.path.join('.', self.time)
        os.makedirs(self.full_path)
        self.file_path = self.full_path + '/REPORT.txt'
        file = open(self.file_path, 'x')
        file.close()

    def log(self, values):
        if isinstance(values, dict):
            with open(self.file_path, 'a') as file:
                for key, value in values.items():
                    print(key, value, file=file)
        elif isinstance(values, list):
            with open(self.file_path, 'a') as file:
                for value in values:
                    print(value, file=file)
        else:
            with open(self.file_path, 'a') as file:
                print(values, file=file)

    def draw_path(self, env, step):
        full_path = os.path.join(self.full_path, 'Path')
        # ob_xy = np.zeros((FLAGS.map_x, FLAGS.map_y))
        # for i in FLAGS.obstacle:
        #     for x in range(i[0], i[0] + i[2], 1):
        #         for y in range(i[1], i[1] + i[3], 1):
        #             ob_xy[x][y] = 1
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        xxx = []
        colors = []
        for x in range(env.mapx):
            xxx.append((x, 1))
        for y in range(env.mapy):
            c = []
            for x in range(env.mapx):
                if env.mapob[x][y] == 1:
                    c.append((1, 0, 0, 1))
                else:
                    c.append((1, 1, 1, 1))
            colors.append(c)

        Fig = plt.figure(figsize=(5, 5))
        PATH = np.array(env.trace)
        for i1 in range(env.mapy):
            plt.broken_barh(xxx, (i1, 1), facecolors=colors[i1])
        plt.scatter(env.datas[:,0], env.datas[:,1], c=env.DATAs[:,2])
        for i in range(env.n):
            # M = Fig.add_subplot(1, 1, i + 1)
            plt.ylim(ymin=0, ymax=env.mapy)
            plt.xlim(xmin=0, xmax=env.mapx)
            color = np.random.random(3)
            plt.plot(PATH[i, :, 0], PATH[i, :, 1], color=color)
            plt.scatter(PATH[i, :, 0], PATH[i, :, 1], color=color,marker='.')
            plt.grid(True, linestyle='-.', color='r')
            plt.title(str(env.normal_energy) + ',\n' + str(env.leftrewards))
        Fig.savefig(full_path + '/path_' + str(step) + '.png')

        plt.close()
