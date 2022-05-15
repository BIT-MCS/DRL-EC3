import time
import os
import matplotlib.pyplot as plt
import numpy as np


class Log(object):
    def __init__(self):
        self.time = str(time.strftime("%Y/%m-%d/%H-%M-%S", time.localtime()))
        self.full_path = os.path.join('.', self.time)
        self.choose_color = ['blue', 'green', 'purple', 'red']
        if os.path.exists(self.full_path):
            self.full_path = os.path.join(self.full_path, '*')
        else:
            pass

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

    def circle(self, x, y, r, color='red', count=100):
        xarr = []
        yarr = []
        for i in range(count):
            j = float(i) / count * 2 * np.pi
            xarr.append(x + r * np.cos(j))
            yarr.append(y + r * np.sin(j))
        plt.plot(xarr, yarr, c=color, linewidth=2)

    def draw_path(self, env, env_i, meaningful_fill, meaningful_get):
        full_path = os.path.join(self.full_path, 'Path')
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        xxx = []
        colors = []
        for x in range(env.mapx):
            xxx.append((x, 1))
        for y in range(env.mapy):
            c = []
            for x in range(env.mapx):
                # 1 represents obstacle,0 is blank
                if env.mapob[x][y] == 1:
                    c.append((0, 0, 0, 1))
                else:
                    c.append((1, 1, 1, 1))
            colors.append(c)

        Fig = plt.figure(figsize=(5, 5))
        PATH = np.array(env.trace)
        ENERGY_PATH = np.array(env.energytrace)

        for i1 in range(env.mapy):
            plt.broken_barh(xxx, (i1, 1), facecolors=colors[i1])

        plt.scatter(env.datas[:, 0], env.datas[:, 1], c=env.DATAs[:, 2], marker="s")

        for i in range(env.n):
            # M = Fig.add_subplot(1, 1, i + 1)
            plt.ylim(ymin=0, ymax=env.mapy)
            plt.xlim(xmin=0, xmax=env.mapx)
            color = self.choose_color[i]
            plt.plot(PATH[i, :, 1], PATH[i, :, 2], color=color)
            for j in range(len(PATH[i])):
                if PATH[i, j, 0] >= 0:
                    plt.scatter(PATH[i, j, 1], PATH[i, j, 2], color=color, marker=".", norm=ENERGY_PATH[i])
                else:
                    plt.scatter(PATH[i, j, 1], PATH[i, j, 2], color=color, marker="+", norm=ENERGY_PATH[i])
            # grid line
            plt.grid(True, linestyle='-.', color='black')
            # title
            plt.title('Meaningful Get:' + str(meaningful_get) + '\nMeaningful Fill:' + str(
                meaningful_fill) + '\nLeft Reward=' + str(env.leftrewards) + '  ( NAIVE VERSION^_^ )')

        plt.scatter(env.fills[:, 0], env.fills[:, 1], c='red', marker="*")
        for (x, y) in zip(env.fills[:, 0], env.fills[:, 1]):
            self.circle(x, y, env.crange)
        Fig.savefig(full_path + '/path_' + str(env_i) + '.png')

        plt.close()

    def step_information(self, action_n, env, step, env_i, meaningful_fill, meaningful_get, indicator):  # -1 fill,1 get
        full_path = os.path.join(self.full_path, 'Path')
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        debug_filename = full_path + '/path_' + str(env_i) + '.txt'

        with open(debug_filename, 'a+') as file:
            print("\nStep ", step, ":", file=file)
            for i in range(env.n):
                if indicator[i] == -1:
                    print("UAV_", i, "------", "Decision: Filling ", env.tmp_energy[i], " energy,current Energy: ",
                          env.energy[i], ",  Reward: ", env.reward[i], ", Penalty: ", env.tmp_penalty[i],
                          "\n\t\tAction detail:", action_n[i], " Station-energy Remain:", env.fills_energy_remain, "\n",
                          file=file)

                    if env.tmp_energy[i] > 0:
                        meaningful_fill[i] += 1
                else:
                    print("UAV_", i, "------", "Decision: Getting ", env.tmp_value[i], " POI,current Energy: ",
                          env.energy[i], ",  Reward: ", env.reward[i], ", Penalty: ", env.tmp_penalty[i],
                          "\n\t\tAction detail:", action_n[i], " Station-energy Remain:", env.fills_energy_remain, "\n",
                          file=file)
                    if env.tmp_value[i] > 0:
                        meaningful_get[i] += 1
