import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# training
eff=pd.read_csv("/home/linc/桌面/标杆/run_.-tag-efficiency.csv")
plt.figure(figsize=(17,17))
plt.plot(eff['Step'],eff['Value'],color='black',linewidth=2)
plt.xlim(xmax=130000,xmin=10240)
plt.ylim()
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.xlabel("Training epoch (1e5)",fontsize=32)
plt.ylabel("Energy efficiency",fontsize=32)
plt.grid(True)
plt.grid(linestyle='--')
ax=plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0,1))
plt.show()

rew=pd.read_csv("/home/linc/桌面/标杆/run_.-tag-acc_reward.csv")
plt.figure(figsize=(17,17))
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.plot(rew['Step'],rew['Value'],color='black',linewidth=2)
plt.xlim(xmax=130000,xmin=10240)
plt.xlabel("Training epoch (1e5)",fontsize=32)
plt.ylabel("Accumulated reward",fontsize=32)
plt.grid(True)
plt.grid(linestyle='--')
ax=plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0,1))
plt.show()

loss0=pd.read_csv("/home/linc/桌面/标杆/run_.-tag-loss_0.csv")
loss1=pd.read_csv("/home/linc/桌面/标杆/run_.-tag-loss_1.csv")
plt.figure(figsize=(17,17))
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.plot(loss0['Step'],loss0['Value'],label='Vehicle 1',color='black',linewidth=2)
plt.plot(loss1['Step'],loss1['Value'],label='Vehicle 2',color='blue',linewidth=2)
plt.xlim(xmax=130000,xmin=10240)
plt.ylim(ymax=30,ymin=5)
plt.xlabel("Training epoch (1e5)",fontsize=32)
plt.ylabel("Loss",fontsize=32)
plt.grid(True)
plt.grid(linestyle='--')
plt.legend(fontsize=32)
ax=plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0,1))
plt.show()


# consumption

plt.figure(figsize=(17,17))
num = [1, 2, 3, 4, 5]
sum_comsumption = [3.576, 4, 4.004, 4.402, 4.668]
average_comsumption=np.true_divide(np.array(sum_comsumption),np.array(num))
plt.xticks(num,num[::1])
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.plot(num,sum_comsumption,label='Total energy consumption',marker='o',markersize=26,markeredgewidth=5,markerfacecolor='none',
         color='blue',linewidth=4)
plt.plot(num,average_comsumption,label='Average energy consumption per vehicle',marker='s',markersize=26,markeredgewidth=5,
         markerfacecolor='none',color='black',linewidth=4)

plt.xlabel("No. of vehicles",fontsize=32)
plt.ylabel("Energy usage (# of full batteries)",fontsize=32)
plt.axhline(y=1,color='red',linestyle='--',label="Initial energy reserve",linewidth=4)
plt.grid(True)
plt.grid(linestyle='--')
plt.legend(fontsize=26)
plt.show()
#
# # charge amount
# num = [1, 2, 3, 4, 5]
# sum_charge_amount = [166.317, 204.866, 201.16, 196.783, 192.793]
# plt.xticks(num,num[::1])
# plt.plot(num,sum_charge_amount)
# plt.xlabel("Num of vehicles")
# plt.ylabel("Total charge amount")
# plt.axhline(y=250,color='green',linestyle='--')
# plt.grid(True)
# plt.grid(linestyle='--')
# plt.legend()
# plt.show()
#
# # charge frequency
# num = [1, 2, 3, 4, 5]
# sum_charge_frequency = [49, 232, 227, 512, 514]
# plt.xticks(num,num[::1])
# plt.plot(num,sum_charge_frequency)
# plt.xlabel("Num of vehicles")
# plt.ylabel("Total charge frequency")
# plt.grid(True)
# plt.grid(linestyle='--')
# plt.legend()
# plt.show()


# yy
num = [1, 2, 3, 4, 5]
plt.figure(figsize=(17,17))
sum_charge_amount = [166.317/50, 204.866/50, 201.16/50, 196.783/50, 192.793/50]
sum_charge_frequency = [49, 232, 227, 512, 514]
plt.xlabel("No. of vehicles",fontsize=32)
plt.plot(num,sum_charge_amount,label='Total # of charged full battery',color='red',marker='o',
         markersize=26,markeredgewidth=5,markerfacecolor='none',linewidth=4)
plt.ylim(ymax=4.4,ymin=3)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.ylabel("Total # of charged full battery",fontsize=32)
plt.legend(loc='upper left',fontsize=26)
plt.grid(True)
plt.grid(linestyle='--')

plt.twinx()
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.plot(num,sum_charge_frequency,label='Total charging frequency',color='blue',marker='s',
         markersize=26,markeredgewidth=5,markerfacecolor='none',linewidth=4)
plt.ylim(ymax=700,ymin=0)
plt.ylabel("Total charging frequency",fontsize=32)
plt.legend(loc='lower right',fontsize=26)

plt.xticks(num,num[::1])

plt.grid(True)
plt.grid(linestyle='--')
plt.show()

