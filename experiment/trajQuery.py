# import pickle
# with open('/home/zlr/t2vec/experimentData/traj0distort_ratio0.pkl', 'rb') as file:
#     data = pickle.load(file)
#     with open('/home/zlr/t2vec/experimentData/feature_trajs','w') as wfile:
#         for line in data:
#             for it in line:
#                 wfile.write(str(it))
#                 wfile.write(' ')
#             wfile.write('\n')
import matplotlib.pyplot as plt
import numpy as np

plt.xlim(-8.77,-8.49)
plt.ylim(41.03,41.38)
a = np.zeros(10)
wfile = open('/home/zlr/t2vec/experimentData/traj0_origin_length', 'w')
with open('/home/zlr/t2vec/experimentData/origin_trajs','r') as file:
    file.readline()
    total_num = 1000
    for i in range(total_num):
        traj = file.readline()
        while(traj=="--\n"):
            traj = file.readline()
        # print(traj)
        num = file.readline()
        # print(num)
        num = (int)(num)
        wfile.write(str(num)+'\n')
        for idx,j in enumerate(range(90,-10,-10)):
            if (num >= j):
                a[9-idx]+=1
                break
        x = np.fromstring(file.readline(), dtype=float, sep=' ')
        y = np.fromstring(file.readline(), dtype=float, sep=' ')
        # plt.scatter(x,y,s=1, c='b')
# plt.title(f'total {total_num} trajs')
# plt.show()
print(a)