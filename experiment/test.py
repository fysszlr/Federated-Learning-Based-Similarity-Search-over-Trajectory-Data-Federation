import pickle

with open('../data/traj0distort_ratio0.1.pkl', 'rb') as file:
    data1 = pickle.load(file)
with open('../data/traj0distort_ratio0.2.pkl', 'rb') as file:
    data3 = pickle.load(file)

for (a, b) in zip(data1, data3):
    print(a)
    print(b)