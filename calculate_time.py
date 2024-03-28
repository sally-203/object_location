import numpy as np

data = np.array([[753, 753, 481, 480, 554, 801, 687, 610],
                 [1064, 1064, 968, 774, 1009, 708, 658, 1041],
                 [1276, 1276, 999, 936, 1288, 999, 962, 1059],
                 [715, 715, 703, 863, 973, 865, 809, 975],
                 [350, 350, 487, 524, 1023, 492, 522, 539]])

times = np.mean(data, axis=0)

print("average time:", times)



