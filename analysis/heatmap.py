import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import torch

# matplotlib.use('pgf')
# With LaTex fonts
# plt.style.use('tex')
# plt.style.use('seaborn')

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     # "pgf.texsystem": "pdflatex",
#     'font.family': 'SimSun',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
#     # Use LaTeX to write all text
#     "text.usetex": True,
#     # Use 10pt font in plots, to match 10pt font in document
#     "axes.labelsize": 10,
#     "font.size": 10,
#     # Make the legend/label fonts a little smaller
#     "legend.fontsize": 8,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8
# })

plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.rc('font', family='SimSun', size=12)

# ('寇克斯', '报告', '的', '震撼')
chars = ['$', '寇', '克', '斯', '报', '告', '的', '震', '撼']

# coarse2fine score
data_value = [[-10, -1.1082e+01, -7.9090e+00, -1.0474e+01, -8.0487e+00,
               -3.8741e+00, -1.1384e+01, -5.4462e+00, -6.6131e-01],
              [-10,  1.2072e+01,  2.5146e+01,  1.7288e+01,  6.5055e+00,
               7.0862e+00, -9.1275e+00,  1.4733e+01,  8.4644e+00],
              [-10,  4.6870e+00,  1.2771e+01,  2.5280e+01,  2.6292e+00,
               8.0154e+00, -6.8049e+00,  4.8694e+00,  6.3692e+00],
              [-10, -6.6123e+00, -4.5169e+00,  3.6029e+00, -9.7404e+00,
               -4.7080e+00, -1.3640e+01, -1.2203e+01, -5.1361e+00],
              [-10, -6.4834e+00,  3.1395e+00, -9.9469e+00,  1.0478e+01,
               1.7798e+01, -2.4870e+01,  3.3053e+00,  1.2773e+01],
              [-10, -1.2513e+01, -9.9434e+00, -6.4002e+00, -5.5455e+00,
               -3.3780e-02, -1.6477e+01, -9.9524e+00, -1.0586e-01],
              [-10, -1.3430e+01, -1.6400e+01, -1.3043e+01, -1.3483e+01,
               -1.1816e+01, -7.7058e+00, -2.2929e+01, -1.0637e+01],
              [-10, -1.5120e+01, -2.1581e-02, -1.9397e+01, -3.0245e-01,
               5.3951e+00, -3.0191e+01,  1.1551e+01,  1.5042e+01],
              [-10, -2.6301e+01, -2.6006e+01, -2.7224e+01, -1.9845e+01,
               -1.3616e+01, -2.4279e+01, -1.8313e+01, -5.4595e+00]]

# # coarse2fine score softmax
# data_value = [[0.0000e+00, 2.8386e-05, 6.7789e-04, 5.2154e-05, 5.8950e-04, 3.8328e-02,
#                2.0990e-05, 7.9567e-03, 9.5235e-01],
#               [0.0000e+00, 0.0000e+00, 9.9958e-01, 3.8658e-04, 8.0237e-09, 1.4340e-08,
#                1.3033e-15, 3.0016e-05, 5.6899e-08],
#               [0.0000e+00, 1.1391e-09, 0.0000e+00, 1.0000e+00, 1.4550e-10, 3.1774e-08,
#                1.1632e-14, 1.3670e-09, 6.1253e-09],
#               [0.0000e+00, 4.9337e-02, 4.0105e-01, 0.0000e+00, 2.1611e-03, 3.3129e-01,
#                4.3740e-05, 1.8407e-04, 2.1593e-01],
#               [0.0000e+00, 2.8313e-11, 4.2772e-07, 8.8679e-13, 0.0000e+00, 9.9347e-01,
#                2.9281e-19, 5.0487e-07, 6.5306e-03],
#               [0.0000e+00, 4.0622e-06, 5.3075e-05, 1.8352e-03, 4.3141e-03, 0.0000e+00,
#                7.7180e-08, 5.2599e-05, 9.9374e-01],
#               [0.0000e+00, 4.0287e-02, 2.0663e-03, 5.9336e-02, 3.8204e-02, 2.0234e-01,
#                0.0000e+00, 3.0181e-06, 6.5777e-01],
#               [0.0000e+00, 7.9578e-14, 2.8714e-07, 1.1047e-15, 2.1683e-07, 6.4644e-05,
#                2.2673e-20, 0.0000e+00, 9.9993e-01],
#               [0.0000e+00, 3.0631e-06, 4.1163e-06, 1.2173e-06, 1.9505e-03, 9.8899e-01,
#                2.3141e-05, 9.0284e-03, 0.0000e+00]]

# coarse2fine
data_value = [[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
               0.0000e+00, 0.0000e+00, 0.0000e+00],
              [0.0000e+00, 0.0000e+00, 9.9962e-01, 3.8660e-04, 1.4764e-16, 1.9613e-15,
               6.1224e-36, 9.4685e-28, 2.8917e-26],
              [0.0000e+00, 4.4037e-13, 0.0000e+00, 1.0000e+00, 2.0953e-17, 8.3351e-15,
               4.5916e-34, 2.8387e-31, 4.9484e-27],
              [0.0000e+00, 3.9423e-16, 4.5719e-15, 0.0000e+00, 7.7212e-10, 1.1836e-07,
               4.0895e-24, 8.9388e-26, 1.8415e-19],
              [0.0000e+00, 7.5197e-27, 1.9699e-21, 8.9262e-13, 0.0000e+00, 1.0000e+00,
               8.6483e-32, 3.5760e-28, 3.7874e-21],
              [0.0000e+00, 1.1282e-29, 2.2917e-31, 1.2482e-13, 7.0125e-20, 0.0000e+00,
               5.2494e-18, 1.0679e-23, 8.5745e-19],
              [0.0000e+00, 1.1052e-28, 1.1866e-32, 4.8071e-23, 7.2729e-25, 1.5715e-10,
               0.0000e+00, 4.6879e-15, 1.0216e-09],
              [0.0000e+00, 1.4293e-43, 3.6526e-37, 3.6275e-37, 1.0034e-30, 2.0317e-14,
               2.2674e-20, 0.0000e+00, 1.0000e+00],
              [0.0000e+00, 0.0000e+00, 0.0000e+00, 3.1961e-30, 2.4591e-37, 1.1835e-19,
               2.0729e-15, 5.2508e-25, 0.0000e+00]]

# # constraint decoding
# data_value = [[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#                0.0000e+00, 0.0000e+00, 0.0000e+00],
#               [0.0000e+00, 0.0000e+00, 1.0000e+00, 4.1491e-06, 2.4489e-23, 8.5258e-30,
#                0.0000e+00, 0.0000e+00, 0.0000e+00],
#               [0.0000e+00, 3.2506e-14, 0.0000e+00, 1.0000e+00, 4.1801e-16, 1.2947e-21,
#                0.0000e+00, 0.0000e+00, 0.0000e+00],
#               [0.0000e+00, 2.4487e-23, 4.1801e-16, 0.0000e+00, 1.0000e+00, 7.1737e-07,
#                0.0000e+00, 0.0000e+00, 0.0000e+00],
#               [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 7.1737e-07,
#                0.0000e+00, 8.7281e-16, 1.0000e+00],
#               [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
#                0.0000e+00, 2.3020e-20, 7.1737e-07],
#               [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 7.1737e-07,
#                0.0000e+00, 0.0000e+00, 0.0000e+00],
#               [8.7284e-16, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#                0.0000e+00, 0.0000e+00, 1.0000e+00],
#               [1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#                0.0000e+00, 8.7284e-16, 0.0000e+00]]

# # vanilla decoding
# data_value = [[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#                0.0000e+00, 0.0000e+00, 0.0000e+00],
#               [5.4089e-29, 0.0000e+00, 1.0000e+00, 4.1491e-06, 7.2374e-14, 3.7729e-20,
#                2.4136e-32, 2.6053e-27, 2.4994e-18],
#               [5.0944e-28, 3.2506e-14, 0.0000e+00, 1.0000e+00, 1.4524e-10, 6.7356e-16,
#                1.1831e-29, 7.9337e-30, 1.4263e-16],
#               [8.7074e-18, 2.4488e-23, 4.1801e-16, 0.0000e+00, 9.9999e-01, 7.1772e-07,
#                1.8911e-25, 8.6856e-23, 4.8382e-06],
#               [1.2257e-06, 1.4021e-31, 7.5647e-27, 1.9898e-13, 0.0000e+00, 2.1630e-06,
#                2.6720e-16, 2.8279e-09, 1.0000e+00],
#               [2.6591e-14, 2.9254e-38, 1.8358e-34, 4.3060e-20, 9.9999e-01, 0.0000e+00,
#                3.6498e-13, 1.0274e-12, 9.8893e-06],
#               [2.0994e-21, 1.4508e-38, 1.7254e-33, 7.5474e-16, 9.9970e-01, 2.9551e-04,
#                0.0000e+00, 4.5550e-14, 2.3337e-07],
#               [8.7287e-16, 4.2039e-45, 9.0398e-42, 1.3252e-34, 1.0704e-09, 2.2810e-11,
#                9.5542e-13, 0.0000e+00, 1.0000e+00],
#               [1.0000e+00, 5.3954e-29, 3.0493e-28, 1.1738e-19, 1.2257e-06, 6.6350e-12,
#                2.2414e-11, 9.9708e-16, 0.0000e+00]]

# fig = plt.figure()
# fig.set_size_inches(w=4.9823, h=.5)

"""
1	金	_	NN	NN	_	2	nn	_	_
2	杯子	_	NN	NN	_	4	assmod	_	_
3	的	_	DEG	DEG	_	2	assm	_	_
4	白开水	_	NN	NN	_	0	root	_	_
"""
# ('金', '杯子', '的', '白开水')
chars = ['$', '金', '杯', '子', '的', '白', '开', '水']
# epoch 1
data_value = [[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 6.4791e-02, 3.1156e-07, 2.8304e-11, 2.4076e-08,
         1.5385e-08, 5.3020e-09],
        [0.0000e+00, 6.0933e-05, 0.0000e+00, 2.3751e-06, 9.1485e-10, 8.0542e-08,
         4.3438e-07, 7.7798e-08],
        [0.0000e+00, 2.4056e-07, 9.9930e-01, 0.0000e+00, 2.5887e-06, 2.9179e-07,
         4.9010e-06, 3.6822e-07],
        [0.0000e+00, 5.7874e-07, 2.5201e-04, 1.5363e-04, 0.0000e+00, 4.3679e-05,
         8.9367e-05, 1.6637e-05],
        [0.0000e+00, 1.3140e-07, 5.0866e-05, 1.7213e-05, 1.3677e-04, 0.0000e+00,
         2.2309e-01, 1.4286e-02],
        [0.0000e+00, 3.8493e-08, 2.2271e-05, 5.3998e-06, 1.5019e-05, 1.0110e-01,
         0.0000e+00, 1.1245e-01],
        [0.0000e+00, 5.7994e-08, 2.2299e-05, 4.2658e-06, 6.4396e-06, 1.1103e-01,
         8.0964e-01, 0.0000e+00]]

# # epoch 2
# data_value = [[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 9.2895e-03, 2.5004e-09, 1.1028e-15, 2.9158e-14,
#          3.9276e-13, 1.4174e-12],
#         [0.0000e+00, 3.3197e-08, 0.0000e+00, 4.3410e-08, 3.8720e-12, 1.1380e-12,
#          1.9374e-10, 2.9519e-10],
#         [0.0000e+00, 4.0932e-11, 9.9996e-01, 0.0000e+00, 2.0149e-10, 3.4035e-13,
#          7.3742e-10, 2.0650e-10],
#         [0.0000e+00, 1.1756e-11, 1.5885e-07, 2.0847e-08, 0.0000e+00, 1.1917e-08,
#          2.6645e-07, 3.0471e-07],
#         [0.0000e+00, 6.2703e-15, 5.1827e-10, 1.4334e-10, 5.9253e-06, 0.0000e+00,
#          5.8071e-01, 1.5037e-01],
#         [0.0000e+00, 1.5765e-14, 4.0809e-10, 2.8788e-10, 1.8468e-06, 5.3066e-02,
#          0.0000e+00, 2.6757e-01],
#         [0.0000e+00, 1.3747e-13, 2.4556e-09, 4.4364e-10, 2.7822e-07, 5.4116e-04,
#          6.1361e-01, 0.0000e+00]]

# data_value = [[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 7.9987e-03, 7.7668e-07, 4.6625e-13, 4.0091e-12,
#          4.9151e-11, 1.7633e-10],
#         [0.0000e+00, 4.4483e-06, 0.0000e+00, 1.2762e-05, 3.3250e-10, 4.0813e-11,
#          7.2309e-09, 1.4144e-08],
#         [0.0000e+00, 6.5145e-09, 9.9992e-01, 0.0000e+00, 3.9642e-10, 5.2553e-13,
#          3.6757e-09, 2.6800e-09],
#         [0.0000e+00, 1.3006e-09, 1.0297e-06, 3.0210e-08, 0.0000e+00, 2.4671e-07,
#          1.7697e-06, 2.0926e-06],
#         [0.0000e+00, 1.1295e-13, 4.9057e-10, 1.5010e-09, 7.2427e-08, 0.0000e+00,
#          4.6903e-01, 4.5820e-01],
#         [0.0000e+00, 3.1782e-12, 1.6713e-08, 9.1928e-08, 2.1663e-07, 9.4754e-03,
#          0.0000e+00, 8.2302e-01],
#         [0.0000e+00, 2.8652e-11, 1.5233e-07, 2.9291e-07, 1.6636e-06, 9.3406e-05,
#          1.4787e-01, 0.0000e+00]]

# data_value = torch.tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 4.0093e-13, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          8.4022e-01, 1.5962e-01],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.1982e-03,
#          0.0000e+00, 3.6945e-01],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 9.9936e-05,
#          6.2942e-01, 0.0000e+00]])

# data_value = data_value.tolist()


data_value = torch.tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 7.8441e-10, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         7.8541e-01, 2.1431e-01],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 3.2838e-03,
         0.0000e+00, 3.5289e-01],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.0242e-04,
         6.4391e-01, 0.0000e+00]]) + \
        torch.tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00],
        [1.0408e-08, 0.0000e+00, 9.9971e-01, 7.7606e-10, 1.2595e-14, 2.7402e-10,
         1.5595e-04, 1.3893e-04],
        [1.2086e-05, 2.9836e-08, 0.0000e+00, 0.0000e+00, 3.6024e-10, 2.7577e-04,
         6.4382e-01, 3.5589e-01],
        [1.2985e-12, 5.4627e-15, 0.0000e+00, 0.0000e+00, 3.2364e-18, 3.7791e-13,
         3.6117e-10, 4.2155e-10],
        [5.6019e-11, 4.5951e-11, 1.0000e+00, 7.8411e-10, 0.0000e+00, 1.6587e-11,
         1.5224e-06, 4.6456e-06],
        [2.7577e-04, 1.8110e-10, 9.9478e-06, 8.6139e-13, 1.7410e-07, 0.0000e+00,
         0.0000e+00, 0.0000e+00],
        [6.4382e-01, 8.2109e-09, 1.4719e-06, 2.6571e-13, 2.1846e-08, 0.0000e+00,
         0.0000e+00, 0.0000e+00],
        [3.5589e-01, 1.9240e-09, 4.6677e-07, 1.6917e-13, 4.0028e-09, 0.0000e+00,
         0.0000e+00, 0.0000e+00]])


print(data_value < 1)
data_value = data_value[1:, 1:].tolist()
for i in range(len(data_value)):
    for j in range(len(data_value[i])):
        # print(j, i, f'{data_value[i][j]:.2f}')
        print(j, i, f'{data_value[j][i]:.2f}')


# data = pd.DataFrame(data_value, index=chars, columns=chars)
# # sns.heatmap(data, vmax=1, vmin=0, annot=True, fmt='.2f', cmap='Blues')
# axs = sns.heatmap(data, square=True, fmt='.2f', cmap='RdBu_r', center=0)

# plt.savefig('analysis/heatmap-c2f-score.png')
# plt.savefig('analysis/heatmap-c2f-10.png')
# plt.show()