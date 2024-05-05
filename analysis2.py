import numpy as np
import seaborn as sns;
import pandas as pd
import matplotlib.pyplot as plt


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

return_list = []
violations_list = []
return_b1_list = []
violations_b1_list = []
return2_list = []
violations2_list = []
return2_b1_list = []
violations2_b1_list = []


lrMu0_list_traj1 = [0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.04, 0.05, 0.05]
a_list_traj1 = [0.05, 0.05, 0.1, 0.1, 0.05, 0.1, 0.05, 0.05, 0.1, 0.05, 0.1 ]
d_0_list_traj1 = [0.005, 0.01, 0.005, 0.01, 0.005, 0.005, 0.005, 0.01, 0.005, 0.005, 0.01]

lrMu0_list_traj2 = [0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03]
a_list_traj2 = [0.05, 0.05, 0.09, 0.09, 0.03, 0.07, 0.07, 0.09, 0.03, 0.03, 0.09 ]
d_0_list_traj2 = [0.001, 0.005, 0.001, 0.005, 0.001, 0.001, 0.005, 0.001, 0.001, 0.005, 0.001]


## for trajectory 1
# for lrMu0,a,d_0 in zip(lrMu0_list_traj1,a_list_traj1,d_0_list_traj1) :
#     foldername = "./runs_gridworld20x20_traj1/maxEp__15000__maxS__38__gm__0.9__lrTh0__0.23__lrMu0__{}__a__{}__mu0__1__d_0__{}".format(
#         lrMu0, a, d_0)
#     returns = np.load(foldername + '/returns.npy')
#     violations = np.load(foldername + '/violations.npy')
#     returns = moving_average(returns, 250)
#     violations = moving_average(violations, 250)
#     returns = returns[:10000]
#     violations = violations[:10000]
#     if len(returns) !=10000 :
#         returns = np.append(returns, [returns[-1]] * (10000 - len(returns)))
#         violations = np.append(violations, [violations[-1]] * (10000 - len(violations)))
#     print(len(violations))
#     return_list.append(returns)
#     violations_list.append(violations)

## for trajectory 2
for lrMu0,a,d_0 in zip(lrMu0_list_traj2,a_list_traj2,d_0_list_traj2) :
    foldername2 = "./runs_gridworld20x20_traj2/maxEp__15000__maxS__38__gm__0.9__lrTh0__0.23__lrMu0__{}__a__{}__mu0__1__d_0__{}".format(
        lrMu0, a, d_0)
    returns2 = np.load(foldername2 + '/returns.npy')
    violations2 = np.load(foldername2 + '/violations.npy')
    returns2 = moving_average(returns2, 250)
    violations2 = moving_average(violations2, 250)
    returns2 = returns2[:10000]
    violations2 = violations2[:10000]
    if len(returns2) !=10000 :
        returns2 = np.append(returns2, [returns2[-1]] * (10000 - len(returns2)))
        violations2 = np.append(violations2, [violations2[-1]] * (10000 - len(violations2)))
    print(len(violations2))
    return2_list.append(returns2)
    violations2_list.append(violations2)

## for baseline for trajectory 1
# for lrMu0,d_0 in zip(lrMu0_list_traj1,d_0_list_traj1) :
#     foldername = "./runs_gridworld20x20_traj1/maxEp__15000__maxS__38__gm__0.9__lrTh0__0.23__lrMu0__{}__a__1.0__mu0__1__d_0__{}".format(
#         lrMu0, d_0)
#     returns_b1 = np.load(foldername + '/returns.npy')
#     violations_b1 = np.load(foldername + '/violations.npy')
#     returns_b1 = moving_average(returns_b1, 250)
#     violations_b1 = moving_average(violations_b1, 250)
#     returns_b1 = returns_b1[:10000]
#     violations_b1 = violations_b1[:10000]
#     if len(returns_b1) !=10000 :
#         returns_b1 = np.append(returns_b1, [returns_b1[-1]] * (10000 - len(returns_b1)))
#         violations_b1 = np.append(violations_b1, [violations_b1[-1]] * (10000 - len(violations_b1)))
#     print(len(violations_b1))
#     return_b1_list.append(returns_b1)
#     violations_b1_list.append(violations_b1)

## for baseline for trajectory 2
for lrMu0,d_0 in zip(lrMu0_list_traj2,d_0_list_traj2) :
    foldername = "./runs_gridworld20x20_traj2/maxEp__15000__maxS__38__gm__0.9__lrTh0__0.23__lrMu0__{}__a__1.0__mu0__1__d_0__{}".format(
        lrMu0, d_0)
    returns2_b1 = np.load(foldername + '/returns.npy')
    violations2_b1 = np.load(foldername + '/violations.npy')
    returns2_b1 = moving_average(returns2_b1, 250)
    violations2_b1 = moving_average(violations2_b1, 250)
    returns2_b1 = returns2_b1[:10000]
    violations2_b1 = violations2_b1[:10000]
    if len(returns2_b1) !=10000 :
        returns2_b1 = np.append(returns2_b1, [returns2_b1[-1]] * (10000 - len(returns2_b1)))
        violations2_b1 = np.append(violations2_b1, [violations2_b1[-1]] * (10000 - len(violations2_b1)))
    print(len(violations2_b1))
    return2_b1_list.append(returns2_b1)
    violations2_b1_list.append(violations2_b1)

c_list = ['#007BFF' , '#89CFF0', '#FF0000', '#FFC0CB']


# df_r = pd.DataFrame(np.array(return_list))
# df_r_b1 = pd.DataFrame(np.array(return_b1_list))
df_r2 = pd.DataFrame(np.array(return2_list))
df_r2_b1 = pd.DataFrame(np.array(return2_b1_list))

# mean_r = df_r.mean(axis=0)
# std_r = df_r.std(axis=0)
# mean_r_b1 = df_r_b1.mean(axis=0)
# std_r_b1 = df_r_b1.std(axis=0)
mean_r2 = df_r2.mean(axis=0)
std_r2 = df_r2.std(axis=0)
mean_r2_b1 = df_r2_b1.mean(axis=0)
std_r2_b1 = df_r2_b1.std(axis=0)

# Plot
plt.figure(figsize=(8*0.55, 6*0.55),dpi=200)
# sns.lineplot(x=range(len(mean_r)), y=mean_r,color=c_list[0],label="traj1")  # Line plot for the mean
# plt.fill_between(range(len(mean_r)), mean_r - std_r, mean_r + std_r, color=c_list[0], alpha=0.1)
#
# sns.lineplot(x=range(len(mean_r_b1)), y=mean_r_b1,color=c_list[1],label="traj1 ($\\alpha_t=1$)", linestyle='dashdot')  # Line plot for the mean
# plt.fill_between(range(len(mean_r_b1)), mean_r_b1 - std_r_b1, mean_r_b1 + std_r_b1, color=c_list[1], alpha=0.1)

sns.lineplot(x=range(len(mean_r2)), y=mean_r2,color=c_list[2],label="traj2")  # Line plot for the mean
plt.fill_between(range(len(mean_r2)), mean_r2 - std_r2, mean_r2 + std_r2, color=c_list[2], alpha=0.1)

sns.lineplot(x=range(len(mean_r2_b1)), y=mean_r2_b1,color=c_list[3],label="traj2 ($\\alpha_t=1$)", linestyle='dashdot')  # Line plot for the mean
plt.fill_between(range(len(mean_r2_b1)), mean_r2_b1 - std_r2_b1, mean_r2_b1 + std_r2_b1, color=c_list[3], alpha=0.1)

#'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'

plt.tight_layout()
# Enhance Plot
plt.title('Returns')
plt.xlabel('Episode')
plt.ylabel('returns')
plt.tight_layout()
plt.ylim([-10,85])

plt.grid(True)
plt.savefig('./gridworld20x20_returns2.pdf')



# df_v = pd.DataFrame(np.array(violations_list))
# df_v_b1 = pd.DataFrame(np.array(violations_b1_list))
df_v2 = pd.DataFrame(np.array(violations2_list))
df_v2_b1 = pd.DataFrame(np.array(violations2_b1_list))

# mean_v = df_v.mean(axis=0)
# std_v = df_v.std(axis=0)
# mean_v_b1 = df_v_b1.mean(axis=0)
# std_v_b1 = df_v_b1.std(axis=0)
mean_v2 = df_v2.mean(axis=0)
std_v2 = df_v2.std(axis=0)
mean_v2_b1 = df_v2_b1.mean(axis=0)
std_v2_b1 = df_v2_b1.std(axis=0)

# Plot
plt.figure(figsize=(8*0.55, 6*0.55),dpi=200)

# sns.lineplot(x=range(len(mean_v)), y=mean_v,c=c_list[0],label="traj1")  # Line plot for the mean
# plt.fill_between(range(len(mean_v)), mean_v - std_v, mean_v + std_v, color=c_list[0], alpha=0.1)
#
# sns.lineplot(x=range(len(mean_v_b1)), y=mean_v_b1,c=c_list[1],label="traj1 ($\\alpha_t=1$)", linestyle='dashdot')  # Line plot for the mean
# plt.fill_between(range(len(mean_v_b1)), mean_v_b1 - std_v_b1, mean_v_b1 + std_v_b1, color=c_list[1], alpha=0.1)

sns.lineplot(x=range(len(mean_v2)), y=mean_v2,c=c_list[2],label="traj2")  # Line plot for the mean
plt.fill_between(range(len(mean_v2)), mean_v2 - std_v2, mean_v2 + std_v2, color=c_list[2], alpha=0.1)

sns.lineplot(x=range(len(mean_v2_b1)), y=mean_v2_b1,c=c_list[3],label="traj2 ($\\alpha_t=1$)", linestyle='dashdot')  # Line plot for the mean
plt.fill_between(range(len(mean_v2_b1)), mean_v2_b1 - std_v2_b1, mean_v2_b1 + std_v2_b1, color=c_list[3], alpha=0.1)


plt.tight_layout()
# Enhance Plot
plt.title('Violations')
plt.xlabel('Episode')
plt.ylabel('Violations')
plt.tight_layout()
plt.ylim([-0.1,6.5])

plt.grid(True)
plt.savefig('./gridworld20x20_violations2.pdf')
plt.show()