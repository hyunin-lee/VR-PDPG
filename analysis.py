import numpy as np
import seaborn as sns;
import pandas as pd
import matplotlib.pyplot as plt


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

return_list = []
violations_list = []
return2_list = []
violations2_list = []

for lrTh0 in [0.23,0.24,0.25] :
    for d_0 in [0.001, 0.005, 0.01, 0.05]:
        foldername = "./runs_gridworld8x8_traj1/maxEp__10000__maxS__14__gm__0.9__lrTh0__{}__lrMu0__0.1__a__0.1__mu0__1__d_0__{}".format(lrTh0,d_0)
        returns = np.load(foldername+'/returns.npy')
        violations = np.load(foldername + '/violations.npy')
        returns = moving_average(returns,250)
        violations = moving_average(violations,250)
        return_list.append(returns)
        violations_list.append(violations)

        foldername2 = "./runs_gridworld8x8_traj2/maxEp__10000__maxS__14__gm__0.9__lrTh0__{}__lrMu0__0.1__a__0.1__mu0__1__d_0__{}".format(
            lrTh0, d_0)
        returns2 = np.load(foldername2+'/returns.npy')
        violations2 = np.load(foldername2 + '/violations.npy')
        returns2 = moving_average(returns2,250)
        violations2 = moving_average(violations2,250)
        return2_list.append(returns2)
        violations2_list.append(violations2)

df_r = pd.DataFrame(np.array(return_list))
df_r2 = pd.DataFrame(np.array(return2_list))

mean_r = df_r.mean(axis=0)
std_r = df_r.std(axis=0)
mean_r2 = df_r2.mean(axis=0)
std_r2 = df_r2.std(axis=0)

# Plot
plt.figure(figsize=(8*0.5, 6*0.5),dpi=200)
sns.lineplot(x=range(len(mean_r)), y=mean_r,color='blue',label="traj1")  # Line plot for the mean
plt.fill_between(range(len(mean_r)), mean_r - std_r, mean_r + std_r, color='blue', alpha=0.1)

sns.lineplot(x=range(len(mean_r2)), y=mean_r2,color='red',label="traj2")  # Line plot for the mean
plt.fill_between(range(len(mean_r2)), mean_r2 - std_r2, mean_r2 + std_r2, color='red', alpha=0.1)

plt.tight_layout()
# Enhance Plot
plt.title('Returns')
plt.xlabel('Episode')
plt.ylabel('returns')
plt.tight_layout()
plt.ylim([-5,85])

plt.grid(True)
plt.savefig('./gridworld8x8_returns.pdf')



df_v = pd.DataFrame(np.array(violations_list))
df_v2 = pd.DataFrame(np.array(violations2_list))

mean_v = df_v.mean(axis=0)
std_v = df_v.std(axis=0)
mean_v2 = df_v2.mean(axis=0)
std_v2 = df_v2.std(axis=0)

# Plot
plt.figure(figsize=(8*0.5, 6*0.5),dpi=200)

sns.lineplot(x=range(len(mean_v)), y=mean_v,c='blue',label="traj1")  # Line plot for the mean
plt.fill_between(range(len(mean_v)), mean_v - std_v, mean_v + std_v, color='blue', alpha=0.1)

sns.lineplot(x=range(len(mean_v2)), y=mean_v2,c='red',label="traj2")  # Line plot for the mean
plt.fill_between(range(len(mean_v2)), mean_v2 - std_v2, mean_v2 + std_v2, color='red', alpha=0.1)
plt.tight_layout()
# Enhance Plot
plt.title('Violations')
plt.xlabel('Episode')
plt.ylabel('Violations')
plt.tight_layout()
plt.ylim([-0.1,5.2])

plt.grid(True)
plt.savefig('./gridworld8x8_violations.pdf')
plt.show()