import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib import rc

plt.rcParams.update({'font.size': 18})

def make_chart(X, RE, C, label_map, col, alpha, save_name, all_labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cur_labels = [label_map[x] for x in C]
    pos = 0
    x_labels = []
    x_colors = []
    x_alphas = []

    for idx in range(len(all_labels)):
        if not (all_labels[idx] in cur_labels):
            continue

        c = col[idx]
        alp = alpha[idx]
        my_position = pos

        data_idx = cur_labels.index(all_labels[idx]) 
        my_data = X[:, data_idx]
        err_mean = RE[:, data_idx].mean()

        x_labels.append(all_labels[idx])
        x_colors.append(c)
        x_alphas.append(alp)

        ax.boxplot(my_data, positions=[pos], widths=0.5, 
                notch=True, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=c, color=c, alpha=alp),
                capprops=dict(color=c, alpha=alp),
                whiskerprops=dict(color=c , alpha=alp),
                flierprops=dict(color=c, markeredgecolor=c, alpha=alp),
                medianprops=dict(color=c , alpha=alp))
        ax.scatter(pos, err_mean, c='r', marker='x', s=24)
        pos += 1

    ax.set_ylim([0., 0.5])
    ax.set_xticks(np.arange(X.shape[1]))
    ax.set_xticklabels(x_labels)

    ax.set_yticks([0., 0.25, 0.55])
    ax.set_yticklabels([0., 0.25, 0.55])

    if 'sheep_0' in save_name:
        ax.set_ylabel('L1 Error', fontsize=24)

    for xtick, color, alp in zip(ax.get_xticklabels(), x_colors, x_alphas):
        xtick.set_color(color)
        xtick.set_alpha(alp)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('{}_L1Errs.png'.format(save_name), bbox_inches='tight')
    plt.show()
    
Xs = []

Xs.append(pickle.load(open('/media/data_cifs/projects/prj_deepspine/hd64/ckpts/StimToEMG/xyencV0_l1_errors.p', 'rb')))

label_map = {
2: 'L-PL', 
3: 'L-GA', 
5: 'L-BF', 
4: 'L-GR', 
6: 'R-PL', 
7: 'R-GA', 
10: 'R-BF',
8: 'R-GR',
}
all_labels = ['L-PL', 'L-GA', 'L-GR', 'L-BF', 'R-PL', 'R-GA', 'R-GR', 'R-BF']

alpha, col = [], []
for jj in range(8):
    if jj < 4:
        col.append(plt.get_cmap('Dark2')(jj))
        alpha.append(1.)
    else:
        col.append(plt.get_cmap('Dark2')(jj-4))
        alpha.append(0.5)

for idx, dataset in enumerate(Xs):
    make_chart(dataset['err'], dataset['randerr'], dataset['channels'], label_map, col, alpha, 'sheep_{}'.format(idx), all_labels)
