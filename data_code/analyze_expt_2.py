'''
analysis script for expt 2
'''

run_process = True

# import modules
import pathlib
from itertools import combinations
from scipy.stats import sem, ttest_rel, ttest_1samp
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import numpy as np
import pandas as pd
import pingouin as pg
import sys
import warnings

# use R library
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
afex = importr('afex')
base = importr('base')

from function import data_func

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")
rcParams['font.family'] = 'CMU Sans Serif'


def manage_path():
    # input
    current_path = pathlib.Path(__file__).parent.absolute()
    data_path = current_path / 'data/expt_2.csv'

    # output
    result_path = current_path / 'results' / 'expt_2'
    pro_data_path = current_path / 'results/expt_2'
    out_stat_path = current_path / 'results/expt_2/stat_result.txt'
    out_graph_path = current_path / 'results/expt_2/graph'

    result_path.mkdir(parents=True, exist_ok=True)
    out_graph_path.mkdir(parents=True, exist_ok=True)

    return data_path, pro_data_path, out_stat_path, out_graph_path


def read(datafile):
    data = pd.read_csv(datafile, delimiter = ',')
    return data


def process(data):
    n_subjs = len(pd.unique(data.subject_ID))
    n_conds = len(pd.unique(data.condition))
    n_columns = 12
    pro_data = np.empty(shape = (n_subjs, n_conds, n_columns))

    for subj in range(n_subjs):
        subj_data = data[data.subject_ID == subj + 1]
        for cond in range(n_conds):
            subj_cond_data = subj_data[data.condition == cond]

            subj_no = subj + 1
            stim_cond = cond
            percept_acc = subj_cond_data.correct.sum() / len(subj_cond_data)
            d_prime, c, beta = data_func.compute_sdt(subj_cond_data.answer, subj_cond_data.percept_resp)
            conf_mean = subj_cond_data.conf_resp.mean()
            c_conf_mean = subj_cond_data[subj_cond_data.correct == 1].conf_resp.mean()
            ic_conf_mean = subj_cond_data[subj_cond_data.correct == 0].conf_resp.mean()
            percept_rt = subj_cond_data.percept_rt.mean()
            conf_rt = subj_cond_data.conf_rt.mean()
            threshold = subj_data[subj_data.condition == 15].stim_tilt_diff.unique()[0]

            # package output
            pro_data[subj, cond, :] = (subj_no, stim_cond, percept_acc,
                                       d_prime, c, beta, conf_mean,
                                       c_conf_mean, ic_conf_mean,
                                       percept_rt, conf_rt,
                                       threshold
                                       )

    # convert to pd.DataFrame
    pro_data = pro_data.reshape(n_subjs * n_conds, n_columns)
    column_name = ['subj_no', 'stim_cond', 'percept_acc',
                   'd-prime', 'c', 'beta', 'conf',
                   'c_conf', 'ic_conf',
                   'percept_rt', 'conf_rt',
                   'threshold'
                   ]
    pro_data = pd.DataFrame(pro_data, columns=column_name)

    return pro_data


def merge8(data):
    n_subjs = len(pd.unique(data.subject_ID))
    n_conds = 4
    n_level = 2
    n_columns = 12
    pro_data = np.empty(shape = (n_subjs, n_conds, n_level, n_columns))

    for subj in range(n_subjs):
        subj_data = data[data.subject_ID == subj + 1]
        for cond in range(n_conds):
            all_conds = [np.binary_repr(x, width=4) for x in range(16)]
            for level in range(n_level):
                conds = [int(x, 2) for x in all_conds if x[cond] == str(level)]
                subj_cond_data = subj_data[data.condition.isin(conds)]

                subj_no = subj + 1
                stim_cond = cond
                percept_acc = subj_cond_data.correct.sum() / len(subj_cond_data)
                d_prime, c, beta = data_func.compute_sdt(subj_cond_data.answer, subj_cond_data.percept_resp)
                conf_mean = subj_cond_data.conf_resp.mean()
                c_conf_mean = subj_cond_data[subj_cond_data.correct == 1].conf_resp.mean()
                ic_conf_mean = subj_cond_data[subj_cond_data.correct == 0].conf_resp.mean()
                percept_rt = subj_cond_data.percept_rt.mean()
                conf_rt = subj_cond_data.conf_rt.mean()

                # package output
                pro_data[subj, cond, level, :] = (subj_no, stim_cond, level, percept_acc,
                                                   d_prime, c, beta, conf_mean,
                                                   c_conf_mean, ic_conf_mean,
                                                  percept_rt, conf_rt

                                           )

    # convert to pd.DataFrame
    pro_data = pro_data.reshape(n_subjs * n_conds * n_level, n_columns)
    column_name = ['subj_no', 'stim_cond', 'stim_level', 'percept_acc',
                   'd-prime', 'c', 'beta', 'conf',
                   'c_conf', 'ic_conf',
                   'percept_rt', 'conf_rt'
                   ]
    pro_data = pd.DataFrame(pro_data, columns=column_name)

    return pro_data


def find_exclude(data):
    # find exclude subject list
    d_mean = data.groupby('subj_no')['percept_acc'].mean()
    conf_mean = data.groupby('subj_no')['conf'].mean()
    exclude = d_mean[(d_mean < 0.55) | (d_mean > 0.95)].index.tolist()
    exclude = exclude + (conf_mean[conf_mean > 3.7].index.tolist())
    return exclude


def exclude(data, exclude_list):
    # exclude and reindex subj no
    try:
        data = data[~data.subj_no.isin(exclude_list)]
        data['subj_no'] = data.subj_no.rank(method='dense').astype(int)
    except AttributeError:
        data = data[~data.subject_ID.isin(exclude_list)]
        data['subject_ID'] = data.subject_ID.rank(method='dense').astype(int)
    return data


def plot_MC_diff(data, path):
    subjs = data.subj_no.unique()
    measures = ['d-prime', 'conf']
    plot_data = np.empty(shape = (2, 4, len(subjs)))

    for m , measure in enumerate(measures):
        for s, subj in enumerate(subjs):
            subj_data = data[data.subj_no == subj]
            for factor in range(4):
                plot_data[m][factor][s] = \
                np.average(subj_data[(subj_data.stim_cond == factor) & (subj_data.stim_level == 0)][measure]) - \
                np.average(subj_data[(subj_data.stim_cond == factor) & (subj_data.stim_level == 1)][measure])

    # compute stats
    label = ['Size', 'SF', 'Noise', 'Tilt']
    paired_t_tests = [[], []]
    for m in range(2):
        for i in range(4):
            for j in range(i+1, 4):
                t_stat, p_val = ttest_rel(plot_data[m, i, :], plot_data[m, j, :])
                bayes10 = float(pg.ttest(plot_data[m, i, :], plot_data[m, j, :], paired=True)['BF10'].values[0])
                bayes01 = 1 / bayes10
                paired_t_tests[m].append((label[i], label[j], t_stat, p_val, bayes10, bayes01))

    # plot graph
    plt.clf()
    fig,ax = plt.subplots(1, 2, layout='constrained', figsize=(7, 4))
    plate = plt.cm.get_cmap('Dark2', 8)
    measures = ['d-prime', 'conf']
    conds_label = ['Size', 'Spatial\nfrequency', 'Noise', 'Tilt\noffset']
    titles = ['d\' difference', 'Confidence difference']
    ylabels = [r'$\Delta$' + 'd\'', r'$\Delta$' + 'Confidence']
    ylims = [[-0.2, 2], [-0.2,1.5]]

    for m in range(2):
        avg = np.average(plot_data, axis = 2)
        se = sem(plot_data, axis = 2)
        for factor in range(4):
            ax[m].bar(factor, avg[m][factor], yerr=se[m][factor], color=plate(factor),
                      alpha=0.5
                        )
            ax[m].set_xticks([0,1,2,3], conds_label, fontsize=14)
            ax[m].set_ylabel(ylabels[m], fontsize=14)
            ax[m].set_title(titles[m], fontsize=12, fontweight='bold')
            ax[m].set_ylim(ylims[m])
            ax[m].spines['top'].set_visible(False)
            ax[m].spines['right'].set_visible(False)

            for sub in range(plot_data.shape[2]):
                ax[m].scatter(factor-0.25, plot_data[m][factor][sub], s=5, color=plate(factor),
                              )
                # if factor == 0:
                #     ax[m].plot(np.array([0,1,2,3])-0.25, plot_data[m,:,sub], color='k',
                #                alpha = 0.1
                #               )

        # annotate
        y_pos_array = [np.array([3, 4, 1.8, 3.25, 1.6, 1.4]),
                       np.array([3, 4, 1.8, 3.25, 1.6, 1.4]) - 0.5,
                       ]

        for i, test in enumerate(paired_t_tests[m]):
            # print(test)
            if i in [2, 4, 5]:      # only plot relationship with tilt
                x_pos = 0.5 * (label.index(test[0]) + label.index(test[1]))
                y_pos = y_pos_array[m][i]

                # Determine if p is over power 3
                if test[3] < 1e-3:
                    power = int(np.floor(np.log10(test[3])))
                    coefficient = test[3] / (10 ** power)
                    annotation = r"$p = {:.2f} \times 10^{{{}}}$".format(coefficient, power)
                else:
                    annotation = r"$p = {:.3f}$".format(test[3])

                ax[m].annotate(annotation, xy=(x_pos, y_pos),
                              xytext=(0, 0), textcoords='offset points',
                              ha='center', va='bottom', fontsize=10, color='black',
                             )
                y_pos = y_pos
                ax[m].plot([label.index(test[0]), label.index(test[1])], [y_pos, y_pos],
                         color='k', alpha = 0.5
                           )

    plt.suptitle('Difference analysis\n', fontsize=16, fontweight='bold')
    plot_name = path / f'merged_conditions_difference_plot.png'
    plt.savefig(plot_name, format='png', dpi=384, transparent=True)
    print(plot_name)


def plot_MC_z_transformed_diff(data, path):
    subjs = data.subj_no.unique()
    measures = ['d-prime', 'conf']
    plot_data = np.empty(shape = (2, 4, len(subjs)))

    for m , measure in enumerate(measures):
        for s, subj in enumerate(subjs):
            subj_data = data[data.subj_no == subj]
            for factor in range(4):
                plot_data[m][factor][s] = \
                np.average(subj_data[(subj_data.stim_cond == factor) & (subj_data.stim_level == 0)][measure]) - \
                np.average(subj_data[(subj_data.stim_cond == factor) & (subj_data.stim_level == 1)][measure])

    mean = np.mean(plot_data, axis = (1,2), keepdims=True)
    std = np.std(plot_data, axis = (1,2), keepdims=True)
    plot_data = (plot_data - mean) / std

    label = ['Size', 'Spatial \nFrequency', 'Noise', 'Tilt \noffset']
    paired_t_tests = []

    # compute stats for within feature, across metrics
    for i in range(4):
        for m in range(2):
            for j in range(m+1, 2):
                t_stat, p_val = ttest_rel(plot_data[m,i], plot_data[j,i])
                bayes10 = float(pg.ttest(plot_data[m,i], plot_data[j,i], paired=True)['BF10'].values[0])
                bayes01 = 1 / bayes10
                paired_t_tests.append((label[i], t_stat, p_val, bayes10, bayes01))

#     # compute stats for cross feature on difference
    plot_data = -np.diff(plot_data, axis=0)
    plot_data = plot_data.squeeze()
    cross_feat_t_tests = []
    for i in range(4):
        for j in range(i+1, 4):
            t_stat, p_val = ttest_rel(plot_data[i], plot_data[j])
            bayes10 = float(pg.ttest(plot_data[i], plot_data[j], paired=True)['BF10'].values[0])
            bayes01 = 1 / bayes10
            cross_feat_t_tests.append((label[i], label[j], t_stat, p_val, bayes10, bayes01))

    # plot graph
    plt.clf()
    fig, ax = plt.subplots(1,1, layout="constrained", figsize=(6,4))
    plate = plt.cm.get_cmap('Dark2', 8)
    plate = [plate(0), plate(1), plate(2), plate(3)]
    titles = ['d\' difference', 'Confidence difference']
    x_pos = np.array([0,1,2,3])
    buffer = [-0.25, 0.25]
    alpha = [0.65, 0.35]
    labels = ['d\'', 'Confidence']

    avg = np.average(plot_data, axis = 1)
    se = sem(plot_data, axis = 1)
    ax.bar(x_pos, avg, yerr=se, color=plate, alpha = 0.5, width=0.75)
    ax.set_xticks(x_pos, label, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for sub in range(plot_data.shape[1]):
        x_position = x_pos - 0.25
        ax.scatter(x_position, plot_data[:, sub], color=plate, alpha=1,
                      s=5
                      )
        # ax.plot(x_position, plot_data[m][sub], color='k', alpha = 0.1)

    x_pos = [0.2,1.2,2.2,3.2]
    y_pos = [-1.1, -1.1, -0.8, -0.5]
    for i, test in enumerate(paired_t_tests):
        print(test)
        # Determine if p is over power 3
        if test[2] < 1e-3:
            power = int(np.floor(np.log10(test[2])))
            coefficient = test[2] / (10 ** power)
            annotation = r"$p = {:.2f} \times 10^{{{}}}$".format(coefficient, power)
        else:
            annotation = r"$p = {:.3f}$".format(test[2])

        ax.annotate(annotation, xy=(x_pos[i], y_pos[i]),
                      xytext=(0, 0), textcoords='offset points',
                      ha='center', va='bottom', fontsize=6, color='black',
                     )

    print('--------------------------------------------------------------------------------')

    y_pos_array = np.array([3, 4, 6, 3.5, 5.25, 4.5])
    for i, test in enumerate(cross_feat_t_tests):
        print(test)
        x_pos = 0.5 * (label.index(test[0]) + label.index(test[1]))

        # Determine if p is over power 3
        if test[3] < 1e-3:
            power = int(np.floor(np.log10(test[3])))
            coefficient = test[3] / (10 ** power)
            annotation = r"$p = {:.2f} \times 10^{{{}}}$".format(coefficient, power)
            alpha = 1
        else:
            annotation = r"$p = {:.2f}$".format(test[3])
            alpha = 0.5

        ax.annotate(annotation, xy=(x_pos, y_pos_array[i]),
                      xytext=(0, 0), textcoords='offset points',
                      ha='center', va='bottom', fontsize=8, color='black',
                    alpha = alpha
                     )
        ax.plot([label.index(test[0]), label.index(test[1])], [y_pos_array[i], y_pos_array[i]],
                 color='k', alpha = alpha
                   )

    # plot a horizontal line at 0, y = 0
    ax.plot([-1.25, 5.5], [0, 0], 'k', alpha=1)
    # upper arrow on the line
    ax.arrow(3.75, 0, 0, 0.7, head_width=0.1, head_length=0.25, width=0.025,fc='k', ec='k')
    ax.arrow(3.75, 0, 0, -0.7, head_width=0.1, head_length=0.25, width=0.025,fc='k', ec='k')
    ax.annotate('Larger effect on \naccuracy than confidence', xy=(3.85, 0.25), xytext=(0, 0), textcoords='offset points',
                fontsize=8, color='black', fontweight='bold')
    ax.annotate('Larger effect on \nconfidence than accuracy', xy=(3.85, -0.95), xytext=(0, 0), textcoords='offset points',
                fontsize=8, color='black', fontweight='bold')
    ax.set_xlim(-0.75, 5.5)

    ax.set_ylabel(r"z($\Delta$d') - z($\Delta$confidence)", fontsize=14)
    plt.suptitle('z-scored difference analysis\n', fontsize=18, fontweight='bold')
    plot_name = path / 'z_difference.png'
    fig.savefig(plot_name, format='png', dpi=384, transparent=True)
    print(plot_name)


def plot_MC_folded_X(data, path):
    plt.clf()
    fig, ax = plt.subplots(2,2, figsize=(5,5), layout='constrained')
    ax = ax.flatten()

    subjs = data.subj_no.unique()
    expt_conds = data.stim_cond.unique()
    levels = data.stim_level.unique()
    rearg_data = np.empty(shape=(2, 2, expt_conds.shape[0], subjs.shape[0]))

    for level in levels:
        for cond in expt_conds:
            cond_data = data[(data.stim_cond == cond) & (data.stim_level == level)]
            rearg_data[int(level)][0][int(cond)][:] = cond_data.c_conf.to_numpy()
            rearg_data[int(level)][1][int(cond)][:] = cond_data.ic_conf.to_numpy()
    avg_array = np.mean(rearg_data, axis = 3)
    sem_array = sem(rearg_data, axis = 3)

    title = ['Size', 'Spatial frequency', 'Noise', 'Tilt offset']
    plate = ['green', 'red']
    # marker= ['o', 'x']
    marker= ['None', 'None']
    label = ['Correct', 'Error']
    for c_ic in range(2):
        for factor in range(4):
            ax[factor].errorbar(np.array([0,1]), [avg_array[1][c_ic][factor], avg_array[0][c_ic][factor]],
                                yerr=[sem_array[1][c_ic][factor], sem_array[0][c_ic][factor]], color=plate[c_ic],
                                marker=marker[c_ic], ecolor='k', lw=2, label=label[c_ic]
                                )
            ax[factor].set_title(title[factor], fontsize=14, fontweight='bold')
            ax[factor].set_xlim(-0.5, 1.5)
            ax[factor].set_xticks([0,1], ['Hard', 'Easy'], fontsize=12)
            ax[factor].spines['top'].set_visible(False)
            ax[factor].spines['right'].set_visible(False)
            # ax[factor].set_ylim(1.75, 3)
            ax[factor].set_ylabel('Confidence', fontsize=14)

#   # run linear regression and annotate
    m_array = np.empty(shape=(2, 4, subjs.shape[0]))
    t_value_array = np.empty(shape=(2,4))
    p_value_array = np.empty(shape=(2,4))
    bayes_value_array = np.empty(shape=(2,4))
    for corr in range(2):
        for feat in range(4):
            for subj in range(subjs.shape[0]):
                fit_x = [1, 0]
                fit_y = rearg_data[:, corr, feat, subj]
                m, _ = fit_to_line(fit_x, fit_y)
                m_array[corr, feat, subj] = m
            results = ttest_1samp(m_array[corr, feat], 0, alternative='two-sided')
            bayes01 = 1 / float(pg.ttest(m_array[corr, feat], 0)['BF10'].values[0])
            t_value_array[corr, feat] = results.statistic
            p_value_array[corr, feat] = results.pvalue
            bayes_value_array[corr,feat] = bayes01

    y_annotate_pos = [[2.5, 2.5, 2.5, 2.55], [2.3, 2.35, 2.3, 2.2]]
    color=['green', 'red']
    for feat in range(4):
        for correct in range(2):
            t_value = t_value_array[correct][feat]
            p_value = p_value_array[correct][feat]
            # Format p-value
            if p_value >= 0.001:
                annotation = r"$p = {:.3f}$".format(p_value)
            else:
                power = int(np.floor(np.log10(p_value)))
                coefficient = p_value / (10 ** power)
                annotation = r"$p = {:.2f} \times 10^{{{}}}$".format(coefficient, power)
            ax[feat].text(1, y_annotate_pos[correct][feat], annotation, fontsize=8, ha='right', color=color[correct])

    # print(np.mean(m_array, axis = 2))
    # print(t_value_array)
    # print(p_value_array)
    plt.suptitle(f'Experiment 2', fontsize=24, fontweight='bold')
    ax[0].set_ylim(1.8, 3.2)
    ax[0].legend(fontsize=10, loc = 'upper left')
    plot_name = path / f'merged_conditions_folded_X.png'
    plt.savefig(plot_name, format='png', dpi=384, transparent=True)
    print(plot_name)


def fit_to_line(x, y):
    # least square regression, y = mx + c
    X = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(X, y, rcond=None)[0]
    return m, c


def plot_MC_acc_conf_scatter(data, path):
    subjs = data.subj_no.unique()
    expt_conds = data.stim_cond.unique()
    levels = data.stim_level.unique()
    slope = np.empty(shape=(subjs.shape[0], expt_conds.shape[0]))

    plt.clf()
    label = ['Size', 'Spatial frequency', 'Noise', 'Tilt offset']
    markers = ['o', 'X', 'D', '*']
    plate = plt.cm.get_cmap('Dark2', 8)
    fig = plt.figure(layout='constrained', figsize=(4, 4))

    for i, cond in enumerate(expt_conds):
        x_axis = []
        y_axis = []
        for level in levels:
            x_axis.append(np.average(
                data[(data.stim_cond == cond) &
                     (data.stim_level == level)]['d-prime']))
            y_axis.append(np.average(
                data[(data.stim_cond == cond) &
                     (data.stim_level == level)]['conf']))

        m, c = fit_to_line(x_axis, y_axis)
        plt.scatter(x_axis, y_axis, label=label[i], color=plate(i), marker=markers[i], s=60, alpha=0.8)
        plt.plot(np.linspace(0,3,100), [m*x+c for x in np.linspace(0,3,100)], color=plate(i), alpha=1, lw=2)
        plt.xlim(1,2.4)
        plt.ylim(2,3.3)
        plt.xlabel('d\'', fontsize=16)
        plt.ylabel('Confidence', fontsize=16)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('Relationship between \nconfidence and d\'\n', fontsize=18, fontweight='bold')
    plt.legend(frameon=True, fontsize=10, loc='upper left')
    plot_name = path / 'merged_conditions_acc_conf_scatter.png'
    plt.savefig(plot_name, format='png', dpi=384, transparent=True)
    print(plot_name)


def plot_MC_acc_conf_slope(data, path):
    subjs = data.subj_no.unique()
    expt_conds = data.stim_cond.unique()
    levels = data.stim_level.unique()
    slope = np.empty(shape=(subjs.shape[0], expt_conds.shape[0]))
    exclude_list = []

    for s, subj in enumerate(subjs):
        for cond in expt_conds:
            x_axis = []
            y_axis = []
            for level in levels:
                x_axis.append(np.average(
                    data[(data.subj_no == subj) &
                         (data.stim_cond == cond) &
                         (data.stim_level == level)]['d-prime']))
                y_axis.append(np.average(
                    data[(data.subj_no == subj) &
                         (data.stim_cond == cond) &
                         (data.stim_level == level)]['conf']))

            m, c = fit_to_line(x_axis, y_axis)
            slope[s][int(cond)] = m
            # exclude participants with d' difference less than 0.15
            # if np.abs(x_axis[1] - x_axis[0]) < 0.12:
            if x_axis[1] - x_axis[0] > 0:
                exclude_list.append(s)

    exclude_list = list(set(exclude_list))
    # print(len(exclude_list))
    slope = np.delete(slope, exclude_list, axis = 0)

    label = ['Size', 'Spatial\nfrequency', 'Noise', 'Tilt\noffset']
    paired_t_tests = []
    for i in range(4):
        for j in range(i+1, 4):
            t_stat, p_val = ttest_rel(slope[:, i], slope[:, j])
            bayes10 = float(pg.ttest(slope[:, i], slope[:, j], paired=True)['BF10'].values[0])
            bayes01 = 1 / bayes10
            paired_t_tests.append((label[i], label[j], t_stat, p_val, bayes10, bayes01))

    plt.clf()
    fig = plt.figure(layout='constrained', figsize=(4,4))
    label = ['Size', 'Spatial\nfrequency', 'Noise', 'Tilt\noffset']
    plate = plt.cm.get_cmap('Dark2', 8)

    # plot graph
    for i in range(4):
        # print(np.average(slope, axis=0)[i])
        plt.bar(i, np.average(slope, axis = 0)[i], yerr=sem(slope, axis = 0)[i],
                color=plate(i), alpha=0.5
                )
        # plot individual
        for j in range(len(slope)):
            plt.scatter(i-0.25, slope[j][i], color=plate(i), alpha = 1, s=5)
            # if i == 0:
                # plt.plot(np.array([0,1,2,3])-0.25, slope[j], color='k', alpha = 0.1)

    # annotate
    y_pos_array = np.array([4, 4.5, 6, 3.5, 5.5, 5]) - 1.5
    for i, test in enumerate(paired_t_tests):
        # print(test)
        x_pos = 0.5 * (label.index(test[0]) + label.index(test[1]))
        y_pos = y_pos_array[i]

        # Determine if p is over power 3
        if test[3] < 1e-3:
            power = int(np.floor(np.log10(test[3])))
            coefficient = test[3] / (10 ** power)
            annotation = r"$p = {:.2f} \times 10^{{{}}}$".format(coefficient, power)
        else:
            annotation = r"$p = {:.3f}$".format(test[3])
        if test[3] > 0.1:
            alpha = 0.5
        else:
            alpha = 1

        plt.annotate(annotation, xy=(x_pos, y_pos),
                      xytext=(0, 0), textcoords='offset points',
                      ha='center', va='bottom', fontsize=12, color='black',
                     alpha=alpha
                     )
        y_pos = y_pos
        plt.plot([label.index(test[0]), label.index(test[1])], [y_pos, y_pos],
                 color='k', alpha = alpha
                 )

    # plt.ylim(0, 1.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([0,1,2,3], label, fontsize=16)
    plt.ylim(-0.5, 5)
    plt.ylabel('Slope', fontsize=16)
    plt.title('Slope analysis', fontsize=18, fontweight='bold')
    plot_name = path / 'merged_conditions_acc_conf_slope.png'
    plt.savefig(plot_name, format='png', dpi=384, transparent=True)
    print(plot_name)


def plot_2way_ix(data, path):
    features = ['Size', 'Spatial frequency', 'Noise', 'Tilt offset']
    data.stim_cond = data.stim_cond.astype(int)
    data[features] = data['stim_cond'].apply(lambda x: pd.Series(list(f"{x:04b}"))).astype(int)

    all_pairs = list(combinations(features, 2))
    # swapping for plot
    all_pairs[1], all_pairs[2] = all_pairs[2], all_pairs[1]
    all_pairs[3], all_pairs[4] = all_pairs[4], all_pairs[3]
    all_pairs[0], all_pairs[1] = all_pairs[1], all_pairs[0]
    all_pairs[2], all_pairs[3] = all_pairs[3], all_pairs[2]
    all_pairs[4], all_pairs[5] = all_pairs[5], all_pairs[4]

    plt.clf()
    fig, ax = plt.subplots(3,4, figsize=(8,7), layout='constrained')
    ax = ax.flatten()
    plate = plt.cm.get_cmap('Dark2', 8)
    measures = ['d-prime', 'conf']

    pair_avg = np.empty(shape=(6, 2,2,2))
    pair_sem = np.empty(shape=(6, 2,2,2))
    pair_p_values = np.empty(shape=(6,2))

    for pair_idx, pair in enumerate(all_pairs):
        if pair_idx % 2 == 0:
            pair = pair[::-1]

        for measure_idx, measure in enumerate(measures):
            anova = pg.rm_anova(
                data = data,
                dv = measure,
                within = [pair[0], pair[1]],
                subject = 'subj_no',
                detailed=True
            )
            p_value = anova[anova['Source'] == f'{pair[0]} * {pair[1]}']['p-unc'][2]
            pair_p_values[pair_idx, measure_idx] = p_value

            posthoc = pg.pairwise_tests(
                data = data,
                dv = measure,
                within = [pair[0], pair[1]],
                subject = 'subj_no',
                interaction = True,
            )

            for cond1 in [0,1]:
                for cond2 in [0,1]:
                    subset = data[(data[f'{pair[0]}'] == cond1) & (data[f'{pair[1]}'] == cond2)]
                    pair_avg[pair_idx, measure_idx, cond1, cond2] = subset[measure].mean()
                    pair_sem[pair_idx, measure_idx, cond1, cond2] = sem(subset[measure])

    color = [3, 0, 3, 0, 3, 2]
    for pair_idx, pair in enumerate(all_pairs):
        if pair_idx % 2 == 0:
            pair = pair[::-1]
        if pair_idx == 5:
            pair = pair[::-1]

        # plot accuracy
        ax[pair_idx*2].errorbar([1,0], pair_avg[pair_idx, 0, 0], yerr=pair_sem[pair_idx, 0, 0],
                                marker='o',
                                color=plate(color[pair_idx]),
                                alpha = 0.5,
                                label=f"{pair[0]} easy", markerfacecolor='None',
                                markersize=0, linewidth=2

                                )
        ax[pair_idx*2].errorbar([1,0], pair_avg[pair_idx, 0, 1], yerr=pair_sem[pair_idx, 0, 1],
                              marker='x',
                                color=plate(color[pair_idx]),
                                label=f'{pair[0]} hard',
                                markersize=0, linewidth=2
                                )

        # plot confidence
        ax[pair_idx*2+1].errorbar([1,0], pair_avg[pair_idx, 1, 0], yerr=pair_sem[pair_idx, 1, 0],
                                  marker='o',
                                  color=plate(color[pair_idx]),
                                  alpha = 0.5,
                                  label=f"{pair[0]} easy", markerfacecolor='None',
                                  markersize=0, linewidth=2
                                  )
        ax[pair_idx*2+1].errorbar([1,0], pair_avg[pair_idx, 1, 1], yerr=pair_sem[pair_idx, 1, 1],
                                  marker='x',
                                  color=plate(color[pair_idx]),
                                  label=f'{pair[0]} hard',
                                  markersize=0, linewidth=2
                                  )

        y_pos = [0.8, 2.1]
        for x in range(2):
            p_value = pair_p_values[pair_idx, x]
            # Format p-value
            if p_value >= 0.001:
                annotation = r"$p = {:.3f}$".format(p_value)
            else:
                power = int(np.floor(np.log10(p_value)))
                coefficient = p_value / (10 ** power)
                annotation = r"$p = {:.2f} \times 10^{{{}}}$".format(coefficient, power)
            ax[pair_idx*2+x].text(1.5, y_pos[x], annotation, fontsize=8, ha='right')

        ax[pair_idx*2].set_xlim(-0.5, 1.5)
        ax[pair_idx*2].set_ylim(0.6,3.5)

        ax[pair_idx*2+1].set_xlim(-0.5, 1.5)
        ax[pair_idx*2+1].set_ylim(2,3.6)
        ax[pair_idx*2+1].set_yticks([2,3],[2,3])

        ax[pair_idx*2].set_xticks([1,0], ['Easy', 'Hard'])
        ax[pair_idx*2+1].set_xticks([1,0], ['Easy', 'Hard'])
        ax[pair_idx*2].set_xlabel(f'{pair[1]}', fontweight='bold')
        ax[pair_idx*2+1].set_xlabel(f'{pair[1]}', fontweight='bold')
        ax[pair_idx*2].set_ylabel('\nd\'', fontweight='bold')
        ax[pair_idx*2+1].set_ylabel('confidence', fontweight='bold')

        ax[pair_idx*2].legend(fontsize=7, loc='upper left')
        ax[pair_idx*2+1].legend(fontsize=7, loc='upper left')
        ax[pair_idx*2].set_title(f"\n\n\nd'", fontsize=10)
        ax[pair_idx*2+1].set_title(f"\n\n\nConfidence", fontsize=10)

        for axis in ax:
            axis.spines['top'].set_visible(False)  # Remove top spine
            axis.spines['right'].set_visible(False)  # Remove right spine



    # plt.suptitle('Two-way interaction effects\n', fontsize=18, fontweight='bold')
    plot_name = path / f'2_way_Ix_plot.png'
    plt.savefig(plot_name, format='png', dpi=384, transparent=False)
    print(plot_name)


def graph(data, path):
    print("\nCreating these graphs ......")

    # Figure 4
    plot_2way_ix(data, path)


def graphMC(data, path):
    # Figure 3a, 3b
    plot_MC_acc_conf_scatter(data, path)
    plot_MC_acc_conf_slope(data, path)

    # Figure 3c
    plot_MC_diff(data, path)

    plot_MC_z_transformed_diff(data, path)

    # Figure 5
    plot_MC_folded_X(data, path)



def four_way_anova(data, measures):
    print('__________________________________________________________________')
    print(f'Four-way ANOVA — {measures}')
    print('__________________________________________________________________')
    model = afex.aov_ez(id='subj_no',
                        dv = measures,
                        data = data,
                        within=['size', 'sf', 'noise','tilt'],
                        anova_table=['pes']
                        )
    print(model)
    print(base.summary(model))


def two_way_posthoc(data, measure):
    print('__________________________________________________________________')
    print(f'Two-way Posthoc — {measure}')
    print('__________________________________________________________________')
    features = ['Size', 'Spatial freq.', 'Noise', 'Tilt offset']
    data.stim_cond = data.stim_cond.astype(int)
    data[features] = data['stim_cond'].apply(lambda x: pd.Series(list(f"{x:04b}"))).astype(int)
    all_pairs = list(combinations(features, 2))
    all_pairs[1], all_pairs[2] = all_pairs[2], all_pairs[1]  # Swap pairs 1 and 2
    all_pairs[3], all_pairs[4] = all_pairs[4], all_pairs[3]  # Swap pairs 3 and 4

    for pair_idx, pair in enumerate(all_pairs):
        posthoc = pg.pairwise_tests(
            data = data,
            dv = measure,
            within = [pair[0], pair[1]],
            subject = 'subj_no',
            interaction = True,
            effsize='cohen'
        )
        print(posthoc)
        print('\n')


def stat(data, path):
    print("\n Running Statistical Tests ......")
    orig_stdout = sys.stdout
    sys.stdout = open(path, 'w')

    # breakdown binary number to four condition
    data['binary'] = data['stim_cond'].apply(lambda x: format(int(x), '04b'))
    data[['size', 'sf', 'noise', 'tilt']] = data['binary'].apply(lambda x: pd.Series(list(x)))

    # do four way anova for measures
    four_way_anova(data, 'd-prime')
    four_way_anova(data, 'conf')
    two_way_posthoc(data, 'd-prime')
    two_way_posthoc(data, 'conf')

    sys.stdout.close()
    sys.stdout = orig_stdout
    print(f"Stat results save in {path}")


def main():
    in_data_path, out_pro_path, stat_path, graph_path = manage_path()
    if run_process:
        print("\nProcessing individuals ......")
        data = read(in_data_path)

        # process data in different ways
        # process all sdt measures on 16 conditions independently
        pro_data = process(data)
        # process all sdt measures merging 8 conditions together
        # (e.g. all size easy vs. all size hard)
        pro_8_data = merge8(data)

        # exclude subjects
        exclude_list = find_exclude(pro_data)
        pro_data = exclude(pro_data, exclude_list)
        pro_8_data = exclude(pro_8_data, exclude_list)

        # exclude subject in raw data
        data = exclude(data, exclude_list)

        # output all to csv
        data.to_csv(in_data_path, sep=',', index=False)
        pro_data.to_csv(f'{out_pro_path}/16c_processed_data.csv', sep=',', index=False)
        pro_8_data.to_csv(f'{out_pro_path}/merge8_processed_data.csv', sep=',', index=False)
        print('--------------------------------------------------------------------------------')
        print('Processing Completed.')

    else:
        pro_data = pd.read_csv(f'{out_pro_path}/16c_processed_data.csv')
        pro_8_data = pd.read_csv(f'{out_pro_path}/merge8_processed_data.csv')
        print("Processed data read from path")

    stat(pro_data, stat_path)
    graph(pro_data, graph_path)
    graphMC(pro_8_data, graph_path)

    print("--------------------------------------------------------------------------------")
    print('ALL DONE')


if __name__ == "__main__":
    main()
