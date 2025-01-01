'''
analysis script
'''

run_process = True

# import modules
from function import data_func
from scipy.stats import pearsonr, norm, sem, ttest_rel, ttest_1samp
from matplotlib import rcParams
import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import os
import pandas as pd
import pathlib
import pingouin as pg
import re
import seaborn as sns
from statsmodels.stats.anova import AnovaRM
import sys
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")
rcParams['font.family'] = 'CMU Sans Serif'


def progress_bar(progress, total):
    percent = 100 * (progress / float(total))
    bar = ' ' * int(percent) + '-' * (100 - int(percent))
    print(f"\r|{bar}| {percent:.2f}%", end="\r")


def manage_path():
    # input
    current_path = pathlib.Path(__file__).parent.absolute()
    data_path = current_path / 'data/expt_1.csv'

    # output
    result_path = current_path / 'results' / 'expt_1'
    pro_data_path = current_path / 'results/expt_1/processed_data.csv'
    out_stat_path = current_path / 'results/expt_1/stat_result.txt'
    out_graph_path = current_path / 'results/expt_1/graph'

    try:
        result_path.mkdir(parents=True, exist_ok=True)
        out_graph_path.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        print('result directory already existed ......')

    return data_path, pro_data_path, out_stat_path, out_graph_path


def read(datafile):
    data = pd.read_csv(datafile, delimiter = ',')
    return data


def find_exclude(data):
    # find exclude subject list
    d_mean = data.groupby('subj_no')['percept_acc'].mean()
    conf_mean = data.groupby('subj_no')['conf_resp'].mean()
    exclude = d_mean[((d_mean < 0.55) | (d_mean > 0.95))].index.tolist()
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

def process(data):
    n_subjs = len(pd.unique(data.subject_ID))
    pro_data = np.empty(shape = (n_subjs, 9, 12))
    for subj in range(n_subjs):
        subj_data = data[data.subject_ID == subj + 1]
        for cond in range(9):
            subj_cond_data = subj_data[data.stim_condition == cond + 1]

            subj_no = subj + 1
            stim_cond = cond + 1
            percept_acc = subj_cond_data.correct.sum() / len(subj_cond_data)
            d_prime, c, beta = data_func.compute_sdt(subj_cond_data.answer, subj_cond_data.percept_resp)
            conf_mean = subj_cond_data.conf_resp.mean()
            c_conf_mean = subj_cond_data[subj_cond_data.correct == 1].conf_resp.mean()
            ic_conf_mean = subj_cond_data[subj_cond_data.correct == 0].conf_resp.mean()
            percept_rt = subj_cond_data.percept_rt.mean()
            conf_rt = subj_cond_data.conf_rt.mean()
            threshold = subj_data[subj_data.stim_condition == 9].stim_tilt_diff.unique()[0]

            # package output
            pro_data[subj, cond, :] = (subj_no, stim_cond, percept_acc,
                                       d_prime, c, beta, conf_mean,
                                       c_conf_mean, ic_conf_mean,
                                       percept_rt, conf_rt,
                                       threshold
                                       )

    # convert to pd.DataFrame
    pro_data = pro_data.reshape(n_subjs*9, 12)
    column_name = ['subj_no', 'stim_cond', 'percept_acc',
                   'd-prime', 'c', 'beta', 'conf_resp',
                   'c_conf_resp', 'ic_conf_resp',
                   'percept_rt', 'conf_rt', 'threshold'
                   ]

    pro_data = pd.DataFrame(pro_data, columns=column_name)
    dummy_code_2_condition = {
        1 : "size_low",
        2 : "size_high",
        3 : "dur_low",
        4 : "dur_high",
        5 : "noise_high",
        6 : "noise_low",
        7 : "tilt_low",
        8 : "tilt_high",
        9 : "baseline",
    }
    pro_data['exp_cond'] = [dummy_code_2_condition[i] for i in pro_data['stim_cond']]

    return pro_data


def assign_condition(data):
    condition = []
    for i in range(len(data)):
        if "baseline" in data['exp_cond'][i]:
            condition.append('baseline')
        elif "size" in data['exp_cond'][i]:
            condition.append('size')
        elif "dur" in data['exp_cond'][i]:
            condition.append('dur')
        elif "tilt" in data['exp_cond'][i]:
            condition.append('tilt')
        elif "noise" in data['exp_cond'][i]:
            condition.append('noise')
    data['stimulus'] = condition

    # triple the baseline and assign dummy stimulus, level
    level = []
    baseline = data[data['exp_cond'] == 'baseline']
    data = data.append([baseline]*3)
    data = data.reset_index(drop = True)

    for i in range(len(data)):
        if "baseline" in data['exp_cond'][i]:
            level.append(2)
        elif "low" in data['exp_cond'][i]:
            if "noise" in data['exp_cond'][i]:
                level.append(3)
            else:
                level.append(1)
        elif "high" in data['exp_cond'][i]:
            if "noise" in data['exp_cond'][i]:
                level.append(1)
            else:
                level.append(3)
    data['stim_level'] = level

    dummy_cond = []
    baseline = data[data['stimulus'] == 'baseline'].sort_values(by='subj_no')
    baseline = baseline.reset_index(drop=True)

    for i in range(int(len(data)/12)):
        dummy_cond.append('size')
        dummy_cond.append('tilt')
        dummy_cond.append('dur')
        dummy_cond.append('noise')
    baseline['dummy_stim'] = dummy_cond

    non_baseline = data[data['stimulus'] != 'baseline']
    non_baseline = non_baseline.reset_index(drop=True)
    non_baseline['dummy_stim'] = non_baseline['stimulus']

    data = pd.concat([baseline, non_baseline])
    data = data.reset_index(drop=True)

    return data


def plot_difference(data, path):
    # plot d' and conf_rating difference
    subjs = data.subj_no.unique()
    split_plot = [['size_low', 'size_high'],
                  ['dur_low', 'dur_high'],
                  ['noise_high', 'noise_low'],
                  ['tilt_low', 'tilt_high']
                  ]
    measures = ['d-prime', 'conf_resp']
    plot_data = np.empty(shape=(2, len(subjs), 4))

    for m, measure in enumerate(measures):
        for s, subj in enumerate(subjs):
            subj_data = data[data.subj_no == subj]
            for i in range(len(split_plot)):
                plot_data[m][s][i] = \
                np.average(subj_data[subj_data.exp_cond == split_plot[i][1]][measure]) - \
                np.average(subj_data[subj_data.exp_cond == split_plot[i][0]][measure])

    label = ['Size', 'Duration', 'Noise', 'Tilt \noffset']
    paired_t_tests = [[], []]

    # compute stats
    for m in range(2):
        for i in range(4):
            for j in range(i+1, 4):
                t_stat, p_val = ttest_rel(plot_data[m, :, i], plot_data[m, :, j])
                bayes10 = float(pg.ttest(plot_data[m, :, i], plot_data[m, :, j], paired=True)['BF10'].values[0])
                bayes01 = 1 / bayes10
                paired_t_tests[m].append((label[i], label[j], t_stat, p_val, bayes10, bayes01))

    # plot graph
    plt.clf()
    fig, ax = plt.subplots(1,2, layout="constrained", figsize=(7,4))
    plate = plt.cm.get_cmap('Dark2', 8)
    plate = [plate(0), plate(5), plate(2), plate(3)]
    titles = ['d\' difference', 'Confidence difference']
    ylabels = [r'$\Delta$' + 'd\'', r'$\Delta$' + 'Confidence']
    ylims = [[-0.7,5.25], [-0.5,3.5]]
    for m in range(2):
        avg = np.average(plot_data, axis = 1)
        se = sem(plot_data, axis = 1)
        ax[m].bar([0,1,2,3], avg[m], yerr=se[m], color=plate, alpha = 0.5)
        ax[m].set_ylabel(ylabels[m], fontsize=14)
        ax[m].set_title(titles[m], fontsize=12, fontweight='bold')
        ax[m].set_xticks([0,1,2,3], label, fontsize=14)
        ax[m].set_ylim(ylims[m])
        ax[m].spines['top'].set_visible(False)
        ax[m].spines['right'].set_visible(False)

        for sub in range(plot_data.shape[1]):
            x_position = np.array([0,1,2,3]) - 0.25
            ax[m].scatter(x_position, plot_data[m][sub], color=plate, alpha=1,
                          s=5
                          )
            # ax[m].plot(x_position, plot_data[m][sub], color='k', alpha = 0.1)

        # annotate
        y_pos_array = [np.array([3, 4, 4.5, 3.25, 4.25, 4]),
                       np.array([3, 4, 4.5, 3.25, 4.25, 4]) - 1.4,
                       ]

        for i, test in enumerate(paired_t_tests[m]):
            print(test)
            if i in [2, 5]:      # only plot relationship with tilt
                x_pos = 0.5 * (label.index(test[0]) + label.index(test[1]))
                y_pos = y_pos_array[m][i]

                # Determine if p is over power 3
                if test[3] < 1e-3:
                    power = int(np.floor(np.log10(test[3])))
                    coefficient = test[3] / (10 ** power)
                    annotation = r"$p = {:.2f} \times 10^{{{}}}$".format(coefficient, power)
                else:
                    annotation = r"$p = {:.2f}$".format(test[3])

                ax[m].annotate(annotation, xy=(x_pos, y_pos),
                              xytext=(0, 0), textcoords='offset points',
                              ha='center', va='bottom', fontsize=10, color='black',
                             )
                y_pos = y_pos
                ax[m].plot([label.index(test[0]), label.index(test[1])], [y_pos, y_pos],
                         color='k', alpha = 0.5
                           )

    plt.suptitle('Difference analysis\n', fontsize=16, fontweight='bold')
    plot_name = path / 'difference.png'
    fig.savefig(plot_name, format='png', dpi=384, transparent=True)
    print(plot_name)


def plot_foldedX(data, path):
    # split correct and incorrect trials and plot confidence separately
    plt.clf()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(5, 5), layout="constrained")
    split_plot = [['size_low', 'baseline', 'size_high'],
                  ['dur_low', 'baseline', 'dur_high'],
                  ['noise_high', 'baseline', 'noise_low'],
                  ['tilt_low', 'baseline', 'tilt_high']
                  ]
    split_plot_name = ['Size', 'Duration', 'Noise', 'Tilt offset']
    axim = [ax1, ax2, ax3, ax4]

    # plot individual data
    plot_data = data[data.subj_no != 27]  # this subject got all correct in one condition
    subj_array = plot_data.subj_no.unique()
    n_subj = len(subj_array)
    for subj in subj_array:
        ind_data = plot_data[plot_data.subj_no == subj]

    stat_data = np.empty(shape=(2, 4, 3, n_subj))
    # plot average
    for i in range(len(split_plot)):
        x = split_plot[i]
        y1 = []
        y2 = []
        y1e = []
        y2e = []
        for j in range(len(x)):
            if j == 1:
                stat_data[0][i][j] = plot_data[plot_data['exp_cond'] == split_plot[i][j]]['c_conf_resp'][::4].to_numpy()
                stat_data[1][i][j] = plot_data[plot_data['exp_cond'] == split_plot[i][j]]['ic_conf_resp'][::4].to_numpy()
            else:
                stat_data[0][i][j] = plot_data[plot_data['exp_cond'] == split_plot[i][j]]['c_conf_resp'].to_numpy()
                stat_data[1][i][j] = plot_data[plot_data['exp_cond'] == split_plot[i][j]]['ic_conf_resp'].to_numpy()
            y1.append(np.average(stat_data[0][i][j]))
            y2.append(np.average(stat_data[1][i][j]))
            y1e.append(sem(stat_data[0][i][j]))
            y2e.append(sem(stat_data[0][i][j]))
        axim[i].errorbar([0.9, 1.9, 2.9],  y1, yerr = y1e, marker='None', c = 'green', linestyle = '-', label='Correct', alpha = 1, lw=2, ecolor='k')
        axim[i].errorbar([1.1, 2.1, 3.1],  y2, yerr = y2e, marker='None', c = 'red', linestyle = '-', label='Error', alpha = 1, lw=2, ecolor='k')
        axim[i].set_title(split_plot_name[i], weight='bold', fontsize=14)
        axim[i].set_xticks([1,2,3], ['Hard', 'Medium', 'Easy'], fontsize=12)
        axim[i].set_xlim([0.5, 3.5])
        # axim[i].set_ylim([1.5,3.2])
        axim[i].set_ylabel('Confidence', fontsize=14)
        # axim[i].set_yticks([1,2,3,4], [1,2,3,4])
        axim[i].spines['top'].set_visible(False)
        axim[i].spines['right'].set_visible(False)

#   # run linear regression and annotate
    m_array = np.empty(shape=(2, 4, n_subj))
    t_value_array = np.empty(shape=(2,4))
    p_value_array = np.empty(shape=(2, 4))
    for corr in range(2):
        for feat in range(4):
            for subj in range(n_subj):
                fit_x = [0, 1, 2]
                fit_y = stat_data[corr, feat, :, subj]
                m, _ = fit_to_line(fit_x, fit_y)
                m_array[corr, feat, subj] = m
            results = ttest_1samp(m_array[corr, feat], 0, alternative='two-sided')
            t_value_array[corr, feat] = results.statistic
            p_value_array[corr, feat] = results.pvalue

    y_annotate_pos = [[2.6, 2.65, 2.6, 2.65], [2.1, 2.25, 2.05, 2.5]]
    color=['green', 'red']
    print(m_array.mean(axis = 2))
    print(t_value_array)
    print(p_value_array)
    for feat in range(4):
        for correct in range(2):
            t_value = t_value_array[correct][feat]
            p_value = p_value_array[correct][feat]
            # Format p-value
            if p_value >= 0.001:
                annotation = r"$p = {:.2f}$".format(p_value)
            else:
                power = int(np.floor(np.log10(p_value)))
                coefficient = p_value / (10 ** power)
                annotation = r"$p = {:.2f} \times 10^{{{}}}$".format(coefficient, power)
            axim[feat].text(3.2, y_annotate_pos[correct][feat], annotation, fontsize=8, ha='right', color=color[correct])


    plt.suptitle('Experiment 1', fontsize=24, fontweight='bold')
    axim[0].set_ylim(1.8, 3.4)
    axim[0].legend(frameon=True, fontsize=10, loc='upper left')
    plot_name = path / 'confidence_folded_x_plot.png'
    plt.savefig(plot_name, format='png', dpi=384, transparent=True)
    print(plot_name)


def fit_to_line(x, y):
    # least square regression, y = mx + c
    X = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(X, y, rcond=None)[0]
    return m, c


def plot_acc_conf_scatter(data, path):
    plt.clf()
    fig = plt.figure(layout='constrained', figsize=(4, 4))
    split_plot = [['size_low', 'baseline', 'size_high'],
                  ['dur_low', 'baseline', 'dur_high'],
                  ['noise_high', 'baseline', 'noise_low'],
                  ['tilt_low', 'baseline', 'tilt_high']
                  ]
    label = ['Size', 'Duration', 'Noise', 'Tilt offset']
    plate = plt.cm.get_cmap('Dark2', 8)
    plate = [plate(0), plate(5), plate(2), plate(3)]

    for i in range(len(split_plot)):
        x = split_plot[i]
        x_axis = []
        y_axis = []
        xerr = []
        yerr = []
        for j in range(len(x)):
            x_axis.append(np.average(data[data.exp_cond == split_plot[i][j]]['d-prime']))
            y_axis.append(np.average(data[data.exp_cond == split_plot[i][j]]['conf_resp']))
            xerr.append(sem(data[data.exp_cond == split_plot[i][j]]['d-prime']))
            yerr.append(sem(data[data.exp_cond == split_plot[i][j]]['conf_resp']))

        m, c = fit_to_line(x_axis, y_axis)
        plt.plot(np.linspace(0,3,100), [m*x+c for x in np.linspace(0,3,100)], color=plate[i],
                 alpha=1, lw=2, zorder=1
                 )
        for k in range(3):
            if k != 1:
                plt.errorbar(x_axis[k], y_axis[k], label=label[i] if k == 0 else None,
                            color=plate[i], marker='o', zorder=2,
                            # xerr=xerr[k], yerr=yerr[k],
                            markersize=5, alpha=1,
                            )
            else:
                plt.errorbar(x_axis[k], y_axis[k], color='black', marker='o', zorder=3,
                            label='Baseline' if i == 3 else None,
                            # xerr=xerr[k]*2, yerr=yerr[k]*2,   # correct for inflated sem
                            markersize=5, alpha=1,
                            )
        # ax[i].plot(x_axis, y_axis, color='red')
        # slope = (y_axis[1] - y_axis[0]) / (x_axis[1] - x_axis[0])
        # ax[i].plot(np.linspace(0,3,100), [y_axis[0] + slope * (x - x_axis[0]) for x in np.linspace(0,3,100)], 'r',
                   # )
        plt.xlim(0,3)
        plt.ylim(1,4)
        plt.xlabel('d\'', fontsize=16)
        plt.ylabel('Confidence', fontsize=16)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('Relationship between \nconfidence and d\'\n', fontsize=18, fontweight='bold')
    plt.legend(frameon=True, fontsize=10, loc='upper left')
    plot_name = path / 'acc_conf_scatter.png'
    plt.savefig(plot_name, format='png', dpi=384, transparent=True)
    print(plot_name)


def plot_acc_conf_slope(data, path):
    subjs = data.subj_no.unique()
    slope = np.empty(shape=(len(subjs), 4))
    c_array = np.empty(shape=(len(subjs), 4))
    exclude_list = []
    split_plot = [['size_low', 'baseline', 'size_high'],
                  ['noise_high', 'baseline', 'noise_low'],
                  ['tilt_low', 'baseline', 'tilt_high'],
                  # ['dur_low', 'baseline', 'dur_high']
                  ]


    for s, subj in enumerate(subjs):
        subj_data = data[data.subj_no == subj]
        for i in range(len(split_plot)):
            x = split_plot[i]
            x_axis = []
            y_axis = []
            for j in range(len(x)):
                x_axis.append(np.average(subj_data[subj_data.exp_cond == split_plot[i][j]]['d-prime']))
                y_axis.append(np.average(subj_data[subj_data.exp_cond == split_plot[i][j]]['conf_resp']))
            # exclude participants with d' difference less than 0.15
            m, c = fit_to_line(x_axis, y_axis)
            slope[s][i] = m
            c_array[s][i] = c
            # if np.abs(x_axis[0] - x_axis[2]) < 0.15:
            if x_axis[0] - x_axis[2] > 0:
                exclude_list.append(s)

    exclude_list = list(set(exclude_list))
    slope = np.delete(slope, exclude_list, axis = 0)
    c_array = np.delete(c_array, exclude_list, axis = 0)

    label = ['Size', 'Noise', 'Tilt offset']
    paired_t_tests = []
    for i in range(3):
        for j in range(i+1, 3):
            t_stat, p_val = ttest_rel(slope[:, i], slope[:, j])
            bayes10 = float(pg.ttest(slope[:, i], slope[:, j], paired=True)['BF10'].values[0])
            bayes01 = 1 / bayes10
            paired_t_tests.append((label[i], label[j], t_stat, p_val, bayes10, bayes01))

    plt.clf()
    fig = plt.figure(layout='constrained', figsize=(4, 4))
    label = ['Size', 'Noise', 'Tilt offset']
    plate = plt.cm.get_cmap('Dark2', 8)
    plate = [plate(0), plate(2), plate(3)]

    # plot graph
    for i in range(3):
        plt.bar(i, np.average(slope, axis = 0)[i], yerr=sem(slope, axis = 0)[i],
                color=plate[i], alpha=0.5
                )
        # plot individual
        for j in range(len(slope)):
            plt.scatter(i-0.25, slope[j][i], color=plate[i], alpha = 1, s=5)
            # if i == 0:
            #     plt.plot(np.array([0,1,2])-0.25, slope[j][:3], color='k', alpha = 0.1)

    # annotate
    y_pos_array = np.array([2.5, 3.5, 3]) - 0.5
    for i, test in enumerate(paired_t_tests):
        print(test)
        x_pos = 0.5 * (label.index(test[0]) + label.index(test[1]))
        y_pos = y_pos_array[i]

        # Determine if p is over power 3
        if test[3] < 1e-3:
            power = int(np.floor(np.log10(test[3])))
            coefficient = test[3] / (10 ** power)
            annotation = r"$p = {:.2f} \times 10^{{{}}}$".format(coefficient, power)
            alpha = 1
        else:
            annotation = r"$p = {:.2f}$".format(test[3])
            alpha = 0.5

        plt.annotate(annotation,
                      xy=(x_pos, y_pos),
                      xytext=(0,0), textcoords='offset points',
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
    plt.xticks([0,1,2], label, fontsize=16)
    plt.ylabel('Slope', fontsize=16)
    plt.title('Slope analysis', fontsize=18, fontweight='bold')
    plot_name = path / 'acc_conf_slope.png'
    plt.savefig(plot_name, format='png', dpi=384, transparent=True)
    print(plot_name)


def graph(data, path):
    print("\nCreating these graphs ......")
    plot_difference(data, path)
    plot_foldedX(data, path)
    plot_acc_conf_scatter(data, path)
    plot_acc_conf_slope(data, path)


def main():
    in_data_path, out_pro_path, stat_path, graph_path = manage_path()
    if run_process:
        print("\nProcessing individuals ......")
        data = read(in_data_path)
        pro_data = process(data)

        exclude_list = find_exclude(pro_data)
        data = exclude(data, exclude_list)
        pro_data = exclude(pro_data, exclude_list)

        # output all to csv
        data.to_csv(in_data_path, sep=',', index=False)
        pro_data.to_csv(out_pro_path, sep=',', index=False)

        print('--------------------------------------------------------------------------------')
        print('Processing Completed.')
    else:
        pro_data = pd.read_csv(out_pro_path)
        print("Processed data read from " + str(out_pro_path))

    pro_data = assign_condition(pro_data)
    graph(pro_data, graph_path)
    print("--------------------------------------------------------------------------------")
    print('ALL DONE')


if __name__ == "__main__":
    main()
