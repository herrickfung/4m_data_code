'''
analysis script for expt 1
'''

run_process = True

# import modules
from matplotlib import rcParams
from scipy.stats import sem, ttest_rel, ttest_1samp
from pymer4.models import Lmer
import pathlib
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import sys
import io

# Local application imports
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
    data_path = current_path / 'data/expt_1.csv'

    # output
    result_path = current_path / 'results' / 'expt_1'
    pro_data_path = current_path / 'results/expt_1/processed_data.csv'
    out_stat_path = current_path / 'results/expt_1/stat_result.txt'
    out_graph_path = current_path / 'results/expt_1/graph'

    result_path.mkdir(parents=True, exist_ok=True)
    out_graph_path.mkdir(parents=True, exist_ok=True)

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
    n_params = 14
    n_subjs = len(pd.unique(data.subject_ID))
    pro_data = np.empty(shape = (n_subjs, 9, n_params))
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
            c_rt_mean = subj_cond_data[subj_cond_data.correct == 1].percept_rt.mean()
            ic_rt_mean = subj_cond_data[subj_cond_data.correct == 0].percept_rt.mean()
            conf_rt = subj_cond_data.conf_rt.mean()
            threshold = subj_data[subj_data.stim_condition == 9].stim_tilt_diff.unique()[0]

            # package output
            pro_data[subj, cond, :] = (subj_no, stim_cond, percept_acc,
                                       d_prime, c, beta, conf_mean,
                                       c_conf_mean, ic_conf_mean,
                                       percept_rt, 
                                       c_rt_mean, ic_rt_mean,
                                       conf_rt,
                                       threshold
                                       )

    # convert to pd.DataFrame
    pro_data = pro_data.reshape(n_subjs*9, n_params)
    column_name = ['subj_no', 'stim_cond', 'percept_acc',
                   'd-prime', 'c', 'beta', 'conf_resp',
                   'c_conf_resp', 'ic_conf_resp',
                   'percept_rt', 
                   'c_rt', 'ic_rt',
                   'conf_rt', 'threshold'
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
    data = pd.concat([data] + [baseline]*3, ignore_index=True)
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

        # annotate
        y_pos_array = [np.array([3, 4, 4.5, 3.25, 4.25, 4]),
                       np.array([3, 4, 4.5, 3.25, 4.25, 4]) - 1.4,
                       ]

        for i, test in enumerate(paired_t_tests[m]):
            # print(test)
            if i in [2, 5]:      # only plot relationship with tilt
                x_pos = 0.5 * (label.index(test[0]) + label.index(test[1]))
                y_pos = y_pos_array[m][i]

                # Determine if p is over power 3
                if test[3] < 1e-3:
                    power = int(np.floor(np.log10(test[3])))
                    coefficient = test[3] / (10 ** power)
                    annotation = r"$p = {:.3f} \times 10^{{{}}}$".format(coefficient, power)
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
    plot_name = path / 'difference.png'
    fig.savefig(plot_name, format='png', dpi=384, transparent=True)
    print(plot_name)
    plt.close()


def plot_difference_rt(data, path):
    subjs = data.subj_no.unique()
    split_plot = [['size_low', 'size_high'],
                  ['dur_low', 'dur_high'],
                  ['noise_high', 'noise_low'],
                  ['tilt_low', 'tilt_high']
                  ]
    plot_data = np.empty(shape=(len(subjs), 4))

    measure = 'percept_rt'
    for s, subj in enumerate(subjs):
        subj_data = data[data.subj_no == subj]
        for i in range(len(split_plot)):
            plot_data[s][i] = \
            np.average(subj_data[subj_data.exp_cond == split_plot[i][0]][measure]) - \
            np.average(subj_data[subj_data.exp_cond == split_plot[i][1]][measure])
    label = ['Size', 'Duration', 'Noise', 'Tilt \noffset']
    paired_t_tests = []
    for i in range(4):
        for j in range(i+1, 4):
            t_stat, p_val = ttest_rel(plot_data[:, i], plot_data[:, j])
            bayes10 = float(pg.ttest(plot_data[:, i], plot_data[:, j], paired=True)['BF10'].values[0])
            bayes01 = 1 / bayes10
            paired_t_tests.append((label[i], label[j], t_stat, p_val, bayes10, bayes01))

    # plot graph
    plt.clf()
    fig, ax = plt.subplots(1,1, layout="constrained", figsize=(3.5, 4))
    plate = plt.cm.get_cmap('Dark2', 8)
    plate = [plate(0), plate(5), plate(2), plate(3)]
    titles = 'Experiment 1'
    ylabels = r'$RT_{Hard} - RT_{Easy}$ (in ms)' 

    avg = np.average(plot_data, axis = 0)
    se = sem(plot_data, axis = 0)
    ax.bar([0,1,2,3], avg, yerr=se, color=plate, alpha = 0.5)
    ax.set_ylim([-1000, 2000])
    ax.set_ylabel(ylabels, fontsize=14)
    ax.set_title(titles, fontsize=12, fontweight='bold')
    ax.set_xticks([0,1,2,3], label, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for sub in range(plot_data.shape[0]):
        x_position = np.array([0,1,2,3]) - 0.25
        ax.scatter(x_position, plot_data[sub], color=plate, alpha=1,
                        s=5
                        )
        
    y_pos_array = np.array([3, 4, 5, 3.25, 4.5, 3.5]) * 350
    for i, test in enumerate(paired_t_tests):
        x_pos = 0.5 * (label.index(test[0]) + label.index(test[1]))
        y_pos = y_pos_array[i]

        # Determine if p is over power 3
        if test[3] < 1e-3:
            power = int(np.floor(np.log10(test[3])))
            coefficient = test[3] / (10 ** power)
            annotation = r"$p = {:.2f} \times 10^{{{}}}$".format(coefficient, power)
        else:
            annotation = r"$p = {:.2f}$".format(test[3])
        ax.annotate(annotation, xy=(x_pos, y_pos),
                        xytext=(0, 0), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10, color='black',
                        )
        y_pos = y_pos
        ax.plot([label.index(test[0]), label.index(test[1])], [y_pos, y_pos],
                    color='k', alpha = 0.5
                    )

    plot_name = path / 'difference_rt.png'
    fig.savefig(plot_name, format='png', dpi=384, transparent=True)
    print(plot_name)
    plt.close()


def plot_z_transform_difference(data, path):
    # z delta d' - z delta confidence

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

    mean = np.mean(plot_data, axis = (1,2), keepdims=True)
    std = np.std(plot_data, axis = (1,2), keepdims=True)
    plot_data = (plot_data - mean) / std

    label = ['Size', 'Duration', 'Noise', 'Tilt \noffset']
    paired_t_tests = []

    # compute stats for within feature, across metrics
    for i in range(4):
        for m in range(2):
            for j in range(m+1, 2):
                t_stat, p_val = ttest_rel(plot_data[m, :, i], plot_data[j, :, i])
                bayes10 = float(pg.ttest(plot_data[m, :, i], plot_data[j, :, i], paired=True)['BF10'].values[0])
                bayes01 = 1 / bayes10
                paired_t_tests.append((label[i], t_stat, p_val, bayes10, bayes01))

    # compute stats for cross feature on difference
    plot_data = -np.diff(plot_data, axis=0)
    plot_data = plot_data.squeeze()
    cross_feat_t_tests = []
    for i in range(4):
        for j in range(i+1, 4):
            t_stat, p_val = ttest_rel(plot_data[:, i], plot_data[:, j])
            bayes10 = float(pg.ttest(plot_data[:, i], plot_data[:, j], paired=True)['BF10'].values[0])
            bayes01 = 1 / bayes10
            cross_feat_t_tests.append((label[i], label[j], t_stat, p_val, bayes10, bayes01))

    # plot graph
    plt.clf()
    fig, ax = plt.subplots(1,1, layout="constrained", figsize=(6,4))
    plate = plt.cm.get_cmap('Dark2', 8)
    plate = [plate(0), plate(5), plate(2), plate(3)]
    titles = ['d\' difference', 'Confidence difference']
    x_pos = np.array([0,1,2,3])
    buffer = [-0.25, 0.25]
    alpha = [0.65, 0.35]
    labels = ['d\'', 'Confidence']

    avg = np.average(plot_data, axis = 0)
    se = sem(plot_data, axis = 0)
    ax.bar(x_pos, avg, yerr=se, color=plate, alpha = 0.5, width=0.75)
    ax.set_xticks(x_pos, label, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for sub in range(plot_data.shape[0]):
        x_position = x_pos - 0.25
        ax.scatter(x_position, plot_data[sub], color=plate, alpha=1,
                      s=5
                      )
        # ax.plot(x_position, plot_data[m][sub], color='k', alpha = 0.1)

    x_pos = [0.2, 1.2, 2.2, 3.2]
    y_pos = [-1, -0.8, -1.1, -0.5]
    for i, test in enumerate(paired_t_tests):
        print(test)
        # Determine if p is over power 3
        if test[2] < 1e-3:
            power = int(np.floor(np.log10(test[2])))
            coefficient = test[2] / (10 ** power)
            annotation = r"$p = {:.2f} \times 10^{{{}}}$".format(coefficient, power)
        else:
            annotation = r"$p = {:.2f}$".format(test[2])

        ax.annotate(annotation, xy=(x_pos[i], y_pos[i]),
                      xytext=(0, 0), textcoords='offset points',
                      ha='center', va='bottom', fontsize=6, color='black',
                     )

#         ax.plot([x_pos[i]-0.25, x_pos[i]+0.25], [y_pos, y_pos],
#                  color='k', alpha = 0.5
#                    )

    print('--------------------------------------------------------------------------------')
    y_pos_array = np.array([3, 4, 6, 3.5, 5.25, 4.5]) - 0.4
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
    plt.close()


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
            y1.append(np.nanmean(stat_data[0][i][j]))
            y2.append(np.nanmean(stat_data[1][i][j]))
            y1e.append(sem(stat_data[0][i][j]))
            y2e.append(sem(stat_data[0][i][j]))
        axim[i].errorbar([0.9, 1.9, 2.9],  y1, yerr = y1e, marker='None', c = 'green', linestyle = '-', label='Correct', alpha = 1, lw=2, ecolor='green')
        axim[i].errorbar([1.1, 2.1, 3.1],  y2, yerr = y2e, marker='None', c = 'red', linestyle = '-', label='Error', alpha = 1, lw=2, ecolor='red')
        axim[i].set_title(split_plot_name[i], weight='bold', fontsize=14)
        axim[i].set_xticks([1,2,3], ['Hard', 'Medium', 'Easy'], fontsize=12)
        axim[i].set_xlim([0.5, 3.5])
        axim[i].set_ylabel('Confidence', fontsize=14)
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
            
            slope_without_nan = m_array[corr, feat][~np.isnan(m_array[corr, feat])]
            results = ttest_1samp(slope_without_nan, 0, alternative='two-sided')
            t_value_array[corr, feat] = results.statistic
            p_value_array[corr, feat] = results.pvalue

    y_annotate_pos = [[2.6, 2.65, 2.6, 2.65], [2.1, 2.25, 2.05, 2.5]]
    color=['green', 'red']
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


def plot_foldedX_rt(data, path):
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
                stat_data[0][i][j] = plot_data[plot_data['exp_cond'] == split_plot[i][j]]['c_rt'][::4].to_numpy()
                stat_data[1][i][j] = plot_data[plot_data['exp_cond'] == split_plot[i][j]]['ic_rt'][::4].to_numpy()
            else:
                stat_data[0][i][j] = plot_data[plot_data['exp_cond'] == split_plot[i][j]]['c_rt'].to_numpy()
                stat_data[1][i][j] = plot_data[plot_data['exp_cond'] == split_plot[i][j]]['ic_rt'].to_numpy()
            y1.append(np.average(stat_data[0][i][j]))
            y2.append(np.average(stat_data[1][i][j]))
            y1e.append(sem(stat_data[0][i][j]))
            y2e.append(sem(stat_data[0][i][j]))
        axim[i].errorbar([0.9, 1.9, 2.9],  y1, yerr = y1e, marker='None', c = 'green', linestyle = '-', label='Correct', alpha = 1, lw=2, ecolor='green')
        axim[i].errorbar([1.1, 2.1, 3.1],  y2, yerr = y2e, marker='None', c = 'red', linestyle = '-', label='Error', alpha = 1, lw=2, ecolor='red')
        axim[i].set_title(split_plot_name[i], weight='bold', fontsize=14)
        axim[i].set_xticks([1,2,3], ['Hard', 'Medium', 'Easy'], fontsize=12)
        axim[i].set_xlim([0.5, 3.5])
        axim[i].set_ylim([600, 1800])
        axim[i].set_ylabel('RT', fontsize=14)
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

    # y_annotate_pos = [[2.6, 2.65, 2.6, 2.65], [2.1, 2.25, 2.05, 2.5]]
    # color=['green', 'red']
    # for feat in range(4):
    #     for correct in range(2):
    #         t_value = t_value_array[correct][feat]
    #         p_value = p_value_array[correct][feat]
    #         # Format p-value
    #         if p_value >= 0.001:
    #             annotation = r"$p = {:.2f}$".format(p_value)
    #         else:
    #             power = int(np.floor(np.log10(p_value)))
    #             coefficient = p_value / (10 ** power)
    #             annotation = r"$p = {:.2f} \times 10^{{{}}}$".format(coefficient, power)
    #         axim[feat].text(3.2, y_annotate_pos[correct][feat], annotation, fontsize=8, ha='right', color=color[correct])


    plt.suptitle('Experiment 1', fontsize=18, fontweight='bold')
    # axim[0].set_ylim(1.8, 3.4)
    # plt.tight_layout()
    axim[0].legend(frameon=True, fontsize=10, loc='upper left')
    plot_name = path / 'rt_folded_x_plot.png'
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
    markers = ['o', 's', 'D', '*']

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
                            color=plate[i], marker=markers[i], zorder=2,
                            # xerr=xerr[k], yerr=yerr[k],
                            markersize=8, alpha=0.8,
                            )
            else:
                plt.errorbar(x_axis[k], y_axis[k], color='black', marker='o', zorder=3,
                            label='Baseline' if i == 3 else None,
                            # xerr=xerr[k]*2, yerr=yerr[k]*2,   # correct for inflated sem
                            markersize=5, alpha=1,
                            )

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
    plt.close()


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
            m, c = fit_to_line(x_axis, y_axis)
            slope[s][i] = m
            c_array[s][i] = c
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
        # print(test)
        x_pos = 0.5 * (label.index(test[0]) + label.index(test[1]))
        y_pos = y_pos_array[i]

        # Determine if p is over power 3
        if test[3] < 1e-3:
            power = int(np.floor(np.log10(test[3])))
            coefficient = test[3] / (10 ** power)
            annotation = r"$p = {:.3f} \times 10^{{{}}}$".format(coefficient, power)
        else:
            annotation = r"$p = {:.3f}$".format(test[3])

        if test[3] < 0.05:
            alpha = 1
        else:
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

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([0,1,2], label, fontsize=16)
    plt.ylabel('Slope', fontsize=16)
    plt.title('Slope analysis', fontsize=18, fontweight='bold')
    plot_name = path / 'acc_conf_slope.png'
    plt.savefig(plot_name, format='png', dpi=384, transparent=True)
    print(plot_name)
    plt.close()


def split_half_test(data, path):
    half_pt = data.trial_no.max() // 2
    first_half, second_half = data[data.trial_no <= half_pt], data[data.trial_no > half_pt]

    for i, half_data in enumerate([first_half, second_half]):
        pro_data = process(half_data)
        pro_data = assign_condition(pro_data)
        split_path = path / f'split_{i}'
        split_path.mkdir(parents=True, exist_ok=True)
        graph(pro_data, split_path)



def trial_level_modelling(data, path):
    stim_man = ['size', 'duration', 'noise', 'tilt']

    for i, stim in enumerate(stim_man):
        cond = 2 * i + 1
        model_data = data[data.stim_condition.isin([cond, cond + 1, 9])] 

        model_data.subject_ID = model_data.subject_ID.astype(str)
        model_data.stim_condition = model_data.stim_condition.astype(str)

        # Set condition 9 as the first (reference) level
        model_data.stim_condition = pd.Categorical(
            model_data.stim_condition,
            categories=['9', str(cond), str(cond + 1)],
            ordered=False
        )

        old_out = sys.stdout
        sys.stdout = mystdout = io.StringIO()

        model = Lmer('correct ~ conf_resp * stim_condition + (1 + conf_resp * stim_condition | subject_ID)', 
                     data=model_data,
                     family='binomial',
                     )
        result = model.fit()

        sys.stdout = old_out
        output = mystdout.getvalue()

        with open(path, "a") as f:
            f.write(f"===== Model for {stim} =====\n")
            f.write(output)
            f.write(str(result))
            f.write("\n\n")
            f.write(str(model.coefs))
            f.write('--------------------------------------------------------------------------------\n')
            f.write("\n\n")



def graph(data, path):
    print("\nCreating these graphs ......")

    # Figure 2a,2b
    plot_acc_conf_scatter(data, path)
    plot_acc_conf_slope(data, path)

    # Figure 2c
    plot_difference(data, path)

    # Figure 2d
    plot_z_transform_difference(data, path)

    # Figure 5
    plot_foldedX(data, path)

    # # Supplementary (RT)
    # plot_difference_rt(data, path)
    # plot_foldedX_rt(data, path)


def main():
    in_data_path, out_pro_path, stat_path, graph_path = manage_path()
    if run_process:
        print("\nProcessing individuals ......")
        # read and process
        data = read(in_data_path)
        pro_data = process(data)

        # exclude subjects
        exclude_list = find_exclude(pro_data)
        data = exclude(data, exclude_list)
        pro_data = exclude(pro_data, exclude_list)

        # output all to csv
        data.to_csv(in_data_path, sep=',', index=False)
        pro_data.to_csv(out_pro_path, sep=',', index=False)

        print('--------------------------------------------------------------------------------')
        print('Processing Completed.')
    else:
        data = read(in_data_path)
        pro_data = read(out_pro_path)
        print("Processed data read from " + str(out_pro_path))

    split_half_test(data, graph_path)
    # trial_level_modelling(data, stat_path)
    # pro_data = assign_condition(pro_data)
    # graph(pro_data, graph_path)
    print("--------------------------------------------------------------------------------")
    print('ALL DONE')


if __name__ == "__main__":
    main()
