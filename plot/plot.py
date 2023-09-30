import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

def winning_ticket_gap(unpruned_acc, avg_acc_list, num_sparsities):
    x = np.arange(num_sparsities)
    x_density = 1-  0.8 ** x
    print("winning ticket acc gap:",np.max(avg_acc_list) - unpruned_acc)
    print("Achived at the sparsity of {}".format(x_density[np.argmax(avg_acc_list)]))
    return 0 

def last_winning_ticket_sparsity(unpruned_acc, avg_acc_list, num_sparsities, err_list):
    x = np.arange(num_sparsities)
    x_density = 1-  0.8 ** x
    print("last winning ticket sparsity:", x_density[avg_acc_list - unpruned_acc >= 0])
    print("Accuracy:", avg_acc_list[avg_acc_list - unpruned_acc >= 0] )
    print("Std:", err_list[avg_acc_list - unpruned_acc >= 0] )
    return 0

def extract_y_err(y_dense, performance_str):
    if '(' in performance_str or ')' in performance_str:
        y = re.findall('(\d+\.\d+)(?=\()', performance_str)
        err = re.findall('(?<=\()(\d+\.\d+)', performance_str)
        y = [float(num) for num in y]
        err = [float(num) for num in err]       
    else:
        y = [float(i) for i in performance_str.split()]
        err = [0] * len(y)

    return np.array(y), np.array(err)

if __name__ == "__main__":
    # num = 14
    # # x_grid = np.array(range(num))
    # step = 1
    # index = np.arange(0, num, step)
    # x = np.arange(num)[index]
    # x_density = 100 - 100 * (0.8 ** x)
    # x_grid = x_density
    # x_density_list = ['{:.2f}'.format(value) for value in x_density]

    # num = 14
    # x_grid = np.array(range(num))
    # step = 1
    # index = np.arange(0, num, step)
    # x = np.arange(num)[index]
    # x_density = 100 - 100 * (0.8 ** x)
    # x_grid = x_density
    # x_density_list = ['{:.2f}'.format(value) for value in x_density]

    # y_PFTT_time = np.insert(np.array([180 for i in range(num - 1)]), 0, 0)
    # y_BiP_time = np.insert(np.array([225 for i in range(num - 1)]), 0, 0)
    # y_hydra_time = np.insert(np.array([136 for i in range(num - 1)]), 0, 0)
    # y_IMP_time = np.array([115.2 * i for i in range(num)])
    # y_OMP_time = np.insert(np.array([115.2 for i in range(num - 1)]), 0, 0)
    # y_Grasp_time = np.insert(np.array([120 for i in range(num - 1)]), 0, 0)
    
    # 7, 11; 9, 20
    title = 'Forth version Loss'
    num, imp_num = 16, 11
    y_baseline = 83.37
    y_min, y_max = 72, 84
    
    # 10, 20
    x_sparsity_list = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5])
    # x_sparsity_list = np.array([0, 0.5, 0.8, 0.9, 1, 1.1, 1.2, 1.5])
    x_grid = x_sparsity_list
    x_IMP_sparsity_list = np.array([0, 20.00, 36.00, 48.80, 59.00, 67.20, 73.80, 79.03, 83.22, 86.58, 89.26, 91.41, 93.13, 94.50, 95.60, 96.50, 97.75, 98.20, 98.56, 98.85][:imp_num])
    
    # y_IMP = np.array([y_dense,73.01,72.72,72.36,71.97,71.18,70.49,69.52,68.43,67.17,65.89])
    # y_IMP_err = np.array([0,0.10,0.16,0.10,0.17,0.39,0.21,0.25,0.39,0.33,0.18])

    multi_2single       ='66.51 67.89 68.30 68.76 69.37 69.63 70.31 69.87 70.04 70.58 71.18 70.62 70.41 70.27 70.44 70.41 '
    multi_2ensemble     ='70.60 71.95 71.63 71.76 71.95 72.07 72.50 72.20 72.5 72.86 72.69 72.13 72.06 71.54 71.71 71.65 '
    multi_3single       ='66.82 68.76 69.76 70.48 70.82 71.02 70.89 71.23 71.43 71.00 70.71 70.76 70.82 70.75 70.69 70.41 '
    multi_3ensemble     ='72.91 73.65 73.56 73.75 73.61 73.90 73.76 73.36 73.49 72.66 72.3 71.99 72.42 71.89 71.61 71.25 '
    consensus_2single   ='66.12 67.48 68.71 69.31 69.20 69.77 69.73 70.05 69.71 70.22 70.15 69.85 70.13 69.90 70.27 69.91 '
    consensus_2ensemble ='70.12 71.18 71.97 72.35 72.18 72.01 71.99 72.40 71.91 72.17 71.74 71.73 71.81 71.58 71.53 71.05 '
    
    forth='80.93 77.94 76.07 74.90 74.04 73.66 72.85 72.54 72.48 72.28 72.11 72.66 72.43 73.46 73.66 73.34'
    
    y_multi_2single, y_multi_2single_err = extract_y_err(y_baseline, multi_2single)
    y_multi_2ensemble, y_multi_2ensemble_err = extract_y_err(y_baseline, multi_2ensemble)
    y_multi_3single, y_multi_3single_err = extract_y_err(y_baseline, multi_3single)
    y_multi_3ensemble, y_multi_3ensemble_err = extract_y_err(y_baseline, multi_3ensemble)
    y_consensus_2single, y_consensus_2single_err = extract_y_err(y_baseline, consensus_2single)
    y_consensus_2ensemble, y_consensus_2ensemble_err = extract_y_err(y_baseline, consensus_2ensemble)

    y_forth, y_forth_err = extract_y_err(y_baseline, forth)

    y_best = np.min(y_forth)

    # print("IMP winning ticket gap:")
    # winning_ticket_gap(y_IMP[0], y_IMP, num)
    # print("Grasp winning ticket gap:")
    # winning_ticket_gap(y_IMP[0], y_Grasp, num)
    # print("OMP winning ticket gap:")
    # winning_ticket_gap(y_IMP[0], y_OMP, num)
    # print("Hydra winning ticket gap:")
    # winning_ticket_gap(y_IMP[0], y_hydra, num)
    # print("BiP winning ticket gap:")
    # winning_ticket_gap(y_IMP[0], y_BiP, num)

    # print("last IMP winning ticket:")
    # last_winning_ticket_sparsity(y_IMP[0], y_IMP, num, y_IMP_err)
    # print("last Grasp winning ticket gap:")
    # last_winning_ticket_sparsity(y_IMP[0], y_Grasp, num, y_grasp_err)
    # print("last OMP winning ticket gap:")
    # last_winning_ticket_sparsity(y_IMP[0], y_OMP, num, y_OMP_err)
    # print("last Hydra winning ticket gap:")
    # last_winning_ticket_sparsity(y_IMP[0], y_hydra, num, y_hydra_err)
    # print("last BiP winning ticket gap:")
    # last_winning_ticket_sparsity(y_IMP[0], y_BiP, num, y_BiP_err)


    x_label = "Alpha"
    y_label = "PPL (%)"

    # Canvas setting
    width = 14
    height = 12
    plt.figure(figsize=(width, height))

    sns.set_theme()
    plt.grid(visible=True, which='major', linestyle='-', linewidth=4)
    plt.grid(visible=True, which='minor')
    plt.minorticks_on()
    plt.rcParams['font.serif'] = 'Times New Roman'

    markersize = 20
    linewidth = 2
    markevery = 1
    fontsize = 50
    alpha = 0.7

    # Color Palette
    best_color = 'green'
    best_alpha = 1.0
    baseline_color = 'black'
    baseline_alpha = 1.0

    multi_2single_color = 'red'
    multi_2single_alpha = 0.9
    multi_2ensemble_color = 'green'
    multi_2ensemble_alpha = 0.9
    multi_3single_color = 'hotpink'
    multi_3single_alpha = alpha
    multi_3ensemble_color = 'blue'
    multi_3ensemble_alpha = alpha - 0.1
    consensus_2single_color = 'darkorange'
    consensus_2single_alpha = alpha
    consensus_2ensemble_color = 'purple'
    consensus_2ensemble_alpha = alpha

    forth_color = 'darkorange'
    forth_alpha = alpha
    # SNIP_color = 'gold'
    # SNIP_alpha = alpha
    # Random_color = 'violet'
    # Random_alpha = alpha


    fill_in_alpha = 0.2

    # plt.rcParams['font.sans-serif'] = 'Times New Roman'
    # plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    # # Show the minor grid lines with very faint and almost transparent grey lines
    # plt.minorticks_on()
    # plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


    l_baseline = plt.axhline(y=y_baseline, color=baseline_color, linestyle='--', linewidth=3, label="Baseline")

    # l_multi_2single = plt.plot(x_grid, y_multi_2single, color=multi_2single_color, marker='o', markevery=markevery, linestyle='-',
    #                   linewidth=linewidth,
    #                   markersize=markersize, label="Multi Loss", alpha=multi_2single_alpha)
    # plt.fill_between(x_grid, y_multi_2single - y_multi_2single_err, y_multi_2single + y_multi_2single_err, color=multi_2single_color, alpha=fill_in_alpha)

    # l_multi_2ensemble = plt.plot(x_grid, y_multi_2ensemble, color=multi_2ensemble_color, marker='o', markevery=markevery, linestyle='-', linewidth=linewidth,
    #                 markersize=markersize, label="Multi Loss", alpha=multi_2ensemble_alpha)
    # plt.fill_between(x_grid, y_multi_2ensemble - y_multi_2ensemble_err, y_multi_2ensemble + y_multi_2ensemble_err, color=multi_2ensemble_color, alpha=fill_in_alpha)

    # l_multi_3single = plt.plot(x_grid, y_multi_3single, color=multi_3single_color, marker='o', markevery=markevery, linestyle='-', linewidth=linewidth,
    #                 markersize=markersize, label="Multi Loss", alpha=multi_3single_alpha)
    # plt.fill_between(x_grid, y_multi_3single - y_multi_3single_err, y_multi_3single + y_multi_3single_err, color=multi_3single_color, alpha=fill_in_alpha)

    # l_multi_3ensemble = plt.plot(x_grid, y_multi_3ensemble, color=multi_3ensemble_color, marker='o', markevery=markevery, linestyle='-',
    #                   linewidth=linewidth,
    #                   markersize=markersize, label="Multi Loss", alpha=multi_3ensemble_alpha)
    # plt.fill_between(x_grid, y_multi_3ensemble - y_multi_3ensemble_err, y_multi_3ensemble + y_multi_3ensemble_err, color=multi_3ensemble_color, alpha=fill_in_alpha)

    # l_consensus_2single = plt.plot(x_grid, y_consensus_2single, color=consensus_2single_color, marker='o', markevery=markevery, linestyle='-',
    #                 linewidth=linewidth,
    #                 markersize=markersize + 4, label="Consensus Loss", alpha=consensus_2single_alpha)
    # plt.fill_between(x_grid, y_consensus_2single - y_consensus_2single_err, y_consensus_2single + y_consensus_2single_err, color=consensus_2single_color, alpha=fill_in_alpha)

    # l_consensus_2ensemble = plt.plot(x_grid, y_consensus_2ensemble, color=consensus_2ensemble_color, marker='o', markevery=markevery, linestyle='-',
    #                 linewidth=linewidth,
    #                 markersize=markersize + 4, label="Consensus Loss", alpha=consensus_2ensemble_alpha)
    # plt.fill_between(x_grid, y_consensus_2ensemble - y_consensus_2ensemble_err, y_consensus_2ensemble + y_consensus_2ensemble_err, color=consensus_2ensemble_color, alpha=fill_in_alpha)


    l_forth = plt.plot(x_grid, y_forth, color=forth_color, marker='o', markevery=markevery, linestyle='-',
                      linewidth=linewidth,
                      markersize=markersize, label="Forth Loss", alpha=forth_alpha)
    plt.fill_between(x_grid, y_forth - y_forth_err, y_forth + y_forth_err, color=forth_color, alpha=fill_in_alpha)


    lbest = plt.axhline(y=y_best, color=best_color, linestyle='--', linewidth=3, alpha=best_alpha,
                        label="Best")
    # lPFTT = plt.plot(x_grid, y_PFTT, color=PFTT_color, marker='*', markevery=markevery, linestyle='-',
    #                  linewidth=linewidth,
    #                  markersize=markersize + 4, label="PFTT", alpha=PFTT_alpha)
    # plt.fill_between(x_grid, y_PFTT - y_PFTT_err, y_PFTT + y_PFTT_err, color=PFTT_color, alpha=fill_in_alpha)

    plt.ylim([y_min, y_max])
    plt.xlim(0, 1.5)

    plt.legend(fontsize=fontsize - 20, loc='upper right', fancybox=True, shadow=True, framealpha=1.0, borderpad=0.3)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xticks(x_grid, x_sparsity_list, rotation=0, fontsize=fontsize)
    plt.xscale("linear")
    plt.yticks(fontsize=fontsize)

    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    # plt.twinx()
    # y_time_label = "Time Consumption (min)"
    # linewidth = 2
    # time_linestyle = '-.'
    # tIMP = plt.plot(x_grid, y_IMP_time, color=IMP_color, alpha=IMP_alpha, label="IMP", linestyle=time_linestyle,
    #                 linewidth=linewidth)
    # tBiP = plt.plot(x_grid, y_BiP_time, color=BiP_color, alpha=BiP_alpha, label="BiP", linestyle=time_linestyle,
    #                  linewidth=linewidth)
    # tPFTT = plt.plot(x_grid, y_PFTT_time, color=PFTT_color, alpha=PFTT_alpha, label="PFTT", linestyle=time_linestyle,
    #                  linewidth=linewidth)
    # tHydra = plt.plot(x_grid, y_hydra_time, color=hydra_color, alpha=hydra_alpha, label="Hydra Global",
    #                   linestyle=time_linestyle, linewidth=linewidth)
    # tGrasp = plt.plot(x_grid, y_Grasp_time, color=Grasp_color, alpha=Grasp_alpha, label="Grasp",
    #                   linestyle=time_linestyle, linewidth=linewidth)
    # tOMP = plt.plot(x_grid, y_OMP_time, color=OMP_color, alpha=OMP_alpha + 0., label="OMP", linestyle=time_linestyle,
    #                 linewidth=linewidth)
    #
    # plt.xlabel(x_label, fontsize=fontsize)
    # plt.ylabel(y_time_label, fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
    # plt.ylim(0, (int(max(y_IMP_time) / 100) + 1) * 100)
    plt.savefig(f"pic/ptb/{title}.pdf")
    plt.show()
    plt.close()










