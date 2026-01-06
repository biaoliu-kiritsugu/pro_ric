import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import glob2 
import seaborn as sns 
import random
# Set up a professional color palette (using seaborn's deep palette)
colors = sns.color_palette('deep')  # Good for colorblind viewers
# Alternatively, use a more distinct palette:
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Define distinct marker styles for each method
# algorithms = ['ric offline', 'ric online', 'reward soups', 'ours_offline', 'ours_online']
algorithms = []
# marker_styles = [
#     ('o', '--'),  # circle with dashed line
#     ('s', '-.'),  # square with dash-dot line
#     ('^', ':'),   # triangle with dotted line
#     ('D', '-'),   # diamond with solid line
#     ('p', ':'),   # pentagon with dotted line
#     ('*', '-'),   # star with solid line
#     ('v', '--'),  # triangle down with dashed line
#     ('X', '-.'),  # filled x with dash-dot line
#     ('h', ':'),   # hexagon with dotted line
#     ('8', '-'),   # octagon with solid line
# ]
# '*', 'v', 'X', 'h', '8'
marker_styles = []
markers = ['o', 's', '^', 'D', 'p']
linestyles = ['--', '-.', ':', '-', ':', '-', '--', '-.', ':', '-']
for marker in markers:
    for linestyle in linestyles:
        marker_styles.append((marker, linestyle))
random.shuffle(marker_styles)


# Line widths and marker sizes
linewidth = 2
markersize = 8

def find_pareto_points(obtained_scores, threshold=0.02):
    n = len(obtained_scores)
    if n == 1:
        return obtained_scores
    pareto_index = []
    high_low = np.max(obtained_scores, axis=0) - np.min(obtained_scores, axis=0)
    for i in range(n):
        # if not any(np.all((obtained_scores - obtained_scores[i] - threshold * high_low) > 0.0, axis=1)):
        pareto_index.append(i)

    points = obtained_scores[np.array(pareto_index)]
    arg_index = np.argsort(points[:, 0])
    points = points[arg_index]
    print(points)
    sorted_index = [0]
    remaining_index = np.ones(len(points))
    i = 0
    remaining_index[i] = 0
    while sum(remaining_index):
        distance = ((points[np.where(remaining_index)] - points[i]) ** 2 ).sum(axis=1)
        min_index = np.where(remaining_index > 0)[0][np.argmin(distance)]
        sorted_index.append(min_index)
        i = min_index
        remaining_index[i] = 0
    return points[np.array(sorted_index)]

def plot_points(dir, label, shift=[0,0], txt_color='black', normalize_path=None, reverse=True):
    threshold = 0.01
    desired_scores = []
    obtained_scores = []

    paths = [os.path.abspath(path) for path in glob2.glob(os.path.join(dir, '*.csv'))]
    paths += [os.path.abspath(path) for path in glob2.glob(os.path.join(dir, '*', '*.csv'))]

    pref_lis = []
    for path in paths:
        if '.csv' in path:
            full_path = path 
            data = pd.read_csv(full_path)
            if 'ppo' in path and len(paths) <= 5:
                threshold = 0.5
            obtained_scores.append([np.mean(data['obtained_score1']), np.mean(data['obtained_score2'])])
            if 'pref' in path:
                if 'eval_data_pref' in path:
                    pref = path.split('eval_data_pref')[-1].strip().split('_')[0]
                    pref_lis.append(float(pref))

    desired_scores = np.array(desired_scores)
    obtained_scores = np.array(obtained_scores)

    if normalize_path is not None:
        norm_info = np.load(normalize_path)
        norm_info = np.array(norm_info).reshape(2, 2)
        for i in range(2):
            obtained_scores[:, i] = (obtained_scores[:, i] - norm_info[i][0]) / norm_info[i][1] 

    # Get the appropriate marker and line style for this label
    if label in algorithms:
        idx = algorithms.index(label)
        marker, linestyle = marker_styles[idx]
        color = colors[idx]
    else:
        algorithms.append(label)
        idx = algorithms.index(label)
        marker, linestyle = marker_styles[idx]
        color = colors[idx]

    # Plot the points with consistent styling
    plt.scatter(obtained_scores[:, 0] + shift[0], obtained_scores[:, 1] + shift[1], 
                marker=marker, color=color, 
                s=markersize*10, edgecolor='k', linewidth=0.8)

    if len(pref_lis):
        for i in range(len(obtained_scores)):
            plt.annotate('{}'.format(round(pref_lis[i], 1)), 
                        (obtained_scores[i, 0] + shift[0], obtained_scores[i, 1] + shift[1]), 
                        size=8, color=txt_color)

    pareto_points = find_pareto_points(obtained_scores, threshold)
    # pareto_points = obtained_scores
    plt.plot(pareto_points[:, 0] + shift[0], pareto_points[:, 1] + shift[1], 
            color=color,
            marker=marker, linestyle=linestyle, 
            linewidth=linewidth, markersize=markersize, 
            label=label, markeredgecolor='k', markeredgewidth=0.5)

# Create figure with improved settings
plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'serif'  # Use serif fonts for publications
plt.rcParams['font.size'] = 12

name1 = 'summary'
name2 = 'deberta'

# Plot each method with distinct styling
plot_points('/data/liubiao/llm/a800_2/RiC/pro_ric/logs_trl_eval_final/ric_offline_summary_pref1deberta_pro_t0.5_rate10', 'ours offline')
plot_points('/data/liubiao/llm/a800_2/RiC/pro_ric/logs_trl_eval_final/pro_online_summary_pref1deberta_pro_t0.5_rate10_gen60000_shift2_iter4', 'ours online iter4')
plot_points('/data/liubiao/llm/a800_2/RiC/ric/logs_trl_eval/ric_offline_summary_pref1deberta_test', 'ric offline')
plot_points('/data/liubiao/llm/a800_2/RiC/ric/logs_trl_eval/ric_online_summary_pref1deberta_test', 'ric online')
# plot_points('', 'ours')

# Improve axis labels and legend
plt.xlabel('$R_1$ ({})'.format(name1), fontsize=12)
plt.ylabel('$R_2$ ({})'.format(name2), fontsize=12)
plt.legend(fontsize=10, framealpha=0.9, loc='best')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('plot/ours_summary_{}_{}.png'.format(name1, name2), dpi=300, bbox_inches='tight')
# plt.savefig('plot/ours_summary_{}_{}.pdf'.format(name1, name2), dpi=300, bbox_inches='tight')  # Vector format for publications
