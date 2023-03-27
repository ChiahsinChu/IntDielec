import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn import metrics

from .style import use_style

use_style("pub")


def ax_setlabel(ax, xlabel, ylabel, **kwargs):
    # set axis label
    ax.set_xlabel(xlabel, **kwargs)
    ax.set_ylabel(ylabel, **kwargs)


def plot_rmse(x, y, xlabel, ylabel, **kwargs):
    """
    plot scatter/ref line
    return rmse
    """
    x = np.array(x)
    y = np.array(y)

    rmse = np.sqrt(metrics.mean_squared_error(x, y))

    fig, ax = plt.subplots(figsize=[4, 4])
    ax_rmse(ax, x, y)
    ax_setlabel(ax, xlabel, ylabel, **kwargs)

    return fig, ax, rmse


def ax_rmse(ax, x, y):
    # scatter
    ax.scatter(x, y, color='steelblue', alpha=0.2)
    # ref line
    ref = np.arange(x.min(), x.max(), (x.max() - x.min()) / 100)
    ax.plot(ref, ref, color='firebrick', lw=1.5)


def plot_bin_stats(x, y, xlabel, ylabel, bins=None):
    """
    plot scatter/ref line
    return rmse
    """
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    if bins is None:
        bins = len(x) // 10

    fig, ax = plt.subplots(figsize=[6, 4], dpi=200)
    ax_bin_stats(ax, x, y, bins=bins)
    ax_setlabel(ax, xlabel, ylabel)
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    return fig, ax


def ax_bin_stats(ax, x, y, bins):
    # ref line (y = 0)
    ax.axhline(y=0, color='gray')
    # mean
    bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, bins=bins)
    bin_means[np.isnan(bin_means)] = 0.0
    bin_centers = bin_edges[1:] - (bin_edges[1] - bin_edges[0]) / 2
    ax.plot(bin_centers, bin_means, color='black', lw=1.5, label='mean')
    # max/min
    bin_maxs, bin_edges, binnumber = stats.binned_statistic(x,
                                                            y,
                                                            statistic='max',
                                                            bins=bins)
    bin_mins, bin_edges, binnumber = stats.binned_statistic(x,
                                                            y,
                                                            statistic='min',
                                                            bins=bins)
    bin_maxs[np.isnan(bin_maxs)] = 0.0
    bin_mins[np.isnan(bin_maxs)] = 0.0
    ax.fill_between(bin_centers,
                    bin_mins,
                    bin_maxs,
                    color='silver',
                    label='[min, max]')
    # std
    bin_stds, bin_edges, binnumber = stats.binned_statistic(x,
                                                            y,
                                                            statistic='std',
                                                            bins=bins)
    bin_stds[np.isnan(bin_stds)] = 0.0
    ax.fill_between(bin_centers,
                    bin_means - bin_stds,
                    bin_means + bin_stds,
                    color='gray',
                    label='[mean-std, mean+std]')


def plot_colormap_lines(xs, ys, legends, xlabel, ylabel, colormap='GnBu'):
    """
    TBC
    """
    fig, ax = plt.subplots(figsize=[6, 4], dpi=200)
    ax_colormap_lines(ax, xs, ys, legends, colormap)
    ax_setlabel(ax, xlabel, ylabel)
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    return fig, ax


def ax_colormap_lines(ax,
                      xs,
                      ys,
                      labels,
                      scale=(0., 1.),
                      colormap="GnBu",
                      **kwargs):
    cm_scales = (np.array(labels) - scale[0]) / (scale[1] - scale[0])
    for x, y, label, cm_scale in zip(xs, ys, labels, cm_scales):
        ax.plot(x,
                y,
                color=plt.get_cmap(colormap)(cm_scale),
                label=label,
                **kwargs)
    ax.set_xlim(np.min(x), np.max(x))


"""
def ax_set_fe_text(ax, text_y, label, color, text_x=800):
    ax.text(x=text_x, y=text_y, s=label, 
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=40, color=color)

def ax_set_fe_frame_1(ax):
    ax.set_xlabel('Temperature / K', fontsize=28)
    ax.set_ylabel('Free Energy / eV', fontsize=28)
    for item in ax.spines:
        ax.spines[item].set_linewidth(2)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.tick_params(length=6, width=2)
    ax.tick_params(which='minor', length=3, width=2)
    ax.set_ylim(-2.5, 1)
    ax.set_xlim(250,1050)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(24)
        
def ax_set_fe_frame_2(ax):
    ax2.axvspan(450, 600, facecolor='#FFCEC1')
    ax.axvspan(600, 750, facecolor='#C5E7E9')

    ax.text(x=650, y=-0.05, s=r"$\Delta_r S$", 
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=44, color=sns.color_palette("tab10")[0])

    ax.text(x=100, y=-0.24, s=r"Solid", 
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=26)
    ax.text(x=600, y=-0.24, s=r"Coexistence", 
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=26)
    ax.text(x=1100, y=-0.24, s=r"Liquid", 
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=26)

    ax.set_xlim(-200,1400)
    ax.set_ylim(-0.25,0.65)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel('Temperature', fontsize=28)
    ax.set_ylabel('Entropy', fontsize=28)

    for item in ax.spines:
        ax.spines[item].set_linewidth(2)
        
def ax_set_fe_frame_3(ax):
    ax.axvspan(450, 550, facecolor='#FFCEC1')
    ax.axvspan(550, 700, facecolor='#C5E7E9')
    ax.set_xlabel('Temperature / K', fontsize=30)
    ax.set_ylabel(r'Entropy / $\mathrm{J \cdot mol^{-1} \cdot K^{-1}}$', fontsize=30)

    at1 = AnchoredText("i", prop=dict(size=48), loc='upper left', frameon=False, pad=0)
    ax.add_artist(at1)
    for label in ax.get_yticklabels():
        label.set_fontsize(24)
    for item in ax.spines:
        ax.spines[item].set_linewidth(2)
    ax.set_ylim(-900,200)
    ax.tick_params(length=6, width=2)
    ax.tick_params(which='minor', length=3, width=2)
    ax.xaxis.set_major_locator(MultipleLocator(200))
def ax_set_fe_frame_4(ax):
    ax.axvspan(450, 550, facecolor='#FFCEC1')
    ax.set_xlabel('Temperature / K', fontsize=30)
    ax.set_ylabel(r'$\delta_{rms}$', fontsize=30)
    at2 = AnchoredText("ii", prop=dict(size=48), loc='upper left', frameon=False, pad=0)
    ax.add_artist(at2)
    for label in ax.get_yticklabels():
        label.set_fontsize(24)
    for item in ax.spines:
        ax.spines[item].set_linewidth(2)
    ax.tick_params(length=6, width=2)
    ax.tick_params(which='minor', length=3, width=2)
    ax.set_ylim(0.0,0.33)

def ax_set_fe_frame_5(ax):
    ax.axvspan(500, 700, facecolor='#C5E7E9')
    ax.set_xlabel('Temperature / K', fontsize=30)
    ax.set_ylabel(r'$\delta_{rms}$', fontsize=30)
    at3 = AnchoredText("iii", prop=dict(size=48), loc='upper left', frameon=False, pad=0)
    ax.add_artist(at3)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(24)
    for item in ax.spines:
        ax.spines[item].set_linewidth(2)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.tick_params(length=6, width=2)
    ax.tick_params(which='minor', length=3, width=2)
    ax.set_xlim(250,1050)
    ax.set_ylim(0.0,0.33)
"""
