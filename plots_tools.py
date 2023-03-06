# import xport
import numpy as np
import pandas as pd
# import os
# import json
# import pyperclip
# import json
# import plotly.express as px
# import plotly.io as pio
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

SMALL_SIZE = 14
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

sns.set(style="ticks", font='Times New Roman', font_scale=1.0)
matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams['font.family'] = "serif"

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def processing_dist(figsize, dpi,
                    bandwidth, histogram, bins, density, legend, nova_bar, series_info,
                    NOVA_predictions_colors_dict, col_processing_score, legend_kws, hist_kws,
                    nova_df, col_nova_class, xlim, ylim, xlabel, ylabel, rand_y_range,
                    remove_title, file_export, dist_kde_kws, x_axes_range, y_axes_range,
                    NOVA_bar_use_NOVA_color
                    ):
    if dist_kde_kws is None:
        dist_kde_kws = {}
    plt.figure(figsize=figsize, dpi=dpi)
    plt.rcParams["patch.force_edgecolor"] = True

    if histogram:
        # plt.hist(x, bins=30, range=(-2,2), color=(0,.6,0), label='Histogram', normed=True)
        # kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40, ec="k")
        for series_dict in series_info:
            x = series_dict["df"][col_processing_score]
            plt.hist(x, bins=bins, color=series_dict["c"], linewidth=1, edgecolor=series_dict["c"],
                     histtype=['step', 'stepfilled'][0],  # ec="k", #edgecolor='black'
                     alpha=series_dict["alpha"], normed=True, label=series_dict["name"], **hist_kws
                     )

        # if legend:
        #     if "loc" not in legend_kws:
        #         legend_kws["loc"] = "best"
        #     plt.legend(**legend_kws)

    if density:
        for series_dict in series_info:
            x = series_dict["df"][col_processing_score]

            if 'c' in series_dict:
                c = series_dict['c']
            else:
                c = None

            if bandwidth is not None:
                dist_kde_kws['bw_adjust'] = bandwidth

            ax = sns.distplot(
                x, hist=False, kde=True,
                # label="{} ({:,} from {:,} products)".format(store, len(df),len(all_df[all_df["store"]==store])),
                label="{}".format(series_dict['name']),
                color=c, kde_kws=dist_kde_kws
            )

        pass

    if nova_bar:
        nova_df['y'] = np.random.uniform(rand_y_range[0], rand_y_range[1], len(nova_df))

        nova_df['NOVA class'] = "NOVA " + (nova_df[col_nova_class] + 1).astype(str)
        nova_df = nova_df.sort_values(by=col_nova_class)

        if NOVA_bar_use_NOVA_color is True:

            sns.scatterplot(data=nova_df, x=col_processing_score, y="y", hue="NOVA class",
                            palette=NOVA_predictions_colors_dict,
                            legend='full' if legend else False,  # legend` must be 'brief', 'full', or False
                            alpha=0.4, linewidth=0)
        else:

            # cmap = sns.cubehelix_palette(as_cmap=True)
            cmap = sns.color_palette("YlOrBr", as_cmap=True)
            # https://stackoverflow.com/questions/39735147/how-to-color-matplotlib-scatterplot-using-a-continuous-value-seaborn-color
            sns.scatterplot(data=nova_df, x=col_processing_score, y="y",
                            # hue=col_processing_score,
                            # palette=sns.color_palette("YlOrBr", as_cmap=True),
                            legend='full' if legend else False,  # legend` must be 'brief', 'full', or False
                            alpha=0.4, linewidth=0,
                            cmap=cmap, c=nova_df[col_processing_score]
                            )

            pass

    ax = plt.axes()

    if legend:
        if "loc" not in legend_kws:
            legend_kws["loc"] = "best"
        # plt.legend(prop={'size': 1}, **legend_kws)

        handles, labels = ax.get_legend_handles_labels()
        # print(len(handles))
        if len(handles) == 5:
            ax.legend(handles=handles[1:], labels=labels[1:])
        else:
            ax.legend(handles=handles, labels=labels)

        # if len(handles) == 8:
        #     ax.legend(handles=handles[0:3] + handles[4:], labels=labels[0:3] + labels[4:], **legend_kws)
        # elif len(handles) == 6:
        #     ax.legend(handles=handles[2:], labels=labels[2:], **legend_kws)

    ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    title = ""
    if density:
        title = 'Banwidth {}'.format(bandwidth)

    if histogram:
        title += ' Bins {} Range {}'.format(bins, hist_kws['range'])

    title += ' NOVA Bar rand-y-range: {}'.format(rand_y_range)

    print(title)
    if remove_title is False:
        plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if y_axes_range is not None:
        plt.yticks(y_axes_range, rotation=00)

    # Remove negative ticks
    num_negative_ticks = len([1 for t in ax.get_yticks() if t < 0])
    for i in range(0, num_negative_ticks):
        ax.yaxis.get_major_ticks()[i].draw = lambda *args: None

    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position(("data",0))

    # x_axes_range = np.arange(-1.0, 1.5, 0.125)
    if x_axes_range is None:
        x_axes_range = np.arange(xlim[0], xlim[1], 0.5)
    # x_axes_range = np.arange(0, 1.01, 0.2)

    plt.xticks(x_axes_range, rotation=00)

    ax2 = ax.twiny()
    ax2.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(("data", 0))

    # ax2.set_xticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    ax2.set_xticks(x_axes_range)
    # ax2.set_xticklabels(newlabel)

    # ax2.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    # ax2.spines['bottom'].set_position(('outward', 36))
    # ax2.set_xlabel('Temperature [K]')
    ax2.set_xlim(ax.get_xlim())
    # ax2.set_xlabel(xlabel)

    ax2.set_xticklabels([])
    ax2.xaxis.set_tick_params(size=4)

    plt.savefig(file_export, bbox_inches='tight')

    pass


def plot_dists_custom(series_dict, plot_title, xaxis_title, kde_kws, figsize, dpi):
    plt.figure(figsize=figsize, dpi=dpi)  # 600
    #     plt.xlim(-1.5, 1.5)

    for series_name, series_vars in series_dict.items():
        if len(series_dict) == 1:
            series_name = ""

        #         quotient = series_vars["df"].values
        #         quotient = [x - 0.15 for x in quotient]
        #         min_val = -1.5
        #         max_val = 1.0
        #         quotient = [(x-min_val)/(max_val-min_val) for x in quotient]
        #         quotient = [(x-min(quotient))/(max(quotient)-min(quotient)) for x in quotient]

        ax = sns.distplot(series_vars["df"], hist=False, kde=True,
                          label=series_name,
                          color=series_vars["color"], kde_kws=kde_kws
                          )

        sns.scatterplot(data=tips, x="total_bill", y="tip")

        sns.rugplot(data=series_vars["df"])

    ax.set_title(plot_title)

    if xaxis_title is not None:
        plt.xlabel(xaxis_title)

    if len(series_dict) > 1:
        plt.legend()

    plt.show()
    plt.clf()

    return ax


if __name__ == "__main__":
    food_only = pd.read_csv("temp/2015_cons_data.csv")

    ax = plot_dists_custom(
        series_dict={
            "Top 100": {"df": food_only['Processing index 11P'][:100], "color": "red"},  ##e41a1c
            # "Top 1000": {"df": food_only['Processing index 11P'][:1000], "color": "blue"},  ##e41a1c
            # "All": {"df": food_only['Processing index 11P'], "color": "grey"},
        },
        plot_title=["", f"Most Consumed Foods (No Beverage) for NHANES 2015"][1],
        xaxis_title="Processing Score", kde_kws={}  # {'clip': (-1.4 , 1.5)}
        , figsize=(8, 6), dpi=150
        #     ,figsize=(8,6), dpi=600
    )
