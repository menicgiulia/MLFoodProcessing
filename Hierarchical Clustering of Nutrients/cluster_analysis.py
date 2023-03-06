# coding=utf-8
import os
# import pyodbc
import pandas as pd
# import joblib
import numpy as np

import matplotlib.font_manager
import matplotlib.pyplot as plt
import seaborn as sns
# import matplotlib._color_data as mcd
import matplotlib.patches as mpatch

# import plotly.express as px
# import plotly
# import plotly.figure_factory as ff
# import plotly.graph_objs as go
#
# from sklearn.decomposition import PCA
# from sklearn import preprocessing
# from sklearn.manifold import TSNE
# import umap
import json

from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, homogeneity_score, \
    v_measure_score, completeness_score

# import scipy.spatial as sp
# from collections import OrderedDict
# import networkx as nx
# import re

from dynamicTreeCut import cutreeHybrid
from scipy.spatial.distance import pdist
# from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

"""
Static variables and functions
"""


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_hex(rgb):
    rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    result = '#%02x%02x%02x' % tuple(rgb)
    # print(result)
    return result


NOVA_label_to_pred_class = {'NOVA 1': 0, 'NOVA 2': 1, 'NOVA 3': 2, 'NOVA 4': 3}

NOVA_predictions_colors_dict = {
    'NOVA 1': rgb_to_hex(np.array([0.4, 0.7607843137254902, 0.6470588235294118]) * 255),
    'NOVA 2': rgb_to_hex(np.array([0.9882352941176471, 0.5529411764705883, 0.3843137254901961]) * 255),
    'NOVA 3': rgb_to_hex(np.array([0.5529411764705883, 0.6274509803921569, 0.796078431372549]) * 255),
    'NOVA 4': rgb_to_hex(np.array([0.9058823529411765, 0.5411764705882353, 0.7647058823529411]) * 255),
}

NOVA_pred_class_color = {
    0: NOVA_predictions_colors_dict["NOVA 1"],
    1: NOVA_predictions_colors_dict["NOVA 2"],
    2: NOVA_predictions_colors_dict["NOVA 3"],
    3: NOVA_predictions_colors_dict["NOVA 4"]
}

NOVA_truth_colors_dict = {
    "NOVA-Truth 0": rgb_to_hex(np.array([102, 102, 102])),
    "NOVA-Truth 12": rgb_to_hex(np.array([204, 204, 204])),
    "NOVA-Truth 1": rgb_to_hex(np.array([0.4, 0.7607843137254902, 0.6470588235294118]) * 255),
    "NOVA-Truth 2": rgb_to_hex(
        np.array([0.9882352941176471, 0.5529411764705883, 0.3843137254901961]) * 255),
    "NOVA-Truth 3": rgb_to_hex(
        np.array([0.5529411764705883, 0.6274509803921569, 0.796078431372549]) * 255),
    "NOVA-Truth 4": rgb_to_hex(
        np.array([0.9058823529411765, 0.5411764705882353, 0.7647058823529411]) * 255)
}


def get_nut_panel_columns(panel_nut_number):
    cols = None
    panel_nut_number = str(panel_nut_number)

    if panel_nut_number == "12":
        cols = [
            'Protein', 'Total Fat', 'Carbohydrate', 'Sugars, total', 'Fiber, total dietary', 'Calcium', 'Iron',
            'Sodium', 'Vitamin C', 'Cholesterol', 'Fatty acids, total saturated', 'Total Vitamin A']

    if panel_nut_number == "99":
        cols = ['Protein', 'Total Fat', 'Carbohydrate', 'Alcohol', 'Water', 'Caffeine', 'Theobromine',
                'Sugars, total', 'Fiber, total dietary', 'Calcium', 'Iron', 'Magnesium', 'Phosphorus',
                'Potassium', 'Sodium', 'Zinc', 'Copper', 'Selenium', 'Retinol', 'Carotene, beta',
                'Carotene, alpha', 'Vitamin E (alpha-tocopherol)', 'Vitamin D (D2 + D3)',
                'Cryptoxanthin, beta', 'Lycopene', 'Lutein + zeaxanthin', 'Vitamin C', 'Thiamin',
                'Riboflavin', 'Niacin', 'Vitamin B-6', 'Folate, total', 'Vitamin B-12', 'Choline, total',
                'Vitamin K (phylloquinone)', 'Folic acid', 'Folate, food', 'Vitamin E, added',
                'Vitamin B-12, added', 'Cholesterol', 'Fatty acids, total saturated', '4:00', '6:00',
                '8:00', '10:00', '12:00', '14:00', '16:00', '18:00', '18:01', '18:02', '18:03', '20:04',
                '22:6 n-3', '16:01', '18:04', '20:01', '20:5 n-3', '22:01', '22:5 n-3',
                'Fatty acids, total monounsaturated', 'Fatty acids, total polyunsaturated', 'Daidzein',
                'Genistein', 'Glycitein', 'Cyanidin', 'Petunidin', 'Delphinidin', 'Malvidin',
                'Pelargonidin', 'Peonidin', '(+)-Catechin', '(-)-Epigallocatechin', '(-)-Epicatechin',
                '(-)-Epicatechin 3-gallate', '(-)-Epigallocatechin 3-gallate', 'Theaflavin',
                'Thearubigins', 'Eriodictyol', 'Hesperetin', 'Naringenin', 'Apigenin', 'Luteolin',
                'Isorhamnetin', 'Kaempferol', 'Myricetin', 'Quercetin', "Theaflavin-3,3'-digallate",
                "Theaflavin-3'-gallate", 'Theaflavin-3-gallate', '(+)-Gallocatechin', 'Total flavonoids',
                'Total anthocyanidins', 'Total catechins (monomeric flavan-3-ols only)',
                'Total flavan-3-ols', 'Total flavanones', 'Total flavones', 'Total flavonols',
                'Total isoflavones']

    if panel_nut_number == "62":
        cols = ['Protein', 'Total Fat', 'Carbohydrate', 'Alcohol', 'Water', 'Caffeine', 'Theobromine',
                'Sugars, total', 'Fiber, total dietary', 'Calcium', 'Iron', 'Magnesium', 'Phosphorus',
                'Potassium', 'Sodium', 'Zinc', 'Copper', 'Selenium', 'Retinol', 'Carotene, beta',
                'Carotene, alpha', 'Vitamin E (alpha-tocopherol)', 'Vitamin D (D2 + D3)',
                'Cryptoxanthin, beta', 'Lycopene', 'Lutein + zeaxanthin', 'Vitamin C', 'Thiamin',
                'Riboflavin', 'Niacin', 'Vitamin B-6', 'Folate, total', 'Vitamin B-12', 'Choline, total',
                'Vitamin K (phylloquinone)', 'Folic acid', 'Folate, food', 'Vitamin E, added',
                'Vitamin B-12, added', 'Cholesterol', 'Fatty acids, total saturated', '4:00', '6:00',
                '8:00', '10:00', '12:00', '14:00', '16:00', '18:00', '18:01', '18:02', '18:03', '20:04',
                '22:6 n-3', '16:01', '18:04', '20:01', '20:5 n-3', '22:01', '22:5 n-3',
                'Fatty acids, total monounsaturated', 'Fatty acids, total polyunsaturated',
                # Flavonoids
                # 'Daidzein', 'Genistein', 'Glycitein', 'Cyanidin', 'Petunidin', 'Delphinidin', 'Malvidin',
                # 'Pelargonidin', 'Peonidin', '(+)-Catechin', '(-)-Epigallocatechin', '(-)-Epicatechin',
                # '(-)-Epicatechin 3-gallate', '(-)-Epigallocatechin 3-gallate', 'Theaflavin', 'Thearubigins',
                # 'Eriodictyol', 'Hesperetin', 'Naringenin', 'Apigenin', 'Luteolin', 'Isorhamnetin', 'Kaempferol',
                # 'Myricetin', 'Quercetin', "Theaflavin-3,3'-digallate", "Theaflavin-3'-gallate",
                # 'Theaflavin-3-gallate', '(+)-Gallocatechin', 'Total flavonoids', 'Total anthocyanidins',
                # 'Total catechins (monomeric flavan-3-ols only)', 'Total flavan-3-ols', 'Total flavanones',
                # 'Total flavones', 'Total flavonols', 'Total isoflavones'
                ]

    return cols


def draw_color_text_box(overlap, figsize, title, title_font_size=16, title_font_y=1.1, fontfamily=None):
    """
    Source: https://matplotlib.org/tutorials/colors/colors.html
    :param overlap:
    :return:
    """
    # return
    # import matplotlib._color_data as mcd
    # import matplotlib.patches as mpatch

    # overlap = {name for name in mcd.CSS4_COLORS
    #            if "xkcd:" + name in mcd.XKCD_COLORS}
    # overlap = {'p1': 'g', 'p2': 'b', 'p3': 'y', 'p4': 'r'}

    # fig = plt.figure(figsize=[4.8, 2])
    # figsize = [2, 2]
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, y=title_font_y,
                 # fontsize=title_font_size, fontfamily=fontfamily
                 )
    ax = fig.add_axes([0, 0, 1, 1])

    # for j, n in enumerate(sorted(overlap.keys(), reverse=True)):
    for j, key in enumerate(reversed(list(overlap.keys()))):
        weight = None
        #     cn = mcd.CSS4_COLORS[n]
        cn = overlap[key]
        #     xkcd = mcd.XKCD_COLORS["xkcd:" + n].upper()
        #     xkcd = n
        #     if cn == xkcd:
        #         weight = 'bold'

        r1 = mpatch.Rectangle((0, j), 1, 1, color=cn)
        #     r2 = mpatch.Rectangle((1, j), 1, 1, color=xkcd)
        txt = ax.text(
            1,
            # figsize[0] / 2,
            j + 0.5,
            '  ' + str(key),
            va='center',
            # fontsize=10, weight=weight, fontfamily=fontfamily
        )
        ax.add_patch(r1)
        #     ax.add_patch(r2)
        ax.axhline(j, color='k')
        pass

    # ax.text(.5, j + 1.5, 'Color', ha='center', va='center')
    # ax.text(1.5, j + 1.5, 'xkcd', ha='center', va='center')

    ax.set_xlim(0, figsize[0])
    ax.set_ylim(0, j + (figsize[1] / 2) + 0.2)
    ax.axis('off')

    return fig
    pass


def do_clusters(
        data,
        features,
        name,
        NOVA_class_column,
        description_column,
        y_label_description=False,
        save_path=None,
        save_file_name=None,
        open_excel=False,
        save_excel=False,
        figsize=(30, 27),
        fig_dpi=150,
        save_clustermap=True,
        font_scale=1,
        show_color_luts_for_pred_and_NOVA_truth=False,
        lut_cat_dynamic_tree=None,
        cmap="Blues",
        fontfamily=None,
        xticklabels=None,
        pdist_args=None,
        linkage_args=None,
        link=None,
        distances=None,
        remove_heat_map_legend=True,
        dynamic_cut_args=None,
        color_level2_FNDDS_category=False,
        FNDDS_cats_code_label_dict=None
):
    sns.set(font="Times New Roman NN")
    matplotlib.rcParams['font.serif'] = fontfamily
    matplotlib.rcParams['font.family'] = "serif"
    # plt.rcParams["font.weight"] = 'normal'
    # plt.rcParams["font.size"] = 12
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    if y_label_description:
        data = data.reset_index().set_index(description_column)

    if dynamic_cut_args is None:
        dynamic_cut_args = {}
    if pdist_args is None:
        pdist_args = {"metric": "euclidean"}
    if linkage_args is None:
        linkage_args = {"method": "ward"}

    if distances is None:
        distances = pdist(data[features], **pdist_args)

    if link is None:
        link = sch.linkage(distances, **linkage_args)

    clusters = cutreeHybrid(link=link, distM=distances, **dynamic_cut_args)

    num_clusters = len(set(clusters["labels"]))

    print("Number of clusters: {} [deepSplit: {} | pred_class: {}]".format(
        num_clusters, dynamic_cut_args, NOVA_class_column))

    '''UNCOMMENT IN CASE WE WANT TO HAVE FOOD CATEGORIES AS VERTICAL COLUMNS'''
    # if "Food code" in data.columns:
    #     data["categories_level_3"] = data["Food code"].astype(str).map(lambda x: int(x[:3]))
    #     data["categories_level_2"] = data["Food code"].astype(str).map(lambda x: int(x[:2]))
    #     data["categories_level_1"] = data["Food code"].astype(str).map(lambda x: int(x[:1]))
    #
    #     lut_cat_level_3 = dict(
    #         zip(set(data["categories_level_3"]), sns.hls_palette(len(set(data["categories_level_3"])))))
    #     lut_cat_level_2 = dict(
    #         zip(set(data["categories_levemutual_info_scorel_2"]),
    #             sns.hls_palette(len(set(data["categories_level_2"])), l=0.4, s=0.8)))
    #     lut_cat_level_1 = dict(
    #         zip(set(data["categories_level_1"]), sns.hls_palette(len(set(data["categories_level_1"])), l=0.71, s=1.0)))
    #
    #     cat_level_3_row_colors = data["categories_level_3"].map(lut_cat_level_3)
    #     cat_level_2_row_colors = data["categories_level_2"].map(lut_cat_level_2)
    #     cat_level_1_row_colors = data["categories_level_1"].map(lut_cat_level_1)
    #     pass

    data["DynamicTreeCut"] = clusters["labels"]

    if lut_cat_dynamic_tree is None:
        lut_cat_dynamic_tree = dict(
            zip(set(data["DynamicTreeCut"]), sns.hls_palette(len(set(data["DynamicTreeCut"])), l=0.5, s=0.7)))

    cat_dynamic_tree = data["DynamicTreeCut"].map(lut_cat_dynamic_tree)

    if save_file_name is None:
        file_name = "{}/DTC_{}".format(
            save_path,
            name.replace(",", "").replace("\n", "").replace("{", "").replace("}", "").replace("'", "")
                .replace(": ", "_").replace("&", "_").replace("-", "_").replace("  ", "_")
                .replace(" _ ", "_").replace(" ", "_")
            # NOVA_class_column.replace("<", "(").replace(">", ")")
        )
    else:
        file_name = "{}/DTC_{}".format(save_path, save_file_name)

    format = "png"

    paths_figures = {
        "NOVA_truth": file_name + "_NOVA_truth_labels." + format,
        "NOVA_Predictions": file_name + "_NOVA_labels." + format,
        "FNDDS_level2_categories": file_name + "_FNDDS_Level2_Cats." + format,
        "clusters_labels": file_name + "_clusters_labels." + format,
        "cluster_map": file_name + "_clustermap." + format,
    }

    # region Show Lut

    if show_color_luts_for_pred_and_NOVA_truth:
        NOVA_label_colors_fig = draw_color_text_box(
            NOVA_predictions_colors_dict, figsize=[2, 2], title="Pred NOVA",
            fontfamily=fontfamily
        )
        NOVA_label_colors_fig.savefig(paths_figures["NOVA_Predictions"], dpi=fig_dpi)
    else:
        paths_figures["NOVA_Predictions"] = None
        pass

    if save_clustermap is True:
        clusters_label_colors_fig = draw_color_text_box(
            {"Cluster {}".format(key): value for key, value in lut_cat_dynamic_tree.items()},
            figsize=[2, 4],
            title="Clusters",
            fontfamily=fontfamily
        )

        clusters_label_colors_fig.savefig(paths_figures["clusters_labels"], dpi=fig_dpi)
    else:
        paths_figures["clusters_labels"] = None

    if show_color_luts_for_pred_and_NOVA_truth:
        if "novaclass" in data.columns:
            NOVA_truth_label_colors_fig = draw_color_text_box(
                NOVA_truth_colors_dict, figsize=[2, 2], title="NOVA-Truth Labels",
                # fontfamily=fontfamily
            )

            NOVA_truth_label_colors_fig.savefig(paths_figures["NOVA_truth"], dpi=fig_dpi)
            pass
    else:
        paths_figures["NOVA_truth"] = None
        pass

    # end region

    lut_pred = None
    if data[NOVA_class_column].max() == 3:
        lut_pred = {
            0: rgb_to_hex(np.array([0.4, 0.7607843137254902, 0.6470588235294118]) * 255),
            1: rgb_to_hex(np.array([0.9882352941176471, 0.5529411764705883, 0.3843137254901961]) * 255),
            2: rgb_to_hex(np.array([0.5529411764705883, 0.6274509803921569, 0.796078431372549]) * 255),
            3: rgb_to_hex(np.array([0.9058823529411765, 0.5411764705882353, 0.7647058823529411]) * 255)
        }
    else:
        raise Exception(
            f"Unexpected value of '{data[NOVA_class_column].max()}' for data[NOVA_class_column].max()")

    pred_row_colors = data[NOVA_class_column].map(lut_pred)

    '''IN CASE WE WANT TO HAVE FOOD CATEGORIES AS VERTICAL COLUMNS'''
    # if "Food code" in data.columns:
    #     row_colors = [cat_dynamic_tree, pred_row_colors, cat_level_1_row_colors]
    # else:
    #     row_colors = [cat_dynamic_tree, pred_row_colors]

    row_colors = [cat_dynamic_tree, pred_row_colors]

    if "novaclass" in data.columns:
        lut_NOVA_truth = {
            0: rgb_to_hex(np.array([102, 102, 102])),
            # -1: rgb_to_hex(np.array([102, 102, 102])),
            # 0: 'black',
            12: rgb_to_hex(np.array([204, 204, 204])),
            1: rgb_to_hex(np.array([0.4, 0.7607843137254902, 0.6470588235294118]) * 255),
            2: rgb_to_hex(np.array([0.9882352941176471, 0.5529411764705883, 0.3843137254901961]) * 255),
            3: rgb_to_hex(np.array([0.5529411764705883, 0.6274509803921569, 0.796078431372549]) * 255),
            4: rgb_to_hex(np.array([0.9058823529411765, 0.5411764705882353, 0.7647058823529411]) * 255)
        }

        NOVA_truth_column_colors = data["novaclass"].map(lut_NOVA_truth)

        row_colors.append(NOVA_truth_column_colors)
        pass

    if color_level2_FNDDS_category:
        cats_num = np.array([int(c) for c in data['Level2'].unique()])
        dz = cats_num

        norm = plt.Normalize()

        # colors = plt.cm.jet(norm(dz))
        # colors = plt.cm.jet(np.linspace(0, 1, len(dz)))

        lower = dz.min()
        upper = dz.max()

        # colors = plt.cm.jet((dz - lower) / (upper - lower))

        def get_colors(inp, colormap, vmin=None, vmax=None):
            norm = plt.Normalize(vmin, vmax)
            return colormap(norm(inp))

        FNDDS_level2_cats_colors = get_colors(dz, plt.cm.jet)

        FNDDS_level2_cats_colors = [matplotlib.colors.rgb2hex(c) for c in FNDDS_level2_cats_colors]

        FNDDS_level2_cats_colors_dict = dict(zip(dz.astype(str), FNDDS_level2_cats_colors))

        FNDDS_level2_cat_colors_mapped = data["Level2"].map(FNDDS_level2_cats_colors_dict)

        row_colors.append(FNDDS_level2_cat_colors_mapped)

        if FNDDS_cats_code_label_dict is not None:
            FNDDS_level2_cats_name_colors_dict = {}
            for code, color in FNDDS_level2_cats_colors_dict.items():
                if code in FNDDS_cats_code_label_dict:
                    FNDDS_level2_cats_name_colors_dict[f'[{code}] {FNDDS_cats_code_label_dict[code]}'] = color
                else:
                    FNDDS_level2_cats_name_colors_dict[f'[{code}] NONE'] = color
        else:
            FNDDS_level2_cats_name_colors_dict = FNDDS_level2_cats_colors_dict
            pass

        '''Save FNDDS Level2 Category color lut'''
        # print(FNDDS_level2_cats_name_colors_dict)
        FNDDS_level2_cats_colors_fig = draw_color_text_box(
            FNDDS_level2_cats_name_colors_dict,
            figsize=[5.1, 12],
            title="FNDDS Level2 Category",
            fontfamily=fontfamily
        )
        FNDDS_level2_cats_colors_fig.savefig(paths_figures["FNDDS_level2_categories"], dpi=fig_dpi)
        pass
    else:
        paths_figures["FNDDS_level2_categories"] = None

    if save_clustermap is True:
        sns.set(
            font_scale=font_scale,
            font=fontfamily,
            # Only worked on title of the clustermap (suptitle)
            rc={
                "font.size": 10,
                # "font.weight": 'bold'
            })

        cm = sns.clustermap(
            data[features],
            metric="euclidean",
            row_linkage=link,
            cmap=cmap,
            col_cluster=False,
            figsize=figsize,
            row_colors=row_colors,
            cbar_kws={
                # use_gridspec=False,
                # orientation="horizontal",
                # ticks=[0.0, 0.2, 0.4, 0.5, 0.7, 0.8, 1.0]
            },
            xticklabels=xticklabels,
            annot_kws={
                # Its not working !!
                # "size": 30
                "weight": 20
            },
            # row_colors=[pred_row_colors, cat_dynamic_tree, cat_level_1_row_colors, cat_level_2_row_colors, cat_level_3_row_colors]
        )
        # cm.cax = plt.gca()
        # cm.ax_cbar.remove()
        # cm.cax.set_visible(False)
        # cm.cax.remove()
        # cm.cax.reset_position()
        # cm.ax_row_dendrogram.set_visible(False)
        # cm.ax_col_dendrogram.set_visible(False)
        # mtransforms.Bbox.from_bounds(width=0,height=0,x0=0,y0=0)
        # import matplotlib.transforms as mtransforms
        #
        # cm.ax_cbar.set_position(mtransforms.Bbox.from_bounds(width=0, height=0, x0=0, y0=0))

        if remove_heat_map_legend:
            pass

        # Remove all y ticks and labels
        ax = cm.ax_heatmap
        ax.set(yticks=[])

        cm.fig.suptitle(f"{name} | #Items: {len(data)} #Clusters: {num_clusters}",  # fontweight=0,
                        # fontsize=24 / 2
                        )

        """
        This provides the order of rows in the dendrogram matrix
        # cm.data2d
        """
        cm.savefig(paths_figures["cluster_map"], dpi=fig_dpi)
    else:
        paths_figures["cluster_map"] = None
        cm = None
        pass

    if save_excel:

        path_excel_export = file_name + ".xls"
        # xlsxwriter openpyxl xlwt

        with pd.ExcelWriter(path_excel_export, engine='xlwt') as writer:

            # Write each dataframe to a different worksheet.
            data.to_excel(writer, sheet_name='Sheet1')

            if isinstance(lut_cat_dynamic_tree, dict):
                color_lut_df = pd.DataFrame([json.dumps(lut_cat_dynamic_tree)], columns=["lut_cat_dynamic_tree"])

                color_lut_df.to_excel(writer, sheet_name='cluster colors A-Z Sorted', index=False)
                pass

            writer.save()
            pass

        if open_excel:
            os.system('start EXCEL.EXE "{}"'.format(os.path.abspath(path_excel_export)))

        print("Saved clusters: " + path_excel_export)

        pass

    plt.close('all')

    return {"link": link, "distances": distances, "clusters DT": clusters,
            "data and clusters": data,
            "sns_cm": cm,
            "features": features,
            "cluster IDs": set(clusters["labels"]),
            "paths_figures": {fig_name: path for fig_name, path in paths_figures.items() if path is not None}
            }
