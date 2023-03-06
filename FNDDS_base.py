# coding=utf-8
import os
import pyodbc
import pandas as pd
import joblib
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# import matplotlib._color_data as mcd
import matplotlib.patches as mpatch

import plotly.express as px
import plotly
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.io as pio

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.manifold import TSNE
import umap
import json

import scipy.spatial as sp
from collections import OrderedDict

# import networkx as nx

import re

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


NOVA_predictions_colors_dict = {
    'NOVA 1': rgb_to_hex(np.array([0.4, 0.7607843137254902, 0.6470588235294118]) * 255),
    'NOVA 2': rgb_to_hex(np.array([0.9882352941176471, 0.5529411764705883, 0.3843137254901961]) * 255),
    'NOVA 3': rgb_to_hex(np.array([0.5529411764705883, 0.6274509803921569, 0.796078431372549]) * 255),
    'NOVA 4': rgb_to_hex(np.array([0.9058823529411765, 0.5411764705882353, 0.7647058823529411]) * 255)
}

# NOVA_class_predictions_colors_dict = {
#     0: NOVA_predictions_colors_dict["NOVA 1"],
#     1: NOVA_predictions_colors_dict["NOVA 2"],
#     2: NOVA_predictions_colors_dict["NOVA 3"],
#     3: NOVA_predictions_colors_dict["NOVA 4"]
# }

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


def load_tables_from_access_database(database_path, table_names):
    if len([x for x in pyodbc.drivers() if x.startswith('Microsoft Access Driver')]) == 0:
        raise Exception(
            'You need to install "Access Data Engine" depending your office X32 or X64 it might become challenging to install it.')

    path, file_name = os.path.split(database_path)

    connecion_string = 'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={0};'.format(database_path)

    cnxn = pyodbc.connect(connecion_string)

    df_tables = {}

    for table_name in table_names:
        path_pickle = "{}/{}.pkl".format(path, table_name)

        if os.path.exists(path_pickle):
            print("Loaded from pickle: {}".format(table_name))
            df_tables[table_name] = pd.read_pickle(path_pickle)
            pass
        else:
            query = "SELECT * FROM {}".format(table_name)

            df_tables[table_name] = pd.read_sql(query, cnxn)

            df_tables[table_name].to_pickle(path_pickle)
            pass

        print("Table: {} | Number of rows: {} | Source: {}".format(
            table_name, len(df_tables[table_name]), database_path))
        pass

    cnxn.close()

    return df_tables


base_path = "D:/Dropbox (CCNR)/Foodome Team Folder/Ravandi, Babak FDB/FoodomeDev/"
# It is better to maintain all classifiers in a single location
Classifiers_path = 'D:/Dropbox (CCNR)/Foodome Team Folder/Menichetti, Giulia/FoodProcessing/Datasets/Classifiers/'

'''I just dont want to reinstall cutreeHybrid!!'''
# from dynamicTreeCut import cutreeHybrid
from scipy.spatial.distance import pdist
# from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


def create_ingredient_description_for_FNDDS_2009_10(line_break_char="<br />"):
    ingreds = pd.read_csv("data/NHANES_2009-10_Giulia_1.csv").rename(columns={"dr12ifdcd": "Food_code"})
    # ingreds = ingreds[:200]

    food_with_ingreds = ingreds.groupby("Food_code").agg({"SR_description_t": np.size}).reset_index().rename(
        columns={"SR_description_t": "Count_ingreds"})

    def make_ingred_desciption(row):
        food_code = row["Food_code"]

        ingred_desc = "Count: {}".format(row["Count_ingreds"])

        for index, row in ingreds[ingreds["Food_code"] == food_code][["SR_description_t", "SR_nova_group"]].iterrows():
            ingred_desc += "{} ☼ (NOVA{}) {}".format(line_break_char, row["SR_nova_group"], row["SR_description_t"])

        return ingred_desc
        pass

    food_with_ingreds["ingreds"] = food_with_ingreds.apply(make_ingred_desciption, axis=1)

    return food_with_ingreds


# region CLUSTERS REGION | PLOT CLUSTER NUTRIENTS

def training_corrections(correction_groups=None,
                         path_corrections="data/FNDDS/Correction Training Data FNDDS 2009_10.xls"):
    #
    training_corrections_df = OrderedDict()

    # ORDER MATTERS THEY OVERRIDE EACH OTHER. For example, "Raw Foods" overrides "Giulia Cleaning"
    training_corrections_df["Giulia Cleaning"] = pd.read_excel(path_corrections, sheet_name="Recom Giulia Cleaning")

    training_corrections_df["Giulia Cleaning"] = training_corrections_df["Giulia Cleaning"][
        training_corrections_df["Giulia Cleaning"]["NOVA Recommendation"].isin([1, 2, 3, 4])
    ]

    training_corrections_df["Bean Soups"] = pd.read_excel(path_corrections, sheet_name="Recom Beans")
    training_corrections_df["Home Recipe"] = pd.read_excel(path_corrections, sheet_name="Recome Home Recipe ")
    training_corrections_df["Raw Foods"] = pd.read_excel(path_corrections, sheet_name="Recom Raws")

    if correction_groups is not None:
        for correction_group in list(training_corrections_df.keys()):
            if correction_group not in correction_groups:
                del training_corrections_df[correction_group]
                pass
            pass
        pass

    correction_groups = list(training_corrections_df.keys())
    correction_groups.reverse()

    for group_high_priority in correction_groups:
        start_checking = False

        for groups_to_be_overridden in correction_groups:
            if start_checking:
                # print("key_up: {} key_fix: {}".format(group_high_priority, groups_to_be_overridden))
                training_corrections_df[groups_to_be_overridden] = training_corrections_df[groups_to_be_overridden][
                    ~training_corrections_df[groups_to_be_overridden]["Food_code"].isin(
                        training_corrections_df[group_high_priority]["Food_code"]
                    )
                ]
                pass
            if groups_to_be_overridden == group_high_priority:
                start_checking = True
                pass
            pass

        pass

    return training_corrections_df


plotly_trace_symbols = [
    'square', 'diamond', 'cross', 'x', 'star', 'hexagram', 'bowtie', 'hourglass', 'triangle-left',
    'triangle-right', 'hexagon', 'triangle-up', 'diamond-tall', 'octagon', 'pentagon', 'star-open', 'x-open',

    'circle-open', 'circle-dot', 'circle-open-dot', 'square-open', 'square-dot',
    'square-open-dot', 'diamond-open', 'diamond-dot', 'diamond-open-dot', 'cross-open', 'cross-dot',
    'cross-open-dot', 'x-dot', 'x-open-dot', 'triangle-up-open', 'triangle-up-dot',
    'triangle-up-open-dot', 'triangle-down', 'triangle-down-open', 'triangle-down-dot',
    'triangle-down-open-dot', 'triangle-left-open', 'triangle-left-dot', 'triangle-left-open-dot',
    'triangle-right-open', 'triangle-right-dot', 'triangle-right-open-dot', 'triangle-ne',
    'triangle-ne-open', 'triangle-ne-dot', 'triangle-ne-open-dot', 'triangle-se', 'triangle-se-open',
    'triangle-se-dot', 'triangle-se-open-dot', 'triangle-sw', 'triangle-sw-open', 'triangle-sw-dot',
    'triangle-sw-open-dot', 'triangle-nw', 'triangle-nw-open', 'triangle-nw-dot', 'triangle-nw-open-dot',
    'pentagon-open', 'pentagon-dot', 'pentagon-open-dot', 'hexagon-open', 'hexagon-dot', 'hexagon-open-dot',
    'hexagon2-open', 'hexagon2-dot', 'hexagon2-open-dot', 'octagon-open', 'octagon-dot', 'octagon-open-dot',

    'star-dot', 'star-open-dot', 'hexagram-open', 'hexagram-dot', 'hexagram-open-dot',
    'star-triangle-up', 'star-triangle-up-open', 'star-triangle-up-dot', 'star-triangle-up-open-dot',
    'star-triangle-down', 'star-triangle-down-open', 'star-triangle-down-dot', 'star-triangle-down-open-dot',
    'star-square', 'star-square-open', 'star-square-dot', 'star-square-open-dot', 'star-diamond',
    'star-diamond-open', 'star-diamond-dot', 'star-diamond-open-dot', 'diamond-tall-open',
    'diamond-tall-dot', 'diamond-tall-open-dot', 'diamond-wide', 'diamond-wide-open', 'diamond-wide-dot',
    'diamond-wide-open-dot', 'hourglass-open', 'bowtie-open', 'circle-cross', 'circle-cross-open',
    'circle-x', 'circle-x-open', 'square-cross', 'square-cross-open', 'square-x', 'square-x-open',
    'diamond-cross', 'diamond-cross-open', 'diamond-x', 'diamond-x-open', 'cross-thin', 'cross-thin-open',
    'x-thin', 'x-thin-open', 'asterisk-open', 'hash-open', 'hash-dot', 'hash-open-dot', 'y-up', 'y-up-open',
    'y-down', 'y-down-open', 'y-left', 'y-left-open', 'y-right', 'y-right-open', 'line-ew', 'line-ew-open',
    'line-ns', 'line-ns-open', 'line-ne', 'line-ne-open', 'line-nw', 'line-nw-open', 'asterisk', 'hash',
]


def plot_predictions_PCA(predictions_df, column_legend, column_food_code, column_hover_description,
                         color_lut_legend, width=700, height=500, trace_color_column=None,
                         trace_food=None, trace_marker_opacity=0.7, marker_opacity=0.8,
                         title="", trace_marker_default_size=10, axis_range=None,
                         hover_label_args=None):
    if trace_food is None:
        trace_food = {}

    def cluster_label(cluster_number):
        return "{} ({})".format(predictions_df["cluster_labels"][cluster_number], cluster_number)
        pass

    # if limit_clusters_df_size is not None:
    #     predictions_df = predictions_df[0:limit_clusters_df_size]

    predictions_df["color"] = predictions_df[column_legend].map(color_lut_legend)

    # predictions_df["Cluster description"] = predictions_df.apply(cluster_description, axis=1)

    lb = {"p1": r'$p_1$', "p4": r'$p_4$'}

    Xpca = predictions_df.loc[:, 'p1': 'p4']

    # Apply PCA
    pca = PCA()
    # pca.fit(Xpca_std)
    pc = pca.fit_transform(Xpca)

    predictions_df['PC 1'] = pc[:, 0]
    predictions_df['PC 2'] = pc[:, 1]

    fig = go.Figure()

    traced_items_all_df = pd.DataFrame([], columns=[column_food_code])
    if len(trace_food) > 0:
        traced_items_all_df = []
        for trace_key, trace_dict in trace_food.items():
            traced_items_all_df.append(trace_dict["df"])
            pass
        traced_items_all_df = pd.concat(traced_items_all_df)
        pass

    for legend_value, color in color_lut_legend.items():
        predictions_selected_df = predictions_df[
            (predictions_df[column_legend] == legend_value) &
            (~predictions_df[column_food_code].isin(traced_items_all_df[column_food_code]))
            ]

        # cluster_df["marker size"] = 5
        # cluster_df["opacity"] = 0.5

        #     cluster_df.loc[cluster_df.index.isin(onions.index), "marker size"] = 20
        #     cluster_df.loc[cluster_df.index.isin(onions.index), "opacity"] = 0.85

        fig.add_trace(go.Scattergl(
            x=predictions_selected_df['PC 1'], y=predictions_selected_df['PC 2'],
            text=predictions_selected_df[column_hover_description],
            mode='markers', marker=dict(color=color,
                                        # size=cluster_df["marker size"],
                                        # opacity=cluster_df["opacity"]
                                        opacity=marker_opacity
                                        ),
            name="{} [{}]".format(legend_value, len(predictions_selected_df)),
            # legendgroup="all",
            visible='legendonly'
        ))
        pass

    unused_trace_symbol = 0
    # if True and trace_food_df is not None:
    for trace_key, trace_dict in trace_food.items():
        # trace_food_df = None, trace_food_mark_food_codes = None
        traced_items_df = predictions_df[predictions_df[column_food_code].isin(trace_dict["df"][column_food_code])]

        if "symbol" in trace_dict:
            traced_items_df["Marker symbol"] = trace_dict["symbol"]
        else:
            traced_items_df["Marker symbol"] = plotly_trace_symbols[unused_trace_symbol]
            unused_trace_symbol += 1
            pass

        if "size" in trace_dict:
            traced_items_df["Marker size"] = trace_dict["size"]
        else:
            traced_items_df["Marker size"] = trace_marker_default_size
            pass

        # # cluster_df[cluster_df.index.isin(onions.index)]["marker size"] = 10
        # # cluster_df.loc[cluster_df.index.isin(onions.index)] = 10
        # # cluster_df.loc[cluster_df.index.isin(onions.index), "marker size"] = 10
        # if trace_food_mark_food_codes is not None:
        #     for food_code, symbol in trace_food_mark_food_codes.items():
        #         food_code_index = traced_items_df[traced_items_df["Food_code"] == food_code].index[0]
        #
        #         traced_items_df.loc[food_code_index, "Marker symbol"] = symbol
        #         traced_items_df.loc[food_code_index, "Marker size"] = trace_food_marker["given_food_codes_marker_size"]
        #         pass
        #     pass
        if "custom_mark_food_codes" in trace_dict:
            for food_code, symbol in trace_dict["custom_mark_food_codes"].items():
                food_code_index = traced_items_df[traced_items_df["Food_code"] == food_code].index[0]

                traced_items_df.loc[food_code_index, "Marker symbol"] = symbol
                traced_items_df.loc[food_code_index, "Marker size"] = trace_food_marker["given_food_codes_marker_size"]
                pass
            pass

        if trace_color_column is None:
            trace_color = traced_items_df[column_legend].map(color_lut_legend)
            pass
        else:
            trace_color = traced_items_df[trace_color_column]

        fig.add_trace(go.Scattergl(
            x=traced_items_df['PC 1'], y=traced_items_df['PC 2'],
            # text=traced_items_df["Cluster description"],
            text=traced_items_df[column_hover_description],
            mode='markers',
            marker=dict(
                color=trace_color,
                size=traced_items_df["Marker size"],
                opacity=trace_marker_opacity,
                symbol=traced_items_df["Marker symbol"],
                line=dict(color='black', width=1.2)
            ),
            name="{} [{}]".format(trace_dict["name"], len(traced_items_df)),
            legendgroup=trace_key,
            visible='legendonly'
        ))

        if False:
            fig.add_trace(go.Scattergl(
                x=traced_items_df['PC 1'], y=traced_items_df['PC 2'], opacity=0.4,
                mode='lines',
                name="Trajectory", line=dict(color="Black"),
                visible='legendonly'
                # legendgroup=trace_food_name
            ))
            pass

        pass

    if axis_range is None:
        axis_range = {}
        axis_range["y"] = dict(range=[predictions_df["PC 1"].min() - 0.1, predictions_df["PC 1"].max() + 0.1]),
        axis_range["x"] = dict(range=[predictions_df["PC 2"].min() - 0.1, predictions_df["PC 2"].max() + 0.1])
        print("Axis is automatic!")
        pass

    fig.update_layout(
        hoverlabel=hover_label_args,
        xaxis_autorange=False,
        yaxis_autorange=False,
        yaxis=axis_range["x"],
        xaxis=axis_range["y"],
        autosize=False,
        title=title,
        width=width,
        height=height,
        margin=dict(l=0, r=0, b=0, t=30, pad=4),
        paper_bgcolor="#fff",
        plot_bgcolor="#fff",
        xaxis_title="PC 1",
        yaxis_title="PC 2",
    )

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    #     fig.show()
    return fig


# endregion

def load_NOVA_classified_food_for_FNDDS_2009_2010_with_prediction():
    return pd.read_csv("data/RFFNDDSpredS_cleaned.csv")


def plot_number_ingredients_recipes(df_all, name, path_save):
    df = df_all.groupby(["Is ingredient", "NOVA class"]).count()

    vals_dict = df.to_dict("index")

    ingred_recipe_vals = np.zeros((2, 4), dtype=int)

    for is_ingredient in range(0, 2):
        for NOVA_class in range(1, 5):
            if (is_ingredient, NOVA_class) in vals_dict:
                ingred_recipe_vals[is_ingredient, NOVA_class - 1] = vals_dict[(is_ingredient, NOVA_class)]["Code"]
                pass

    NOVA_groups = ['NOVA 1', 'NOVA 2', 'NOVA 3', 'NOVA 4']

    fig = go.Figure(data=[
        go.Bar(name='Ingredients', x=NOVA_groups, y=ingred_recipe_vals[1]),
        go.Bar(name='Recipes', x=NOVA_groups, y=ingred_recipe_vals[0])
    ])

    fig.update_layout({"title": name})
    fig.write_image("{}/food_ingredients_NOVA_class_count_{}.png".format(path_save, name))

    # fig.show()
    return fig


def load_USDA_Nutrient_Retention_Factors():
    print("Loading USDA_Nutrient_Retention_Factors Release 6 2007")
    ret = pd.read_csv("data/USDA_Nutrient_Retention_Factors_Release_6_2007/NutrientRetention.csv")
    ret["Retn_Factor"] = (
            ret["Retn_Factor"].str.replace("Sep-75", "100")
            .astype(float)
            / 100
    )
    # ret["Retn_Factor"] = ret["Retn_Factor"] / 100

    return ret


def plot_NOVA_classes_dist(df, name, path_save):
    df["class"] += 1

    group_data = [df['p1'], df['p2'], df['p3'], df['p4']]
    group_labels = ['p1', 'p2', 'p3', 'p4']

    fig = ff.create_distplot(group_data, group_labels, bin_size=.05)
    fig.update_layout({"title": "NOVA classes distribution for: " + name})

    plotly.offline.plot(fig, filename="{}/NOVA_classes_distribution{}.html".format(path_save, name),
                        auto_open=True)

    return fig


def plot_NOVA_class_histogram(df, NOVA_df, name, path_save):
    presentation_columns = ["Code", "Is ingredient", "Label_full", "WWEIA Category description"]

    df = (pd.merge(df[presentation_columns],
                   NOVA_df[["Code", "class", "p1", "p2", "p3", "p4"]],
                   left_on="Code", right_on="Code")
          .rename(columns={"class": "NOVA class"})
          )

    df["NOVA class"] += 1

    group_data = [df['p1'], df['p4']]
    group_labels = ['p1', 'p4']
    # colors = ['rgb(255, 255, 229)', 'rgb(102, 37, 6)'] # creamy-white
    colors = ['rgb(68, 114, 196)', 'rgb(102, 37, 6)']  # blue
    # colors = ['rgb(112, 173, 71)', 'rgb(102, 37, 6)'] # green

    # Create distplot with custom bin_size
    fig = ff.create_distplot(group_data, group_labels, bin_size=0.075, colors=colors)
    fig.update_layout({"title": name})

    fig.write_image("{}/food_ingredients_p1_p4_dist_{}.png".format(path_save, name))

    # df.groupby("")

    return df, fig
    pass


def presentation_ingred(ingred_df, NOVA_df):
    presentation_columns = ["Seq num", "Ingredient code", "Ingredient description", "Retention code",
                            "Ingredient weight", "Food ingredient weight normalized"]
    ingred_df = (pd.merge(ingred_df[presentation_columns],
                          NOVA_df[["Code", "class", "p1", "p2", "p3", "p4"]],
                          left_on="Ingredient code", right_on="Code")
                 .drop(columns="Code")
                 .rename(columns={"class": "NOVA class"})
                 )

    ingred_df["NOVA class"] += 1

    return ingred_df


def hover_data_for_charts(df):
    cols = []

    for column in df.columns:
        if column in ["Main food description", "pred class", "novaclass", "Food code", "category_groupped",
                      "WWEIA Category description", "Count Ingredients", "Ingredient description", "p2", "p3"]:
            cols.append(column)
            pass
        pass

    return cols


from pandas import ExcelWriter


def save_xls(dfs_dict, xls_path, open=False):
    with ExcelWriter(xls_path) as writer:
        for df_name, df in dfs_dict.items():
            df.to_excel(writer, df_name)
        writer.save()

    if open is True:
        os.system('start EXCEL.EXE "{}"'.format(os.path.abspath(xls_path)))


def save_excel(df, relative_path, open, index):
    df.to_excel(relative_path, index=index)

    if open is True:
        os.system('start EXCEL.EXE "{}"'.format(os.path.abspath(relative_path)))
    pass


# region GIULIA CLASSIFIER COLUMN NAMES and ORDERS

FNDDS_table_salt = {"Ingredient code": 2047, "Description": "Salt, table"}

NOVA_color_discrete_map = {'Unprocessed': 'green', 'processed culinary': 'blue', 'processed': 'yellow',
                           'ultra processed': 'red'}

giulia_classifier_12_panel_cols = clf_cols_order = [
    'Protein', 'Total Fat', 'Carbohydrate', 'Sugars, total', 'Fiber, total dietary', 'Calcium', 'Iron',
    'Sodium', 'Vitamin C', 'Cholesterol', 'Fatty acids, total saturated', 'Total Vitamin A']

giulia_classifier_99_panel_cols = [  # Used to be giulia_classifier_order_cols
    "Protein", "Total Fat", "Carbohydrate", "Alcohol", "Water", "Caffeine", "Theobromine",
    "Sugars, total", "Fiber, total dietary", "Calcium", "Iron", "Magnesium", "Phosphorus", "Potassium",
    "Sodium", "Zinc", "Copper", "Selenium",
    "Retinol", "Carotene, beta", "Carotene, alpha", "Vitamin E (alpha-tocopherol)",
    "Vitamin D (D2 + D3)",  # not in 58
    "Cryptoxanthin, beta", "Lycopene", "Lutein + zeaxanthin", "Vitamin C", "Thiamin", "Riboflavin",
    "Niacin", "Vitamin B-6", "Folate, total", "Vitamin B-12",
    "Choline, total",  # not in 58
    "Vitamin K (phylloquinone)", "Folic acid", "Folate, food",
    "Vitamin E, added",  # not in 58
    "Vitamin B-12, added", # not in 58
    "Cholesterol", "Fatty acids, total saturated", "4:0", "6:0", "8:0", "10:0", "12:0",
    "14:0", "16:0", "18:0", "18:1", "18:2", "18:3", "20:4", "22:6 n-3", "16:1", "18:4", "20:1", "20:5 n-3",
    "22:1", "22:5 n-3", "Fatty acids, total monounsaturated", "Fatty acids, total polyunsaturated",
    # Flavonoids nutrients only available in special database
    "Daidzein", "Genistein", "Glycitein", "Cyanidin", "Petunidin", "Delphinidin", "Malvidin",
    "Pelargonidin", "Peonidin", "(+)-Catechin", "(-)-Epigallocatechin", "(-)-Epicatechin", "(-)-Epicatechin 3-gallate",
    "(-)-Epigallocatechin 3-gallate", "Theaflavin", "Thearubigins", "Eriodictyol",
    "Hesperetin", "Naringenin", "Apigenin", "Luteolin", "Isorhamnetin", "Kaempferol", "Myricetin", "Quercetin",
    "Theaflavin-3,3\'-digallate", "Theaflavin-3\'-gallate", "Theaflavin-3-gallate",
    "(+)-Gallocatechin", "Total flavonoids", "Total anthocyanidins", "Total catechins (monomeric flavan-3-ols only)",
    "Total flavan-3-ols", "Total flavanones", "Total flavones", "Total flavonols", "Total isoflavones"]

giulia_classifier_62_panel_cols = giulia_classifier_99_panel_cols[:62]
# giulia_classifier_order_cols_for_FNDDS_2015_2016 = giulia_classifier_order_cols + ['Folate, DFE', 'Vitamin A, RAE']

giulia_classifier_cols_nutrient_facts = [
    'Protein', 'Total Fat', 'Carbohydrate', 'Sugars, total', 'Fiber, total dietary', 'Calcium', 'Iron',
    'Sodium', 'Vitamin C', 'Cholesterol', 'Fatty acids, total saturated', 'Total Vitamin A']

giulia_classifier_cols_removed_on_FNDDS_2015_2016 = [
    'Daidzein', 'Genistein', 'Glycitein', 'Cyanidin', 'Petunidin', 'Delphinidin', 'Malvidin', 'Pelargonidin',
    'Peonidin', '(+)-Catechin', '(-)-Epigallocatechin', '(-)-Epicatechin', '(-)-Epicatechin 3-gallate',
    '(-)-Epigallocatechin 3-gallate', 'Theaflavin', 'Thearubigins', 'Eriodictyol', 'Hesperetin', 'Naringenin',
    'Apigenin', 'Luteolin', 'Isorhamnetin', 'Kaempferol', 'Myricetin', 'Quercetin', "Theaflavin-3,3'-digallate",
    "Theaflavin-3'-gallate", 'Theaflavin-3-gallate', '(+)-Gallocatechin', 'Total flavonoids', 'Total anthocyanidins',
    'Total catechins (monomeric flavan-3-ols only)', 'Total flavan-3-ols', 'Total flavanones', 'Total flavones',
    'Total flavonols', 'Total isoflavones']

giulia_classifier_order_cols_2015_2016 = [
    'Protein', 'Total Fat', 'Carbohydrate', 'Alcohol', 'Water', 'Caffeine',
    'Theobromine', 'Sugars, total', 'Fiber, total dietary', 'Calcium',
    'Iron', 'Magnesium', 'Phosphorus', 'Potassium', 'Sodium', 'Zinc',
    'Copper', 'Selenium', 'Retinol', 'Carotene, beta', 'Carotene, alpha',
    'Vitamin E (alpha-tocopherol)', 'Vitamin D (D2 + D3)',
    'Cryptoxanthin, beta', 'Lycopene', 'Lutein + zeaxanthin', 'Vitamin C',
    'Thiamin', 'Riboflavin', 'Niacin', 'Vitamin B-6', 'Folate, total',
    'Vitamin B-12', 'Choline, total', 'Vitamin K (phylloquinone)',
    'Folic acid', 'Folate, food', 'Vitamin E, added', 'Vitamin B-12, added',
    'Cholesterol', 'Fatty acids, total saturated', '4:0', '6:0', '8:0', '10:0', '12:0', '14:0', '16:0', '18:0', '18:1',
    '18:2', '18:3',
    '20:4', '22:6 n-3', '16:1', '18:4', '20:1', '20:5 n-3', '22:1', '22:5 n-3', 'Fatty acids, total monounsaturated',
    'Fatty acids, total polyunsaturated'
]

giulia_calc_vitamin_A_by_sum_of_columns = [
    (319, 'Retinol'), (321, 'Carotene, beta'), (322, 'Carotene, alpha'), (334, 'Cryptoxanthin, beta')
]

# giulia_classifier_99_nuts_codes = {
#     203: 'Protein', 204: 'Total Fat', 205: 'Carbohydrate', 221: 'Alcohol', 255: 'Water', 262: 'Caffeine',
#     263: 'Theobromine', 269: 'Sugars, total', 291: 'Fiber, total dietary', 301: 'Calcium', 303: 'Iron',
#     304: 'Magnesium', 305: 'Phosphorus', 306: 'Potassium', 307: 'Sodium', 309: 'Zinc', 312: 'Copper',
#     317: 'Selenium', 319: 'Retinol', 321: 'Carotene, beta', 322: 'Carotene, alpha', 323: 'Vitamin E (alpha-tocopherol)',
#     328: 'Vitamin D (D2 + D3)', 334: 'Cryptoxanthin, beta', 337: 'Lycopene', 338: 'Lutein + zeaxanthin',
#     401: 'Vitamin C', 404: 'Thiamin', 405: 'Riboflavin', 406: 'Niacin', 415: 'Vitamin B-6', 417: 'Folate, total',
#     418: 'Vitamin B-12', 421: 'Choline, total', 430: 'Vitamin K (phylloquinone)', 431: 'Folic acid',
#     432: 'Folate, food', 573: 'Vitamin E, added', 578: 'Vitamin B-12, added', 601: 'Cholesterol',
#     606: 'Fatty acids, total saturated', 607: '4:0', 608: '6:0', 609: '8:0', 610: '10:0', 611: '12:0',
#     612: '14:0', 613: '16:0', 614: '18:0', 617: '18:1', 618: '18:2', 619: '18:3', 620: '20:4',
#     621: '22:6 n-3', 626: '16:1', 627: '18:4', 628: '20:1', 629: '20:5 n-3', 630: '22:1', 631: '22:5 n-3',
#     645: 'Fatty acids, total monounsaturated', 646: 'Fatty acids, total polyunsaturated', 7000: 'Total flavonoids',
#     731: 'Cyanidin', 740: 'Petunidin', 741: 'Delphinidin', 742: 'Malvidin', 743: 'Pelargonidin', 745: 'Peonidin',
#     7100: 'Total anthocyanidins', 749: '(+)-Catechin', 750: '(-)-Epigallocatechin', 751: '(-)-Epicatechin',
#     752: '(-)-Epicatechin 3-gallate', 753: '(-)-Epigallocatechin 3-gallate', 755: 'Theaflavin', 756: 'Thearubigins',
#     791: "Theaflavin-3,3'-digallate", 792: "Theaflavin-3'-gallate", 793: 'Theaflavin-3-gallate',
#     794: '(+)-Gallocatechin', 7200: 'Total catechins (monomeric flavan-3-ols only)', 7300: 'Total flavan-3-ols',
#     758: 'Eriodictyol', 759: 'Hesperetin', 762: 'Naringenin', 7400: 'Total flavanones', 770: 'Apigenin',
#     773: 'Luteolin', 7500: 'Total flavones', 785: 'Isorhamnetin', 786: 'Kaempferol', 788: 'Myricetin',
#     789: 'Quercetin', 7600: 'Total flavonols', 710: 'Daidzein', 711: 'Genistein', 712: 'Glycitein',
#     7700: 'Total isoflavones'}
# giulia_classifier_99_nuts_codes_df = pd.DataFrame(
#     zip(giulia_classifier_99_nuts_codes.keys(), giulia_classifier_99_nuts_codes.values()),
#     columns=["Nutrient code", "Nutrient description"]
# )

giulia_classifier_99_nuts_codes_df = pd.DataFrame(
    [
        [203, 'Protein', 'g'], [204, 'Total Fat', 'g'], [205, 'Carbohydrate', 'g'], [221, 'Alcohol', 'g'],
        [255, 'Water', 'g'], [262, 'Caffeine', 'mg'], [263, 'Theobromine', 'mg'], [269, 'Sugars, total', 'g'],
        [291, 'Fiber, total dietary', 'g'], [301, 'Calcium', 'mg'], [303, 'Iron', 'mg'], [304, 'Magnesium', 'mg'],
        [305, 'Phosphorus', 'mg'], [306, 'Potassium', 'mg'], [307, 'Sodium', 'mg'], [309, 'Zinc', 'mg'],
        [312, 'Copper', 'mg'], [317, 'Selenium', 'mcg'], [319, 'Retinol', 'mcg'], [321, 'Carotene, beta', 'mcg'],
        [322, 'Carotene, alpha', 'mcg'], [323, 'Vitamin E (alpha-tocopherol)', 'mg'],
        [328, 'Vitamin D (D2 + D3)', 'mcg'],
        [334, 'Cryptoxanthin, beta', 'mcg'], [337, 'Lycopene', 'mcg'], [338, 'Lutein + zeaxanthin', 'mcg'],
        [401, 'Vitamin C', 'mg'], [404, 'Thiamin', 'mg'], [405, 'Riboflavin', 'mg'], [406, 'Niacin', 'mg'],
        [415, 'Vitamin B-6', 'mg'], [417, 'Folate, total', 'mcg'], [418, 'Vitamin B-12', 'mcg'],
        [421, 'Choline, total', 'mg'], [430, 'Vitamin K (phylloquinone)', 'mcg'], [431, 'Folic acid', 'mcg'],
        [432, 'Folate, food', 'mcg'], [573, 'Vitamin E, added', 'mg'], [578, 'Vitamin B-12, added', 'mcg'],
        [601, 'Cholesterol', 'mg'], [606, 'Fatty acids, total saturated', 'g'], [607, '4:0', 'g'], [608, '6:0', 'g'],
        [609, '8:0', 'g'], [610, '10:0', 'g'], [611, '12:0', 'g'], [612, '14:0', 'g'], [613, '16:0', 'g'],
        [614, '18:0', 'g'], [617, '18:1', 'g'], [618, '18:2', 'g'], [619, '18:3', 'g'], [620, '20:4', 'g'],
        [621, '22:6 n-3', 'g'], [626, '16:1', 'g'], [627, '18:4', 'g'], [628, '20:1', 'g'], [629, '20:5 n-3', 'g'],
        [630, '22:1', 'g'], [631, '22:5 n-3', 'g'], [645, 'Fatty acids, total monounsaturated', 'g'],
        [646, 'Fatty acids, total polyunsaturated', 'g'],
        # Dine 62 nuts
        [7000, 'Total flavonoids', 'mg'], [731, 'Cyanidin', 'mg'],
        [740, 'Petunidin', 'mg'], [741, 'Delphinidin', 'mg'], [742, 'Malvidin', 'mg'], [743, 'Pelargonidin', 'mg'],
        [745, 'Peonidin', 'mg'], [7100, 'Total anthocyanidins', 'mg'], [749, '(+)-Catechin', 'mg'],
        [750, '(-)-Epigallocatechin', 'mg'], [751, '(-)-Epicatechin', 'mg'], [752, '(-)-Epicatechin 3-gallate', 'mg'],
        [753, '(-)-Epigallocatechin 3-gallate', 'mg'], [755, 'Theaflavin', 'mg'], [756, 'Thearubigins', 'mg'],
        [791, "Theaflavin-3,3'-digallate", 'mg'], [792, "Theaflavin-3'-gallate", 'mg'],
        [793, 'Theaflavin-3-gallate', 'mg'], [794, '(+)-Gallocatechin', 'mg'],
        [7200, 'Total catechins (monomeric flavan-3-ols only)', 'mg'], [7300, 'Total flavan-3-ols', 'mg'],
        [758, 'Eriodictyol', 'mg'], [759, 'Hesperetin', 'mg'], [762, 'Naringenin', 'mg'],
        [7400, 'Total flavanones', 'mg'], [770, 'Apigenin', 'mg'], [773, 'Luteolin', 'mg'],
        [7500, 'Total flavones', 'mg'], [785, 'Isorhamnetin', 'mg'], [786, 'Kaempferol', 'mg'],
        [788, 'Myricetin', 'mg'], [789, 'Quercetin', 'mg'], [7600, 'Total flavonols', 'mg'], [710, 'Daidzein', 'mg'],
        [711, 'Genistein', 'mg'], [712, 'Glycitein', 'mg'], [7700, 'Total isoflavones', 'mg']
    ],
    columns=["Nutrient code", "Nutrient description", "unit"]
)

flavones_nutrient_desc = pd.DataFrame(
    [
        (7000, 'Total flavonoids', 'mg'), (731, 'Cyanidin', 'mg'), (740, 'Petunidin', 'mg'), (741, 'Delphinidin', 'mg'),
        (742, 'Malvidin', 'mg'), (743, 'Pelargonidin', 'mg'), (745, 'Peonidin', 'mg'),
        (7100, 'Total anthocyanidins', 'mg'), (749, '(+)-Catechin', 'mg'), (750, '(-)-Epigallocatechin', 'mg'),
        (751, '(-)-Epicatechin', 'mg'), (752, '(-)-Epicatechin 3-gallate', 'mg'),
        (753, '(-)-Epigallocatechin 3-gallate', 'mg'), (755, 'Theaflavin', 'mg'), (756, 'Thearubigins', 'mg'),
        (791, "Theaflavin-3,3'-digallate", 'mg'), (792, "Theaflavin-3'-gallate", 'mg'),
        (793, 'Theaflavin-3-gallate', 'mg'), (794, '(+)-Gallocatechin', 'mg'),
        (7200, 'Total catechins (monomeric flavan-3-ols only)', 'mg'), (7300, 'Total flavan-3-ols', 'mg'),
        (758, 'Eriodictyol', 'mg'), (759, 'Hesperetin', 'mg'), (762, 'Naringenin', 'mg'),
        (7400, 'Total flavanones', 'mg'), (770, 'Apigenin', 'mg'), (773, 'Luteolin', 'mg'),
        (7500, 'Total flavones', 'mg'), (785, 'Isorhamnetin', 'mg'), (786, 'Kaempferol', 'mg'),
        (788, 'Myricetin', 'mg'), (789, 'Quercetin', 'mg'), (7600, 'Total flavonols', 'mg'),
        (710, 'Daidzein', 'mg'), (711, 'Genistein', 'mg'), (712, 'Glycitein', 'mg'), (7700, 'Total isoflavones', 'mg')
    ],
    columns=["Nutrient code", "Nutrient description", "unit"]
)

# endregion

"""
PCA visualization
"""


def NOVA_PCA_analysis(n_components, df, hover_data, method, col_pred_class="pred class",
                      features=["p1", "p2", "p3", "p4"], width=1000, height=650, standard_scaler_fit=True):
    x = df.loc[:, features].values

    if standard_scaler_fit is True:
        x = preprocessing.StandardScaler().fit_transform(x)
        pass

    method = method.lower()

    if method == "umap":
        reducer = umap.UMAP(n_components=n_components)
        principalComponents = reducer.fit_transform(x)

        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['PC{}'.format(i) for i in range(1, n_components + 1)])
        pass
    elif method == "pca":
        pca = PCA(n_components=n_components)

        principalComponents = pca.fit_transform(x)

        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['PC{}'.format(i) for i in range(1, n_components + 1)])
    elif method == "tsne":
        tsne = TSNE(n_components=n_components, verbose=0, perplexity=60, n_iter=800)
        # tsne = TSNE(n_components=n_components)

        principalComponents = tsne.fit_transform(x)

        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['PC{}'.format(i) for i in range(1, n_components + 1)])
    else:
        raise Exception("Method {} is not handeled.".format(method))

    # hover_data = list(set(hover_data) - set(features))

    finalDf = pd.concat([principalDf, df[list(set(hover_data) - set(features)) + features]], axis=1)

    finalDf['cat'] = df[col_pred_class].map(
        # {1: 'Unprocessed', 2: 'processed culinary', 3: 'processed', 4: 'ultra processed'}
        {1: 'NOVA 1', 2: 'NOVA 2', 3: 'NOVA 3', 4: 'NOVA 4'}
    )
    # NOVA_color_discrete_map = {'Unprocessed': 'green', 'processed culinary': 'blue', 'processed': 'yellow',
    #                            'ultra processed': 'red'}
    color_discrete_map = {'NOVA 1': 'green', 'NOVA 2': 'blue', 'NOVA 3': 'yellow', 'NOVA 4': 'red'}

    if n_components == 1:
        # hover_data=hover_data + features
        # return finalDf
        fig = px.scatter(finalDf, x="PC1", y="PC1", hover_data=hover_data, width=width, height=height,
                         color="cat", color_discrete_map=color_discrete_map)

    elif n_components == 2:
        # hover_data=hover_data + features
        fig = px.scatter(finalDf, x="PC1", y="PC2", hover_data=hover_data, width=width, height=height,
                         color="cat", color_discrete_map=color_discrete_map)

    elif n_components == 3:
        # hover_data[:-2] + features
        fig = px.scatter_3d(finalDf, x="PC1", y="PC2", z="PC3", hover_data=hover_data,
                            color="cat", color_discrete_map=color_discrete_map)
        pass
    else:
        raise Exception("n_components can be only 1, 2 or 3")

    fig.update_traces(mode='markers', marker_line_width=0.0, marker_size=4, marker_opacity=0.5)

    #     fig.update_layout(title="PCA | Num Food-Items: {}".format(len(df)))

    #     # fig.update_xaxes(range=[0.0, 1.0])
    #     # fig.update_yaxes(range=[0.0, 1.0])
    # fig.show()

    return fig


def convert_to_mg(value, unit):
    if unit == "g":
        r = value * 1000

    elif unit == "mg":
        r = value

    elif unit == "mcg":
        r = value * 0.001
        pass
    else:
        return value
        pass

    return r


def convert_to_g_supported_units():
    return ["iu", "fl oz", "ml", "liter", "g", "gram", "lb", "fluid ounce",
            "lbs", "oz"]
    pass


def scale_to_100_g():
    raise Exception("NOT COMPLETE")
    pass


def convert_to_g_accepted_units():
    return ["g", "gram", "gm", "grams"] + ["mg"] + ['µg', "mcg", "mcg_rae", "mcg_dfe"] + \
           ["lb", "lbs"] + ["oz"] + ["fl oz", "fl", "fluid ounce"] + ["ml"] + ["liter"]  # + #["iu"] + ["re"]


def convert_to_g(value, unit, nutrient):
    # https: // dietarysupplementdatabase.usda.nih.gov / Conversions.php
    if value is None:
        raise Exception("X")

    nutrient = nutrient.lower()
    r = None

    unit = unit.lower()

    if unit in ["g", "gram", "grams", "gm"]:
        r = value

    elif unit == "mg":
        r = value * 0.001

    elif unit in ['µg', "mcg", "mcg_rae", "mcg_dfe", 'µg_dfe']:
        r = value * 0.000001

    elif unit in ["lb", "lbs"]:
        r = value * 453.59237

    elif unit in ["oz"]:
        r = value * 28.3495

    elif unit == "iu":
        if nutrient in ['vitamin e', 'vitamine']:
            # 1 IU = 0.67 mg for d-alpha-tocopherol (natural)
            # 1 IU = 0.9 mg for dl-alpha-tocopherol (synthetic)
            r = value * 0.67 * 0.001
            pass

    elif unit in ["re", 'µg_rae']:
        # https://www.dietobio.com/vegetarisme/en/vit_a.html
        """WARNING!!! DOUBLE CHECK 'RE' UNIT CONVERSION TO GRAMS!!!!"""
        if nutrient in ['carotene']:
            r = value * 0.000001 * 6
            pass
        if nutrient in ['vitamin a', 'vitamina', 'vitamin a, rae']:
            r = value * 0.000001
            pass

    elif any(x in unit for x in ["calories", "kcal", "cals", 'kj']):
        r = value

    elif any(x in unit for x in ["fl oz", "fl", "fluid ounce"]):
        r = value * 29.57352956

    elif unit == "ml":
        r = value * 1

    elif unit == "liter":
        r = value * 1000

    else:
        raise Exception("Can not convert unit '{}' for {}".format(unit, nutrient))

    return r


def convert_nutrient_unit_to_grams(df, nutrient_values_column):
    """
    Convert all units to MG
    """
    # if "Nutrient value grams" not in df.columns:
    #

    df.insert(
        list(df.columns).index(nutrient_values_column),
        nutrient_values_column + " grams",
        df.apply(
            lambda row: convert_to_g(row[nutrient_values_column], row["Unit"], row['Nutrient description']), axis=1),
        True
    )

    pass


def draw_color_text_box(overlap, figsize, title, title_font_size=16, title_font_y=1.1, fontfamily=None):
    """
    Source: https://matplotlib.org/tutorials/colors/colors.html
    :param overlap:
    :return:
    """
    # import matplotlib._color_data as mcd
    # import matplotlib.patches as mpatch

    # overlap = {name for name in mcd.CSS4_COLORS
    #            if "xkcd:" + name in mcd.XKCD_COLORS}
    # overlap = {'p1': 'g', 'p2': 'b', 'p3': 'y', 'p4': 'r'}

    # fig = plt.figure(figsize=[4.8, 2])
    # figsize = [2, 2]
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=title_font_size, y=title_font_y, fontfamily=fontfamily)
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
        txt = ax.text(figsize[0] / 2, j + 0.5, '  ' + key, va='center', fontsize=10,
                      weight=weight, fontfamily=fontfamily)
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


def predict_NOVA_class(clf, clf_cols_order, nutrients_df, pos_insert_preds, convert_to_log, new_cols_name_prefix=""):
    """
    Run classifier_DEL_ME on a given dataframe
    :param clf:
    :param clf_cols_order:
    :param nutrients_df:
    :param pos_insert_preds:
    :param convert_to_log:
    :return:
    """
    if new_cols_name_prefix != "":
        new_cols_name_prefix = new_cols_name_prefix.strip() + " "

    nutrients_df_nuts = nutrients_df[clf_cols_order]

    if convert_to_log is True:
        nutrients_df_nuts = np.log(nutrients_df_nuts)
        nutrients_df_nuts = nutrients_df_nuts.fillna(-20)
        nutrients_df_nuts = nutrients_df_nuts.replace([np.inf, -np.inf], -20)
        pass

    predicted_class = clf.predict(nutrients_df_nuts)

    predict_prob = clf.predict_proba(nutrients_df_nuts)

    nutrients_df.insert(pos_insert_preds, new_cols_name_prefix + "class", predicted_class, True)

    nutrients_df.insert(pos_insert_preds + 1, new_cols_name_prefix + "p4", predict_prob[:, 3], True)
    nutrients_df.insert(pos_insert_preds + 1, new_cols_name_prefix + "p3", predict_prob[:, 2], True)
    nutrients_df.insert(pos_insert_preds + 1, new_cols_name_prefix + "p2", predict_prob[:, 1], True)
    nutrients_df.insert(pos_insert_preds + 1, new_cols_name_prefix + "p1", predict_prob[:, 0], True)

    return nutrients_df


def get_giulia_classifier(name, only_clf_cols_order=False):
    nut_panels = {
        '58P': [
            'Protein', 'Total Fat', 'Carbohydrate', 'Alcohol', 'Water',
            'Caffeine', 'Theobromine', 'Sugars, total', 'Fiber, total dietary',
            'Calcium', 'Iron', 'Magnesium', 'Phosphorus', 'Potassium',
            'Sodium', 'Zinc', 'Copper', 'Selenium', 'Retinol',
            'Carotene, beta', 'Carotene, alpha',
            'Vitamin E (alpha-tocopherol)', 'Cryptoxanthin, beta', 'Lycopene',
            'Lutein + zeaxanthin', 'Vitamin C', 'Thiamin', 'Riboflavin',
            'Niacin', 'Vitamin B-6', 'Folate, total', 'Vitamin B-12',
            'Vitamin K (phylloquinone)', 'Folic acid', 'Folate, food',
            'Cholesterol', 'Fatty acids, total saturated', '4:0', '6:0', '8:0',
            '10:0', '12:0', '14:0', '16:0', '18:0', '18:1', '18:2', '18:3',
            '20:4', '22:6 n-3', '16:1', '18:4', '20:1', '20:5 n-3', '22:1',
            '22:5 n-3', 'Fatty acids, total monounsaturated',
            'Fatty acids, total polyunsaturated'
        ]
    }

    clf_cols_order = None
    clf = None

    if name == "nutritionfacts_with_salt":
        if only_clf_cols_order is False:
            clf = joblib.load(
                base_path + "classifier_DEL_ME/randomforestmodel_train_all_SMOTEcleaned_nutritionfacts_withsalt.pkl")

        clf_cols_order = giulia_classifier_12_panel_cols

    elif name == "99 nutrition facts with salt and 8 raw items":
        if only_clf_cols_order is False:
            clf = joblib.load(
                base_path + "classifier_DEL_ME/classifier_newlabels_99.pkl")

        clf_cols_order = giulia_classifier_99_panel_cols

    elif name == "62 nutrition facts with salt and 8 raw items":
        if only_clf_cols_order is False:
            clf = joblib.load(
                base_path + "classifier_DEL_ME/classifier_newlabels_62.pkl")
        # raise Exception("Fix this!")
        clf_cols_order = giulia_classifier_62_panel_cols

    ###################################

    elif name == "Ensemble 58P":
        if only_clf_cols_order is False:
            clf = joblib.load(
                Classifiers_path + "/FNDDS0910_58Nutrients_5folds_SMOTE.pkl"
            )

        clf_cols_order = nut_panels['58P']

    elif name == "58 nutrition facts with salt and 8 raw items":
        if only_clf_cols_order is False:
            clf = joblib.load(
                # base_path + "classifier_DEL_ME/classifier_newlabels_58.pkl"
                Classifiers_path + 'CCNR2019_classifier_58_recoms_Giulia_Salt_Raws_TrainTestFNDDS2009.pkl'
            )

        clf_cols_order = nut_panels['58P']

    ###################################

    elif name == "12 nutrition facts with salt and 8 raw items":
        if only_clf_cols_order is False:
            clf = joblib.load(
                base_path + "classifier_DEL_ME/classifier_newlabels_12.pkl"
            )

        clf_cols_order = [
            'Protein', 'Total Fat', 'Carbohydrate', 'Sugars, total', 'Fiber, total dietary', 'Calcium', 'Iron',
            'Sodium', 'Vitamin C', 'Cholesterol', 'Fatty acids, total saturated', 'Total Vitamin A']

    elif name == "11 nutrition facts with salt and 8 raw items":
        if only_clf_cols_order is False:
            clf = joblib.load(
                base_path + "classifier_DEL_ME/classifier_newlabels_11.pkl"
            )
        # Protein	'Total Fat'	Carbohydrate	'Sugars, total'	'Fiber, total dietary'	'Calcium'	'Iron'
        # Sodium	'Vitamin C'	'Cholesterol'	'Fatty acids, total saturated'
        clf_cols_order = [
            'Protein', 'Total Fat', 'Carbohydrate', 'Sugars, total', 'Fiber, total dietary', 'Calcium', 'Iron',
            'Sodium', 'Vitamin C', 'Cholesterol', 'Fatty acids, total saturated']

    elif name == "10 nutrition facts with salt and 8 raw items":
        if only_clf_cols_order is False:
            clf = joblib.load(
                base_path + "classifier_DEL_ME/classifier_newlabels_10.pkl")

        clf_cols_order = [
            'Protein', 'Total Fat', 'Carbohydrate', 'Sugars, total', 'Fiber, total dietary', 'Calcium', 'Iron',
            'Sodium', 'Cholesterol', 'Fatty acids, total saturated']

    elif name == "nutritionfacts":
        if only_clf_cols_order is False:
            clf = joblib.load(
                base_path + "classifier_DEL_ME/randomforestmodel_train_all_SMOTEcleaned_nutritionfacts.pkl")

    elif name == "basic_panel":
        if only_clf_cols_order is False:
            clf = joblib.load(
                base_path + "classifier_DEL_ME/randomforestmodel_train_all_SMOTEcleaned_basicnutrientpanel.pkl")
    else:
        raise Exception("The given name '{}' is not known.".format(name))
    # clf = joblib.load(
    #     "D:/Dropbox (Foodome)/Foodome Team Folder/FOODOME/GiuliaPres/NOVAclassifier/randomforestmodel_train_all_SMOTEcleaned.pkl")

    return clf, clf_cols_order


# function to predict over a generic db
def ensemble_classify(db, model_per_fold, nut_sel, log_transform, pre):
    dbsel = db.loc[:, nut_sel]

    if log_transform:
        dbsel = np.log(dbsel)
        dbsel = dbsel.fillna(-20)
        dbsel = dbsel.replace([np.inf, -np.inf], -20)
        pass

    Xnut = dbsel.values

    indfold = 0
    for model in model_per_fold:
        indfold += 1
        y_pred = model.predict(Xnut)
        y_probs = model.predict_proba(Xnut)
        db[pre + 'classf' + str(indfold)] = y_pred
        db[pre + 'p1f' + str(indfold)] = y_probs[:, 0]
        db[pre + 'p2f' + str(indfold)] = y_probs[:, 1]
        db[pre + 'p3f' + str(indfold)] = y_probs[:, 2]
        db[pre + 'p4f' + str(indfold)] = y_probs[:, 3]
        db[pre + 'FPSf' + str(indfold)] = (
                (1 - db[pre + 'p1f' + str(indfold)] + db[pre + 'p4f' + str(indfold)]) / 2
        )

    for p in range(1, 5):
        db[pre + 'p' + str(p)] = (
            db.loc[:,
            [pre + 'p' + str(p) + 'f1',
             pre + 'p' + str(p) + 'f2',
             pre + 'p' + str(p) + 'f3',
             pre + 'p' + str(p) + 'f4',
             pre + 'p' + str(p) + 'f5']
            ]
                .mean(axis=1)
        )
        db[pre + 'std_p' + str(p)] = (
            db.loc[:,
            [pre + 'p' + str(p) + 'f1',
             pre + 'p' + str(p) + 'f2',
             pre + 'p' + str(p) + 'f3',
             pre + 'p' + str(p) + 'f4',
             pre + 'p' + str(p) + 'f5']
            ]
                .std(axis=1)
        )

    db[pre + 'FPS'] = (
        db.loc[:, [pre + 'FPSf1', pre + 'FPSf2', pre + 'FPSf3', pre + 'FPSf4', pre + 'FPSf5']]
            .mean(axis=1)
    )
    db[pre + 'std_FPS'] = (
        db.loc[:, [pre + 'FPSf1', pre + 'FPSf2', pre + 'FPSf3', pre + 'FPSf4', pre + 'FPSf5']]
            .std(axis=1)
    )
    db[pre + 'min_FPS'] = (
        db.loc[:, [pre + 'FPSf1', pre + 'FPSf2', pre + 'FPSf3', pre + 'FPSf4', pre + 'FPSf5']]
            .min(axis=1)
    )

    db[pre + 'max_p'] = db.loc[:, [pre + 'p1', pre + 'p2', pre + 'p3', pre + 'p4']].idxmax(axis=1)
    db[pre + 'class'] = (
        [int(s[-1]) - 1 for s in db.loc[:, [pre + 'p1', pre + 'p2', pre + 'p3', pre + 'p4']].idxmax(axis=1)]
    )

    db[pre + 'min_fold_id'] = (
        [int(s[-1]) for s in db.loc[:, [pre + 'FPSf1', pre + 'FPSf2', pre + 'FPSf3', pre + 'FPSf4', pre + 'FPSf5']]
            .idxmin(axis=1)]
    )
    db[pre + 'min_class'] = (
        [db[pre + 'classf' + str(db[pre + 'min_fold_id'].iloc[n])].iloc[n] for n in range(db.shape[0])]
    )

    return db


class FNDDS:
    def __init__(self, years, desc):
        self.years = years
        self.desc = desc

        self.foodDesc = None
        self.nutVal = None
        self.nutDesc = None

        # TODO bad idea do not use self.foodNutValDesc and self.nutValF = None ANYWHERE!
        # self.foodNutValDesc = None
        # self.nutValF = None  # store flavonid database nutrient value

        # dd stands for dietry data
        #         self.dd_DR1IFF = None
        #         self.dd_DR2IFF = None
        """
        Dictionary of all dataframes
        """
        self.df = {}

        self.tables_loaded = []
        #         self.df_FNDDS = {}

        #         self.num_participants = None

        #         self.df_food_categories = {
        #             "raw": None
        #         }

        pass

    def load_data(self, dataset_path, FNDDS_tables):
        self.df = load_tables_from_access_database(database_path=dataset_path, table_names=FNDDS_tables)

        pass

    """
    Add nutrients from Flavonoid database
    """

    def add_flavonid_nutrients(self, fndds_flav):
        raise Exception("OLD redesign this!")

        self.nutDescF = pd.concat([
            self.nutDesc[["Nutrient code", "Nutrient description", "Tagname", "Unit", "Decimals"]],
            fndds_flav.nutDesc[["Nutrient code", "Flavonoid description", "Tagname", "Unit", "Decimals"]].rename(
                columns={"Flavonoid description": "Nutrient description"})
        ])

        self.nutDescF = self.nutDescF.reset_index()
        pass

    def add_flavonid_NutVal(self, fndds_flav):

        self.nutValF = pd.concat([
            self.nutVal[["Food code", "Nutrient code", "Start date", "End date", "Nutrient value"]],
            fndds_flav.nutVal[["Food code", "Nutrient code", "Start date", "End date", "Nutrient value"]]
        ])

        self.nutValF = self.nutValF.reset_index()
        pass

    # TODO OLD REMOVE THIS
    # def prep_for_giulia_classifier(self):
    #     self.nutDescF = self.nutDescF[~self.nutDescF["Nutrient code"].isin([
    #         # These nutrients were missing in the classifier_DEL_ME matrix
    #         208,  # Energy
    #         320,  # Vitamin A, RAE
    #         435  # Folate, DFE
    #     ])]
    #
    #     self.nutDescF = self.nutDescF.reset_index()
    #     pass

    def get_nutrient_values(self, foods=None, ingredients=None, convert_to_unit="g", log_values=True,
                            logs_values_round_to_decimal=30, log_values_replace_inf=-20):
        if foods is not None and ingredients is not None:
            raise Exception("Either works on getting nutrients for ingredients or foods, not both.")

        if foods is not None:
            all_nuts = pd.merge(self.df["FNDDSNutVal"], foods, on=["Food code"])

            all_nuts = pd.merge(all_nuts, self.df["NutDesc"], on="Nutrient code")

            col_name_code = "Food code"
            col_name_desc = "Main food description"
            is_ingredient = 0

        elif ingredients is not None:
            all_nuts = pd.merge(self.df["INGREDNutVal"], ingredients, on=["Ingredient code"])

            all_nuts = pd.merge(all_nuts, self.df["NutDesc"], on="Nutrient code")

            col_name_code = "Ingredient code"
            col_name_desc = "Ingredient description"
            is_ingredient = 1
            pass

        all_nuts["Nutrient value in grams"] = \
            all_nuts.apply(lambda row: convert_to_g(row["Nutrient value"], row["Unit"]), axis=1)

        all_nuts_pivot = pd.pivot_table(all_nuts, values='Nutrient value in grams',
                                        index=[col_name_code, col_name_desc],
                                        columns=['Nutrient description'], aggfunc=np.mean)

        if log_values is True:
            all_nuts_pivot = np.round(np.log(all_nuts_pivot), logs_values_round_to_decimal)
            all_nuts_pivot = all_nuts_pivot.replace([np.inf, -np.inf], log_values_replace_inf)
            pass

        columns_order = []
        for col in giulia_classifier_order_cols_2015_2016:
            if col in all_nuts_pivot.columns:
                columns_order.append(col)

        all_nuts_pivot = all_nuts_pivot[columns_order]

        all_nuts_pivot = all_nuts_pivot.reset_index()

        all_nuts_pivot = all_nuts_pivot.rename(columns={col_name_code: "Code", col_name_desc: "Description"})

        columns = list(all_nuts_pivot.columns)

        all_nuts_pivot["Is ingredient"] = is_ingredient

        all_nuts_pivot = all_nuts_pivot[columns[:2] + ["Is ingredient"] + columns[2:]]

        return all_nuts, all_nuts_pivot

    def import_NOVA_and_giulia_classes(self, df, insert_at_column_index):
        NOVA = load_NOVA_classified_food_for_FNDDS_2009_2010_with_prediction()

        import_cols = ["Food code", "novaclass", "class"]
        columns = list(df.columns)

        df2 = pd.merge(df, NOVA[import_cols], left_on="Code", right_on="Food code", how="left")

        df2 = df2[columns[:insert_at_column_index] + import_cols + columns[insert_at_column_index:]]

        df2 = df2.rename(columns={"novaclass": "NOVA class imported", "class": "Giulia class imported"})

        df2 = df2.drop(columns=["Food code"])

        return df2

    def query_foods_ingredients_with_63_nutrients(self, log_values=True, log_values_replace_inf=-20, save_excel=False):
        food = f16.df["MainFoodDesc"]

        ingreds = f16.query_get_all_ingredients_code_description()

        all_nuts_food, all_nuts_pivot_food = f16.get_nutrient_values(
            foods=food, convert_to_unit="g", log_values_replace_inf=log_values_replace_inf, log_values=log_values)

        all_nuts_pivot_food = f16.import_NOVA_and_giulia_classes(all_nuts_pivot_food, insert_at_column_index=3)
        #
        all_nuts_ingred, all_nuts_pivot_ingred = f16.get_nutrient_values(
            ingredients=ingreds,
            convert_to_unit="g", log_values_replace_inf=log_values_replace_inf, log_values=log_values)

        all_nuts_pivot_ingred = f16.import_NOVA_and_giulia_classes(all_nuts_pivot_ingred, insert_at_column_index=3)

        food_ingred_nutrients = pd.concat([all_nuts_pivot_food, all_nuts_pivot_ingred]).reset_index(drop=True)
        food_ingred_nutrients = food_ingred_nutrients.fillna(-1)

        if save_excel:
            save_excel(food_ingred_nutrients, "data/FNDDS_2015_2016_foods_ingredients_with_62_nutrients.xlsx")

        return food_ingred_nutrients

    def nutrient_values_fix_unit(self, convert_to_unit="g"):
        """
        Convert all units to MG
        """
        if "Nutrient value grams" not in self.df["FNDDSNutVal"].columns:

            FNDDSNutVal = pd.merge(self.df["FNDDSNutVal"], self.df['NutDesc'], on="Nutrient code")
            #
            self.df["FNDDSNutVal"].insert(
                3,
                "Nutrient value grams",
                FNDDSNutVal.apply(
                    lambda row: convert_to_g(row["Nutrient value"], row["Unit"]), axis=1),
                True
            )

            pass
        else:
            print("FNDDSNutVal is already converted to grams")

        if "Nutrient value grams" not in self.df["INGREDNutVal"].columns:
            INGREDNutVal = pd.merge(self.df["INGREDNutVal"][:100], self.df['NutDesc'], on="Nutrient code")
            self.df["INGREDNutVal"].insert(
                4,
                "Nutrient value grams",
                INGREDNutVal.apply(
                    lambda row: convert_to_g(row["Nutrient value"], row["Unit"]), axis=1),
                True
            )

            pass
        else:
            print("INGREDNutVal is already converted to grams")

        return
        # INGREDNutVal

        if convert_to_unit == "g":
            self.foodNutValDesc["Nutrient value"] = self.foodNutValDesc.apply(
                lambda row: convert_to_g(row["Nutrient value"], row["Unit"]), axis=1)
        elif convert_to_unit == "mg":
            self.foodNutValDesc["Nutrient value"] = self.foodNutValDesc.apply(
                lambda row: convert_to_mg(row["Nutrient value"], row["Unit"]), axis=1)
        else:
            raise Exception("Given unit is unknown")
        pass

    def join_with_NOVA_classes(self, df):
        RFFNDDSpred_cleaned_Guilia = pd.read_csv("RFFNDDSpredS_cleaned.csv")

        if self.years == "2015_2016":
            food_items_ranked = pd.read_csv("food_items_ranked_by_weight_count_2015_2016.csv".format(self.years))

            q = pd.merge(df, food_items_ranked, on=["Food code"], how="left")

            q = pd.merge(q, self.foodDesc[["Food code", "WWEIA Category description", "WWEIA Category code"]],
                         on=["Food code"], how="left")

            # len(food_items_ranked)
            # len(m)
            # len(m)
            # RFFNDDSpred_cleaned_Guilia[["Food code", "novaclass"]]

            q = pd.merge(q, RFFNDDSpred_cleaned_Guilia[["Food code", "novaclass", "class"]], on=["Food code"],
                         how="left")

            q = q[list(df.columns[0:2]) +
                  ["WWEIA Category description", "WWEIA Category code"] +
                  list(food_items_ranked.columns[1:]) +
                  ["novaclass", "class"] +
                  list(df.columns[2:])]

            q["novaclass"] = q["novaclass"].fillna(-1)

            # "WWEIA Category description",
        else:
            q = pd.merge(df, RFFNDDSpred_cleaned_Guilia[["Food code", "novaclass", "class"]], on=["Food code"],
                         how="left")

            q["novaclass"] = q["novaclass"].fillna(-1)
            pass

        q = q.rename(columns={"class": "class_giulia"})

        q["class_giulia"] = q["class_giulia"] + 1

        return q, RFFNDDSpred_cleaned_Guilia

    def count_number_of_ingredients(self, df):
        if self.years == "2015_2016":
            if "Count Ingredients" in df.columns:
                q = df.drop(columns=["Count Ingredients"])

            f16_ingred_food_count = self.df["FNDDSIngred"].groupby(["Food code"]).count()
            f16_ingred_food_count = f16_ingred_food_count.reset_index()[["Food code", "Ingredient code"]].rename(
                columns={"Ingredient code": "Count Ingredients"})

            q = pd.merge(df, f16_ingred_food_count, on="Food code")

            """
            Add Ingredients
            """

            if "Ingredient description" in q.columns:
                q = q.drop(columns=["Ingredient description"])

            f16_food_ingrid_names = self.df["FNDDSIngred"][["Food code", "Ingredient description"]]
            # .groupby(["Food code"]).count()

            q = pd.merge(q,
                         f16_food_ingrid_names.groupby(["Food code"])["Ingredient description"].apply(
                             '<br />⚫ '.join).reset_index(),
                         on=["Food code"])

            return q

        else:
            print("count_number_of_ingredients() only works for FNDDS 2015-2016.")
            return df
            pass
        pass

    def ingro_prob(self, food_code, food_name, compare_target_nutrient_code, ingreds, clf, SR24_df, flav_approx_to_SR,
                   check_target=False):

        if check_target:
            FNDDS_nuts = self.get_food_nutrients(food_code=food_code)  # 203	2009-01-01	2010-12-31	9.460

            target_nut_val = FNDDS_nuts[FNDDS_nuts["Nutrient code"] == compare_target_nutrient_code]["Nutrient value"]
            target_nut_val = list(target_nut_val)[0]

            # print("FNDDS {}".format(FNDDS_nuts[FNDDS_nuts["Nutrient code"] == compare_target_nutrient_code]))
            print("Compare Target {}, value: {}".format(
                giulia_classifier_99_nuts_codes_df[
                    giulia_classifier_99_nuts_codes_df["Nutrient code"] == compare_target_nutrient_code][
                    "Nutrient description"].values,
                target_nut_val
            ))
            pass

        if ingreds is None:
            ingreds = self.get_ingredients(food_code=food_code, break_food_code_as_ingredient=True)

        ingreds_weight_total = ingreds["Food ingredient weight"].sum()
        print("ingreds_weight_total: {}".format(ingreds_weight_total))

        #     ingreds["Weight normalized"] = ingreds["Weight"] / ingreds_weight_total

        if SR24_df["NUT_DATA"]['NDB_No'].dtype != np.int64:
            SR24_df["NUT_DATA"]['NDB_No'] = SR24_df["NUT_DATA"]['NDB_No'].astype(np.int64)
        if SR24_df["NUT_DATA"]['Nutr_No'].dtype != np.int64:
            SR24_df["NUT_DATA"]['Nutr_No'] = SR24_df["NUT_DATA"]['Nutr_No'].astype(np.int64)

        ingred_nuts_df = pd.merge(
            SR24_df["NUT_DATA"][["NDB_No", "Nutr_No", "Nutr_Val"]],
            ingreds,
            left_on='NDB_No', right_on='SR code')

        if flav_approx_to_SR is not None:
            flav_ingred_nuts = pd.merge(
                flav_approx_to_SR[["SR code", "Parent food code", "Nutrient code", "Nutrient value"]],
                ingreds,
                on=["Parent food code", "SR code"]).rename(columns={
                "Nutrient code": "Nutr_No", "Nutrient value": "Nutr_Val"
            })
            flav_ingred_nuts["NDB_No"] = None
            flav_ingred_nuts = flav_ingred_nuts[ingred_nuts_df.columns]

            ingred_nuts_df = pd.concat([ingred_nuts_df, flav_ingred_nuts])
            pass

        # return ingred_nuts_df

        ingred_nuts_df["Nutr_weighted_Val"] = ingred_nuts_df["Nutr_Val"] * ingred_nuts_df[
            "Food ingredient weight normalized"]

        # Scale the amount of nutrients
        calculated_nutrient_val = ingred_nuts_df[ingred_nuts_df["Nutr_No"] == compare_target_nutrient_code][
            "Nutr_weighted_Val"].sum()

        if check_target is True:
            print("Amount {} - dif %: {}".format(
                calculated_nutrient_val,
                round(1 - (calculated_nutrient_val / target_nut_val), 2)
            ))
            pass

        # SR_ingred_nuts[SR_ingred_nuts["Nutr_No"] == nutrient_code]
        #     ingred_nuts = SR_ingred_nuts[SR_ingred_nuts["SR code"] == ingridient_SR_code]
        ingred_nuts = ingred_nuts_df

        nuts_to_feed_classifier = pd.merge(
            giulia_classifier_99_nuts_codes_df,
            ingred_nuts,
            how='left', left_on='Nutrient code', right_on='Nutr_No')[
            ["Nutrient code", "Nutrient description", "Nutr_Val", "SR code", "SR description",
             "Weight",
             "Food ingredient weight",
             "Food ingredient weight normalized",
             "Nutr_weighted_Val", "unit"]
        ]

        nuts_to_feed_classifier["Nutr_weighted_Val_g_unit"] = nuts_to_feed_classifier.apply(
            lambda row: convert_to_g(row["Nutr_weighted_Val"], row["unit"], row["Nutrient description"]),
            axis=1
        )

        # return nuts_to_feed_classifier

        # nuts_to_feed_classifier["Nutr_weighted_Val_log"] = np.log(
        #     nuts_to_feed_classifier["Nutr_weighted_Val_g_unit"]).replace([np.inf, -np.inf], -20)

        # return nuts_to_feed_classifier

        # nuts_to_feed_classifier_2 = nuts_to_feed_classifier[[
        #     "Nutrient description", "Nutr_weighted_Val_log", "SR code", "SR description",
        #     "Food ingredient weight normalized"]]

        # return nuts_to_feed_classifier_2

        ingredient_nutrients_calculated = pd.pivot_table(nuts_to_feed_classifier, values='Nutr_weighted_Val_g_unit',
                                                         index=["SR code", "SR description",
                                                                "Food ingredient weight",
                                                                "Food ingredient weight normalized"],
                                                         columns=['Nutrient description'],
                                                         aggfunc=np.mean).reset_index()

        ingredient_nutrients_calculated = ingredient_nutrients_calculated.sort_values(
            by=["Food ingredient weight normalized"], ascending=False)

        ingredient_nutrients_calculated = ingredient_nutrients_calculated[
            list(ingredient_nutrients_calculated.columns[:4]) + giulia_classifier_99_panel_cols
            ]

        # return ingredient_nutrients_calculated

        data_to_classifier = ingredient_nutrients_calculated[giulia_classifier_99_panel_cols].sum()
        data_to_classifier = pd.DataFrame(data_to_classifier).T
        data_to_classifier = np.log(data_to_classifier).replace([np.inf, -np.inf], -20)

        predicted_class = clf.predict(data_to_classifier)

        predict_prob = clf.predict_proba(data_to_classifier)

        """
        """

        data_to_classifier.insert(0, "Food code", food_code, True)
        data_to_classifier.insert(1, "Main food description", food_name, True)

        data_to_classifier.insert(2, "class", predicted_class, True)

        data_to_classifier.insert(3, "p4", predict_prob[:, 3], True)
        data_to_classifier.insert(3, "p3", predict_prob[:, 2], True)
        data_to_classifier.insert(3, "p2", predict_prob[:, 1], True)
        data_to_classifier.insert(3, "p1", predict_prob[:, 0], True)

        return {"prediction": data_to_classifier,
                "ingredient_nutrients_calculated": ingredient_nutrients_calculated}

    def get_food_nutrients(self, food_code):
        return self.df["FNDDSNutVal"][self.df["FNDDSNutVal"]["Food code"] == food_code]
        pass

    def get_food_moisture_change(self, food_code):
        if food_code is None:
            return self.df["MoistAdjust"]
        else:
            return self.df["MoistAdjust"][self.df["MoistAdjust"]["Food code"] == food_code]

    def find_food(self, keywords=None, food_code=None, category=None):
        food_df = self.df["MainFoodDesc"]

        if keywords is None and food_code is None and category is None:
            return food_df

        if keywords is None and food_code is None:
            return food_df[food_df["WWEIA Category description"].str.contains(category, case=False)]

        if keywords is not None:
            if type(keywords) != list:
                keywords = [keywords]

            food_df = self.df["MainFoodDesc"]

            for keyword in keywords:
                food_df = food_df[
                    food_df["Main food description"].str.contains(keyword, case=False)
                ]
                # Fish_NFS= Fish_NFS[Fish_NFS["Main food description"].str.contains("steak", case=False)]
                pass

            if category is not None:
                if self.years == "2015_2016":
                    food_df = food_df[food_df["WWEIA Category description"].str.contains(category, case=False)]
                else:
                    raise Exception("Cannot search by category if year is not 2015-2016.")

            return food_df
        elif food_code is not None:
            return self.df["MainFoodDesc"][self.df["MainFoodDesc"]["Food code"] == food_code]
            pass
        else:
            raise Exception("no argument is given to the function.")
        pass

    def describe_food(self, food_code):
        f = self.find_food(food_code=food_code)

        return "[FC: {}] {} [CAT: {}]".format(
            f["Food code"].values[0],
            f["Main food description"].values[0],
            f["WWEIA Category description"].values[0]
        )

    def query_network_food_food(self, FI_nodes_df, FI_links_df, weights_attr_to_sum, name, path_save=None):
        """
        Creates a network of food-food where the weighs are the number of ingredients shared between two foods.
        The nodes (foods) color is the likelihood of being ultra-processed given by FI_nodes_df column 'p4'.
        Parameters, FI_nodes_df and FI_links_df must be passed from function query_network_food_ingredients
        """
        g = nx.from_pandas_edgelist(FI_links_df, source='Source', target='Target',
                                    edge_attr=weights_attr_to_sum
                                    # edge_attr=["Food ingredient weight", "Food ingredient weight normalized"]
                                    )

        attrs = FI_nodes_df.set_index("Code").to_dict("index")
        nx.set_node_attributes(g, attrs)

        A = nx.adjacency_matrix(g, weight=None, nodelist=g.nodes()).todense()

        path_length_2_count = A * A.T

        triu_indices = np.triu_indices(path_length_2_count.shape[0])

        nodes_list_preserved_indices = list(g.nodes)

        rows = []

        for i in range(0, len(triu_indices[0])):
            m = triu_indices[0][i]
            n = triu_indices[1][i]

            if m == n:
                continue

            if path_length_2_count[m, n] > 0 and g.nodes[nodes_list_preserved_indices[m]]["Is ingredient"] == 0 and \
                    g.nodes[nodes_list_preserved_indices[n]]["Is ingredient"] == 0:
                #         print((m , n))
                source_node = nodes_list_preserved_indices[m]
                target_node = nodes_list_preserved_indices[n]
                edge_weight = path_length_2_count[m, n]

                # print("{} ({}) -- {} ({}) [# ing share: {}]".format(
                #     nodes_list_preserved_indices[m],
                #     g.nodes[nodes_list_preserved_indices[m]]["Is ingredient"],
                #     nodes_list_preserved_indices[n],
                #     g.nodes[nodes_list_preserved_indices[n]]["Is ingredient"],
                #     path_length_2_count[m, n]
                # ))

                if True:
                    sum = 0

                    for path in nx.all_simple_paths(g, source_node, target_node, cutoff=2):
                        s_node = source_node

                        for node in path[1:]:
                            t_node = node

                            sum += g.get_edge_data(s_node, t_node)[weights_attr_to_sum]
                            # print(g.edges(data=True))

                            s_node = node
                            pass
                        pass
                    pass

                rows.append((source_node, target_node, edge_weight, sum))
                pass
            pass

        food_food_links = pd.DataFrame(data=rows,
                                       columns=['Source', 'Target', 'Shared ingredients count', weights_attr_to_sum])

        """
        scale the weights between (0.1, 1)
        """
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 1))
        # float_array = food_food_links[["Weight"]].values.astype(float)
        # scaled_array = min_max_scaler.fit_transform(float_array)
        # food_food_links["Weight normalized"] = scaled_array

        if "Size" not in FI_nodes_df.columns:
            FI_nodes_df["Size"] = FI_nodes_df["Code"].apply(lambda x: g.degree(x))

        if path_save is not None:
            food_food_links.to_csv("{}/food_food_{}_links.csv".format(path_save, name), index=False)
            FI_nodes_df.to_csv("{}/food_food_{}_nodes.csv".format(path_save, name), index=False)

        return FI_nodes_df, food_food_links, g

    def query_network_food_ingredients(self, food_df, classification_df, name, normalize_weights, labels_length=30,
                                       path_save=None, WWEIA_pie_chart=True, WWEIA_pie_chart_auto_open_html=False):
        # 51301120     11100000
        food_ingreds = []

        food_ingreds_weights = []

        for index, row in food_df.iterrows():
            food_code = row["Food code"]
            food_desc = row["Main food description"]

            # food_code, Ingredient code, Food ingredient weight

            ingredients = self.get_ingredients(food_code=food_code, break_food_code_as_ingredient=True)

            # Filter water
            ingredients = ingredients[ingredients["Ingredient code"] != 14411]

            food_ingreds_weights.append(ingredients[["Food code", "Ingredient code", "Food ingredient weight"]])

            food_ingreds.append(ingredients)

            pass

        """
        Average weighted p values
        """
        food_ingreds_weights_df = pd.concat(food_ingreds_weights)
        del food_ingreds_weights

        food_ingreds_weights_df = pd.merge(food_ingreds_weights_df, classification_df[["Code", "p1", "p2", "p3", "p4"]],
                                           left_on="Ingredient code", right_on="Code")

        def s(data):
            data["Sum weight"] = data["Food ingredient weight"].sum()
            # data["Count ingred"] = len(data)
            return data

        food_ingreds_weights_df = food_ingreds_weights_df.groupby("Food code").apply(s)

        food_ingreds_weights_df["<p1>"] = (food_ingreds_weights_df["Food ingredient weight"] * food_ingreds_weights_df[
            "p1"]) / food_ingreds_weights_df["Sum weight"]
        food_ingreds_weights_df["<p2>"] = (food_ingreds_weights_df["Food ingredient weight"] * food_ingreds_weights_df[
            "p2"]) / food_ingreds_weights_df["Sum weight"]
        food_ingreds_weights_df["<p3>"] = (food_ingreds_weights_df["Food ingredient weight"] * food_ingreds_weights_df[
            "p3"]) / food_ingreds_weights_df["Sum weight"]
        food_ingreds_weights_df["<p4>"] = (food_ingreds_weights_df["Food ingredient weight"] * food_ingreds_weights_df[
            "p4"]) / food_ingreds_weights_df["Sum weight"]

        food_ingreds_weights_df = food_ingreds_weights_df.groupby("Food code").sum().reset_index()
        food_ingreds_weights_df = food_ingreds_weights_df[["Food code", "<p1>", "<p2>", "<p3>", "<p4>"]]

        """
        """

        links = pd.concat(food_ingreds).reset_index(drop=True)

        links = links[["Food code", "Ingredient code", "Food ingredient weight", "Food ingredient weight normalized"]]

        food_nodes_prop = pd.merge(
            links[["Food code"]].drop_duplicates(),
            classification_df,
            left_on="Food code",
            right_on="Code"
        )[["Code", "Description", "p4", "Is ingredient"]]

        ingred_nodes_prop = pd.merge(
            links[["Ingredient code"]].drop_duplicates(),
            classification_df,
            left_on="Ingredient code",
            right_on="Code"
        )[["Code", "Description", "p4", "Is ingredient"]]

        nodes = pd.concat([food_nodes_prop, ingred_nodes_prop]).reset_index(drop=True)

        links = links.rename(columns={
            "Food code": "Source",
            "Ingredient code": "Target",
            # "Food ingredient weight normalized": "Weight",
        })

        nodes = nodes.rename(columns={
            "Description": "Label"
        })

        nodes["Label_full"] = nodes["Label"]

        labels_length = int(labels_length / 2)

        nodes["Label"] = nodes["Label"].apply(
            lambda x: "{}..{}".format(
                x[0:labels_length].strip(), x[-labels_length:].strip()
            ))
        #     str.slice(start=-10)

        nodes = pd.merge(
            nodes,
            food_df[["Food code", "WWEIA Category description", "WWEIA Category code"]],
            left_on="Code", right_on="Food code", how="outer") \
            .drop(columns=["Food code"])

        nodes = pd.merge(
            nodes,
            food_ingreds_weights_df,
            left_on="Code", right_on="Food code", how="outer") \
            .drop(columns=["Food code"])

        g = nx.from_pandas_edgelist(links, source='Source', target='Target',
                                    edge_attr=["Food ingredient weight", "Food ingredient weight normalized"]
                                    )

        attrs = nodes.set_index("Code").to_dict("index")
        nx.set_node_attributes(g, attrs)

        if "Size" not in nodes.columns:
            nodes["Size"] = nodes["Code"].apply(lambda x: g.degree(x))

        """"""

        def p4_selector(row):

            if np.isnan(row["<p4>"]):
                return row["p4"]

            return row["<p4>"]

        nodes["<p4>_ingred_p4"] = nodes.apply(p4_selector, axis=1)
        """"""

        if path_save is not None:
            links.to_csv("{}/food_ingredients_{}_links.csv".format(path_save, name), index=False)
            nodes.to_csv("{}/food_ingredients_{}_nodes.csv".format(path_save, name), index=False)

        fig = None

        if WWEIA_pie_chart is True:
            df = (nodes[nodes["Is ingredient"] == 0].groupby(["WWEIA Category description"])
                  .count()
                  .rename(columns={"Code": "Count items"})
                  .reset_index())

            fig = px.pie(df, values='Count items', names='WWEIA Category description', width=900, height=400,
                         title="Categories of {} (Total {} food-items)".format(name, df["Count items"].sum()))
            # fig.show()

            fig.write_image("{}/WWEIA_categories_{}.png".format(path_save, name))
            plotly.offline.plot(fig, filename="{}/WWEIA_categories_{}.html".format(path_save, name),
                                auto_open=WWEIA_pie_chart_auto_open_html)
            pass

        return nodes, links, g, fig
        pass

    def query_food_food_cosine_similarity(self, food_ingredient_weight_nx_graph, edge_attr_weights, FI_nodes_df,
                                          path_save, name):
        FF_A_w = nx.adjacency_matrix(
            food_ingredient_weight_nx_graph,
            weight=edge_attr_weights,
            nodelist=food_ingredient_weight_nx_graph.nodes()).todense()

        food_ingredient_weight_nx_graph = food_ingredient_weight_nx_graph

        cosine_similarity_distance_A = 1 - sp.distance.cdist(FF_A_w, FF_A_w.T, 'cosine')

        """
        """
        triu_indices = np.triu_indices(cosine_similarity_distance_A.shape[0])

        nodes_list_preserved_indices = list(food_ingredient_weight_nx_graph.nodes)

        rows = []

        for i in range(0, len(triu_indices[0])):
            m = triu_indices[0][i]
            n = triu_indices[1][i]

            if m == n:
                continue

            if cosine_similarity_distance_A[m, n] > 0 and \
                    food_ingredient_weight_nx_graph.nodes[nodes_list_preserved_indices[m]]["Is ingredient"] == 0 and \
                    food_ingredient_weight_nx_graph.nodes[nodes_list_preserved_indices[n]]["Is ingredient"] == 0:
                #         print((m , n))
                source_node = nodes_list_preserved_indices[m]
                target_node = nodes_list_preserved_indices[n]
                edge_weight = cosine_similarity_distance_A[m, n]

                rows.append((source_node, target_node, edge_weight))
                pass
            pass

        food_food_cosine_similarity_links = pd.DataFrame(data=rows,
                                                         columns=['Source', 'Target', 'Similarity'])

        FI_nodes_df = FI_nodes_df[FI_nodes_df["Is ingredient"] == 0]

        food_food_cosine_similarity_links = (
            pd.merge(food_food_cosine_similarity_links,
                     FI_nodes_df[["Code", "Label_full", "WWEIA Category description"]],
                     left_on="Source", right_on="Code", how="inner")
                .drop(columns=["Code"])
                .rename(columns={"Label_full": "Source_label", "WWEIA Category description": "Source_category"})
        )

        food_food_cosine_similarity_links = (
            pd.merge(food_food_cosine_similarity_links,
                     FI_nodes_df[["Code", "Label_full", "WWEIA Category description"]],
                     left_on="Target", right_on="Code")
                .drop(columns=["Code"])
                .rename(columns={"Label_full": "Target_label", "WWEIA Category description": "Target_category"})
        )

        """
        scale the weights between (0.1, 1)
        """
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 1))
        # float_array = food_food_links[["Weight"]].values.astype(float)
        # scaled_array = min_max_scaler.fit_transform(float_array)
        # food_food_links["Weight normalized"] = scaled_array

        if "Size" not in FI_nodes_df.columns:
            FI_nodes_df["Size"] = FI_nodes_df["Code"].apply(lambda x: food_ingredient_weight_nx_graph.degree(x))

        if path_save is not None:
            food_food_cosine_similarity_links.to_csv("{}/food_cosine_similarity_{}_links.csv".format(path_save, name),
                                                     index=False)
            FI_nodes_df.to_csv("{}/food_cosine_similarity_{}_nodes.csv".format(path_save, name), index=False)
        #

        return FI_nodes_df, food_food_cosine_similarity_links, cosine_similarity_distance_A
        pass

    def run_classifier_on_all_foods_and_ingredients(self, clf, cls_columns, include_ingredients):

        print(">> Running classifier_DEL_ME with {} Num Nutrients: {}".format(len(cls_columns), cls_columns))

        fix_cols_for_older_FNDDS = {
            "Food Code": "Food code",
            "Nutrient Code": "Nutrient code",
            "Nutrient Value": "Nutrient value"
        }

        self.df["FNDDSNutVal"] = self.df["FNDDSNutVal"].rename(columns=fix_cols_for_older_FNDDS)
        self.df["MainFoodDesc"] = self.df["MainFoodDesc"].rename(columns=fix_cols_for_older_FNDDS)

        foods_nuts = pd.merge(
            self.df["FNDDSNutVal"][["Food code", "Nutrient code", "Nutrient value"]],
            self.df["NutDesc"],
            on="Nutrient code")

        foods_nuts = pd.merge(foods_nuts,
                              self.df["MainFoodDesc"][["Food code", "Main food description"]],
                              on="Food code").rename(columns={
            "Main food description": "Description",
            "Food code": "Code"
        })

        foods_nuts["Is ingredient"] = 0

        if include_ingredients is True:

            ingred_nuts = pd.merge(
                self.df["INGREDNutVal"][["Ingredient code", "Nutrient code", "Nutrient value", "SR description"]],
                self.df["NutDesc"],
                on="Nutrient code"
            ).rename(columns={
                "SR description": "Description",
                "Ingredient code": "Code"
            })

            # ingred_nuts = pd.merge(ingred_nuts, self.df["FNDDSIngred"][["Ingredient code", "Ingredient description"]],
            #                        on="Ingredient code").rename(columns={
            #     "Ingredient description": "Description",
            #     "Ingredient code": "Code"
            # })

            ingred_nuts["Is ingredient"] = 1

            food_ingred_nuts = pd.concat([
                foods_nuts,
                ingred_nuts
            ])
            pass
        else:
            food_ingred_nuts = foods_nuts
            pass

        convert_nutrient_unit_to_grams(food_ingred_nuts, "Nutrient value")

        column_pivot_nutrient = 'Nutrient code'

        food_ingred_nuts_pivot = pd.pivot_table(food_ingred_nuts, values='Nutrient value grams',
                                                index=['Code', 'Description', "Is ingredient"],
                                                columns=[column_pivot_nutrient], aggfunc=np.mean)

        if False:
            if len(self.df["NutDesc"][self.df["NutDesc"]["Nutrient description"] == "Cryptoxanthin, beta"]) == 0:
                # this is for 2001 FNDDS
                vitamin_A_by_sum_of_columns[3] = "Cryptoxanthin"

        """This part is for working with older FNDDS"""
        nuts_99_panel = giulia_classifier_99_nuts_codes_df.set_index("Nutrient code")[
            "Nutrient description"].to_dict()

        # Tests if an older FNDDS has a nutrient that is not in our predication panel
        for c in food_ingred_nuts_pivot.columns:
            if c not in nuts_99_panel:
                food_ingred_nuts_pivot = food_ingred_nuts_pivot.drop(columns=c)

                print("[WARNING] Nutrient '{} - {}' is not in 99 panel so its removed.".format(
                    c,
                    self.df['NutDesc'][self.df['NutDesc']['Nutrient code'] == c]['Nutrient description'].values
                ))
                pass
            pass

        """Total Vitamin A is not given by FNDDS so manually calculate it """
        food_ingred_nuts_pivot['Total Vitamin A'] = food_ingred_nuts_pivot[
            [nut_code for nut_code, nut_desc in giulia_calc_vitamin_A_by_sum_of_columns]
        ].sum(axis=1)

        food_ingred_nuts_pivot = food_ingred_nuts_pivot.rename(
            columns=giulia_classifier_99_nuts_codes_df.set_index("Nutrient code")[
                "Nutrient description"].to_dict())

        # food_ingred_nuts_pivot = food_ingred_nuts_pivot[cls_columns]

        """
        """
        food_ingred_nuts_pivot_log = np.log(food_ingred_nuts_pivot)
        # food_ingred_nuts_pivot = np.log(food_ingred_nuts_pivot)

        food_ingred_nuts_pivot_log = food_ingred_nuts_pivot_log.fillna(-20)

        food_ingred_nuts_pivot_log = food_ingred_nuts_pivot_log.replace([np.inf, -np.inf], -20)

        predicted_class = clf.predict(food_ingred_nuts_pivot_log[cls_columns])

        predict_prob = clf.predict_proba(food_ingred_nuts_pivot_log[cls_columns])

        """
        """

        food_ingred_nuts_pivot.insert(0, "class", predicted_class, True)

        food_ingred_nuts_pivot.insert(1, "p4", predict_prob[:, 3], True)
        food_ingred_nuts_pivot.insert(1, "p3", predict_prob[:, 2], True)
        food_ingred_nuts_pivot.insert(1, "p2", predict_prob[:, 1], True)
        food_ingred_nuts_pivot.insert(1, "p1", predict_prob[:, 0], True)

        food_ingred_nuts_pivot = food_ingred_nuts_pivot.reset_index()

        if include_ingredients is False:
            food_ingred_nuts_pivot = food_ingred_nuts_pivot.rename(columns={
                "Description": "Main food description",
                "Code": "Food code"
            })

            food_ingred_nuts_pivot = food_ingred_nuts_pivot.drop(columns=['Is ingredient'])

        # Calculate FPro
        food_ingred_nuts_pivot["Processing index J1"] = (
                (1 - food_ingred_nuts_pivot['p1'] + food_ingred_nuts_pivot['p4']) / 2
        )

        return food_ingred_nuts_pivot

    def calculate_nuts_run_classifier(self, food_code, ingreds, clf, retention_factors=None,
                                      use_calculated_nutrients=True):
        if retention_factors is None:
            retention_factors = load_USDA_Nutrient_Retention_Factors()
            pass

        print(self.find_food(food_code=food_code)["Main food description"])

        ingreds_nuts = pd.merge(ingreds, self.get_ingredient_nutrients(), on="Ingredient code")

        ingreds_nuts_ret = pd.merge(ingreds_nuts, retention_factors,
                                    left_on=["Retention code", "Nutrient code"],
                                    right_on=["Retn_Code", "Nutr_No"], how="left"
                                    )

        ingreds_nuts_ret["Retn_Factor"] = ingreds_nuts_ret["Retn_Factor"].fillna(1)

        ingreds_nuts_ret = ingreds_nuts_ret[["Food code", "Ingredient code", "Ingredient description", "Retention code",
                                             "Ingredient weight", "Food ingredient weight",
                                             "Food ingredient weight normalized",
                                             # "Parent food code",  # "SR description",
                                             "Nutrient code", "Nutrient value", "Nutrient description", "Tagname",
                                             "Unit", "Decimals", "RetnDesc", "Nutr_No", "NutrDesc", "Retn_Factor"]]

        food_ingreds_nuts = (
                ingreds_nuts_ret["Nutrient value"]
                * ingreds_nuts_ret["Food ingredient weight normalized"]
                * ingreds_nuts_ret["Retn_Factor"]
                #      * (1 + (0.060 * 2))
                * 1.00
        )

        ingreds_nuts_ret.insert(10, "Calculated Nutrient value", food_ingreds_nuts, True)

        food_nuts_calced = (
            ingreds_nuts_ret[
                ["Food code", "Nutrient code", "Nutrient description", "Unit", "Calculated Nutrient value"]]
                .groupby(["Food code", "Nutrient code", "Nutrient description", "Unit"])
                .sum()
        ).round(3)

        food_nuts_calced = pd.merge(
            food_nuts_calced,
            self.get_food_nutrients(food_code).set_index(["Food code", "Nutrient code"]),
            left_index=True, right_index=True
        ).drop(columns=["Start date", "End date"])

        food_nuts_calced["diff"] = food_nuts_calced["Nutrient value"] - food_nuts_calced["Calculated Nutrient value"]
        food_nuts_calced["diff_abs"] = (
                food_nuts_calced["Nutrient value"] - food_nuts_calced["Calculated Nutrient value"]).abs()
        # food_nuts_calced["FN"] = food_nuts_calced['Food Nutrient value'].apply(np.ceil)
        # food_nuts_calced["FN"] = food_nuts_calced['Food Nutrient value'].round(2)

        food_nuts_calced = food_nuts_calced.sort_values(by="diff_abs", ascending=False)

        food_nuts_calced = food_nuts_calced.reset_index()

        convert_nutrient_unit_to_grams(food_nuts_calced, nutrient_values_column="Calculated Nutrient value")
        convert_nutrient_unit_to_grams(food_nuts_calced, nutrient_values_column="Nutrient value")

        if use_calculated_nutrients is True:
            nutrient_values_column_name = "Calculated Nutrient value grams"
        else:
            nutrient_values_column_name = "Nutrient value grams"

        to_classifier = pd.pivot_table(food_nuts_calced, values=nutrient_values_column_name,
                                       index=["Food code"], columns=['Nutrient description'], aggfunc=np.mean)

        to_classifier = np.log(to_classifier)
        to_classifier = to_classifier.fillna(-20)
        to_classifier = to_classifier.replace([np.inf, -np.inf], -20)

        to_classifier = to_classifier.reset_index()

        predicted_class = clf.predict(to_classifier[giulia_classifier_order_cols_2015_2016])

        predicted_class = predicted_class + 1

        predict_prob = clf.predict_proba(to_classifier[giulia_classifier_order_cols_2015_2016])

        to_classifier.insert(2, "pred class", predicted_class, True)
        to_classifier.insert(3, "p4", predict_prob[:, 3], True)
        to_classifier.insert(3, "p3", predict_prob[:, 2], True)
        to_classifier.insert(3, "p2", predict_prob[:, 1], True)
        to_classifier.insert(3, "p1", predict_prob[:, 0], True)

        return to_classifier[
                   ["Food code", "pred class", "p1", "p2", "p3",
                    "p4"] + giulia_classifier_order_cols_2015_2016], food_nuts_calced

        # FNDDS_base.save_excel(to_classifier, "temp/t1.xlsx", True)

        # print(clf.n_features_)
        # food_nuts_calced
        # ingreds
        # ingreds_nuts_ret
        # With calculated nuts: 0	58146662	4	0.043889	0.005556	0.13787	0.812685

        pass

    def get_ingredient(self, ingredient_code=None):
        if ingredient_code is None:
            return self.df["FNDDSIngred"]
        else:
            return self.df["FNDDSIngred"][self.df["FNDDSIngred"]["Ingredient code"] == ingredient_code]

        pass

    def get_ingredients(self, food_code=None, break_food_code_as_ingredient=False, drop_dates=True):
        result = None
        if self.years == "2015_2016":
            column_name_weight = "Ingredient weight"
            column_name_ingredient_code = "Ingredient code"
        else:
            column_name_weight = "Weight"
            column_name_ingredient_code = "SR code"
            pass

        if food_code is not None:
            # print("[get_ingredients] {}".format(self.describe_food(food_code)))

            if self.years == "2015_2016":
                result = self.df["FNDDSIngred"][self.df["FNDDSIngred"]["Food code"] == food_code]
            else:
                result = self.df["FNDDSSRLinks"][self.df["FNDDSSRLinks"]["Food code"] == food_code]
        else:
            if self.years == "2015_2016":
                result = self.df["FNDDSIngred"]
            else:
                result = self.df["FNDDSSRLinks"]
                pass
            pass

        result = result.reset_index(drop=True)

        result["Food ingredient weight"] = result[column_name_weight]

        result["Food ingredient weight normalized"] = result[column_name_weight] / result[column_name_weight].sum()

        if drop_dates is True:
            result = result.drop(columns=["Start date", "End date"])

        if break_food_code_as_ingredient is False:
            return result

        result["Parent food code"] = -1
        result["Recursive level"] = 0
        result["Parent food code first"] = -1

        ingred_from_food_ids = []
        all_dfs = []

        ingred_from_food = result[result[column_name_ingredient_code].astype(str).str.len() == 8]

        ingred_from_food_ids += list(
            zip(ingred_from_food[column_name_ingredient_code], ingred_from_food["Food ingredient weight normalized"],
                ingred_from_food["Food ingredient weight"], ingred_from_food["Retention code"],
                ingred_from_food["Recursive level"], result["Parent food code first"]
                )
        )

        all_dfs.append(result[~result[column_name_ingredient_code].isin(ingred_from_food[column_name_ingredient_code])])

        while len(ingred_from_food_ids) > 0:
            parent_food_code, weight_fraction_in_parent, weight_in_parent, retention_code_parent, recursive_level, parent_food_code_first = \
                ingred_from_food_ids.pop(0)
            # print(weight_in_parent)
            # print("-- Food Code Given as Ingredient: {} | weighted_normed: {}".format(
            #     parent_food_code, weight_normed))

            result = self.get_ingredients(food_code=parent_food_code, break_food_code_as_ingredient=False) \
                .reset_index(drop=True)

            # result["Weight"] = result[column_name_weight].sum()
            result["Food ingredient weight"] = (result[column_name_weight] / result[
                column_name_weight].sum()) * weight_in_parent
            # result["Weight"] = weight_in_parent

            result["Food ingredient weight normalized"] = (result[column_name_weight] / result[
                column_name_weight].sum()) * weight_fraction_in_parent
            result["Parent food code"] = parent_food_code

            if recursive_level == 0:
                result["Parent food code first"] = parent_food_code
            else:
                result["Parent food code first"] = parent_food_code_first

            result["Parent retention code"] = retention_code_parent
            result["Recursive level"] = recursive_level + 1

            ingred_from_food = result[result[column_name_ingredient_code].astype(str).str.len() == 8]

            all_dfs.append(
                result[~result[column_name_ingredient_code].isin(ingred_from_food[column_name_ingredient_code])])

            ingred_from_food_ids += list(
                zip(ingred_from_food[column_name_ingredient_code],
                    ingred_from_food["Food ingredient weight normalized"],
                    result["Food ingredient weight"],
                    result["Retention code"],
                    result["Recursive level"],
                    result["Parent food code first"]
                    ))

            pass

        # all_dfs = [all_dfs[-1]] + all_dfs[1:-1] + [all_dfs[0]]

        all_ing = pd.concat(all_dfs)

        all_ing["Food code"] = food_code
        #
        # # all_ing  # ["Weight nomarlized"].sum()
        # all_ing.drop(columns=["index"]).reset_index(drop=True)

        return all_ing.reset_index(drop=True)  # .drop(columns=[column_name_weight])

    def find_ingredients_with_no_nutrients(self):

        raise Exception("This is not a stable concept ")
        ingredients_with_no_nutrients = self.df["FNDDSIngred"][~self.df["FNDDSIngred"]["Ingredient code"].isin(
            self.df["INGREDNutVal"]["Ingredient code"]
        )]

        ingredients_with_no_nutrients = ingredients_with_no_nutrients["Ingredient code"].unique()

        return self.df["FNDDSIngred"][self.df["FNDDSIngred"]["Ingredient code"].isin(ingredients_with_no_nutrients)]

        """
        f16.find_ingredients_with_no_nutrients()
all_ingreds_food = f16.df["FNDDSIngred"]["Ingredient code"].unique()
all_ingreds_nuts = f16.df["INGREDNutVal"]["Ingredient code"].unique()

"all_ingreds_food: {} all_ingreds_nuts: {}".format(len(all_ingreds_food), len(all_ingreds_nuts))

set(all_ingreds_food) - set(all_ingreds_nuts)
        """
        # return ingredients_with_no_nutrients
        pass

    def get_ingredient_nutrients(self, ingredient_code=None, drop_dates=True):

        if ingredient_code is None:
            nuts = self.df["INGREDNutVal"]
        else:
            nuts = self.df["INGREDNutVal"][self.df["INGREDNutVal"]["Ingredient code"] == ingredient_code]

        nuts = pd.merge(nuts, self.df["NutDesc"], on="Nutrient code")

        if drop_dates is True:
            nuts = nuts.drop(columns=["Start date", "End date"])

        nuts["Nutrient value in grams"] = \
            nuts.apply(lambda row: convert_to_g(row["Nutrient value"], row["Unit"]), axis=1)

        return nuts

    def query_get_all_ingredients_code_description(self):
        ingred = self.df['FNDDSIngred']

        all_ingred_code = ingred["Ingredient code"].unique()
        # all_ingred_code = [c for c in all_ingred_code if len(str(c)) != 8]

        all_ingred = ingred[ingred["Ingredient code"].isin(all_ingred_code)][
            ["Ingredient code", "Ingredient description"]]

        all_ingred = all_ingred.drop_duplicates()

        all_ingred = all_ingred.sort_values(by="Ingredient code").reset_index(drop=True)

        return all_ingred

    def keep_query_NOVA_or_classifier_group4(self, NOVA_classification):
        df_NOVA_or_classifier_group4 = NOVA_classification[
            (NOVA_classification['NOVA class imported'] == 4) | (NOVA_classification['class'] == 4 - 1)]

        df_NOVA_or_classifier_group4 = df_NOVA_or_classifier_group4[df_NOVA_or_classifier_group4["Is ingredient"] == 0]

        df_NOVA_or_classifier_group4 = df_NOVA_or_classifier_group4[
            list(df_NOVA_or_classifier_group4.columns[:5]) + list(df_NOVA_or_classifier_group4.columns[-5:])]

        df_NOVA_or_classifier_group4 = df_NOVA_or_classifier_group4.drop(
            columns=["Giulia class imported", "Is ingredient"])

        df_NOVA_or_classifier_group4 = df_NOVA_or_classifier_group4.sort_values(by="p4")

        df_NOVA_or_classifier_group4 = pd.merge(df_NOVA_or_classifier_group4,
                                                self.df["MainFoodDesc"][["Food code", "WWEIA Category description"]],
                                                left_on="Code", right_on="Food code").drop(columns=["Food code"])

        save_excel(df_NOVA_or_classifier_group4.sort_values(by="p4"), "temp/NOVA_or_classifier_group4.xlsx",
                   open=True)
        return df_NOVA_or_classifier_group4

    def get_weighted_avg_probs(self, food_code, classified_df):

        r = pd.concat([
            classified_df[classified_df["Code"] == food_code][
                ["Code", "Is ingredient", "Description", "pred class", "p1", "p2", "p3", "p4"]].reset_index(drop=True),

            pd.merge(self.get_ingredients(food_code=food_code, break_food_code_as_ingredient=True).rename(
                columns={"Ingredient description": "Description"}),
                classified_df[["Code", "Is ingredient", "pred class", "p1", "p2", "p3", "p4"]],
                left_on="Ingredient code", right_on="Code")
        ]).drop(columns="Food code")[["Code", "Is ingredient", "Description", "pred class", "p1", "p2", "p3", "p4",
                                      "Food ingredient weight"]].reset_index(drop=True)

        return r

    def get_ingredients_of_foods(self, food_df, break_food_code_as_ingredient):

        # 51301120     11100000do_clusters
        food_ingreds = []

        for index, row in food_df.iterrows():
            food_code = row["Food code"]

            ingredients = self.get_ingredients(food_code=food_code,
                                               break_food_code_as_ingredient=break_food_code_as_ingredient)

            food_ingreds.append(ingredients)
            pass

        return pd.concat(food_ingreds)

    def get_weighted_avg_probs_foods(self, food_df, classification_df):

        if "Main food description" in food_df.columns:
            food_df = food_df.rename(columns={"Main food description": "Description"})

        # 51301120     11100000
        food_ingreds = []

        food_ingreds_weights = []

        for index, row in food_df.iterrows():
            food_code = row["Food code"]

            # food_code, Ingredient code, Food ingredient weight

            ingredients = self.get_ingredients(food_code=food_code, break_food_code_as_ingredient=True)

            # Filter water
            ingredients = ingredients[ingredients["Ingredient code"] != 14411]

            food_ingreds_weights.append(ingredients[["Food code", "Ingredient code", "Food ingredient weight"]])

            food_ingreds.append(ingredients)

            pass

        """
        Average weighted p values
        """
        food_ingreds_weights_df = pd.concat(food_ingreds_weights)
        del food_ingreds_weights

        food_ingreds_weights_df = pd.merge(food_ingreds_weights_df, classification_df[["Code", "p1", "p2", "p3", "p4"]],
                                           left_on="Ingredient code", right_on="Code")

        def s(data):
            data["Sum weight"] = data["Food ingredient weight"].sum()

            # TODO read this from source directly since WATER might be removed as an ingredient
            data["Count ingred"] = len(data)
            return data

        food_ingreds_weights_df = food_ingreds_weights_df.groupby("Food code").apply(s)

        food_ingreds_weights_df["<p1>"] = (food_ingreds_weights_df["Food ingredient weight"] * food_ingreds_weights_df[
            "p1"]) / food_ingreds_weights_df["Sum weight"]
        food_ingreds_weights_df["<p2>"] = (food_ingreds_weights_df["Food ingredient weight"] * food_ingreds_weights_df[
            "p2"]) / food_ingreds_weights_df["Sum weight"]
        food_ingreds_weights_df["<p3>"] = (food_ingreds_weights_df["Food ingredient weight"] * food_ingreds_weights_df[
            "p3"]) / food_ingreds_weights_df["Sum weight"]
        food_ingreds_weights_df["<p4>"] = (food_ingreds_weights_df["Food ingredient weight"] * food_ingreds_weights_df[
            "p4"]) / food_ingreds_weights_df["Sum weight"]

        # food_ingreds_weights_df = food_ingreds_weights_df.groupby("Food code").sum().reset_index()
        food_ingreds_weights_df = food_ingreds_weights_df.groupby("Food code").agg(
            {
                "<p1>": np.sum,
                "<p2>": np.sum,
                "<p3>": np.sum,
                "<p4>": np.sum,
                "Count ingred": np.mean
            }).reset_index()

        food_ingreds_weights_df = food_ingreds_weights_df[["Food code", "Count ingred", "<p1>", "<p2>", "<p3>", "<p4>"]]

        food_ingreds_weights_df = pd.merge(
            food_ingreds_weights_df,
            classification_df[["Code", "Description", "p1", "p2", "p3", "p4"]].rename(columns={"Code": "Food code"}),
            on="Food code"
        )[["Food code", "Description", "Count ingred", "<p1>", "<p2>", "<p3>", "<p4>", "p1", "p2", "p3", "p4"]]

        return food_ingreds_weights_df

    def __str__(self):
        return ("Dataset {}: {} with tables {}".format(
            self.desc, self.years,
            [(t, str(len(self.df[t])) + " rows") for t in self.tables_loaded])
        )
        pass

    pass


# plot_fig_ranks
def fig_ranks(df, y_column_name, chart_title, color_map="color_discrete_map", showscale=True,
              y_column_title=None, x_column_title='Item', color_value_column=None, height=800, width=1000,
              remove_all_labels=False, limit_num_chars_x_axis=40, textfont_size=14, text_of_bar_column_name=None,
              fixed_color="brown", x_axis_label_font=11, y_axis_label_font=11, x_axis_tick_angel=45,
              hover_data=None, categoryorder='total descending', color_continuous_scale=None,
              range_color=None, auto_open_html=False, save_html=False, save_png=False,
              x_column_name='Main food description', pred_class_column_name='pred class',
              update_traces_args=None, plotly_layout_args=None):
    #
    if plotly_layout_args is None:
        plotly_layout_args = {}

    p4_gradient_continues_scale = [
        # "rgb(1.0, 0.977362552864283, 0.7820222991157247)", # its red dont use it!
        # "rgb(255,250,250)", # snow
        "rgb(240,255,240)",  # honeydew
        "rgb(240,255,255)",  # azure
        "rgb(0.9982622068435217, 0.9338715878508266, 0.6625297962322184)",
        "rgb(0.996078431372549, 0.8701730103806228, 0.5259976931949251)",
        "rgb(0.996078431372549, 0.7786389850057671, 0.331118800461361)",
        "rgb(0.996078431372549, 0.6608381391772395, 0.21454825067281819)",
        "rgb(0.9706113033448673, 0.5419915417147251, 0.1310726643598616)",
        "rgb(0.9151557093425606, 0.42758938869665514, 0.07261822376009228)",
        "rgb(0.8206689734717417, 0.3212918108419838, 0.01946943483275663)",
        "rgb(0.6886274509803921, 0.24562860438292963, 0.012210688196847366)",
        "rgb(0.5443137254901961, 0.1875432525951557, 0.017870049980776622)"
    ]

    if update_traces_args is None:
        update_traces_args = {}

    if color_continuous_scale is None:
        color_continuous_scale = p4_gradient_continues_scale

    if range_color is None:
        range_color = [df[color_value_column].min(), df[color_value_column].max()]

    df = df.copy()

    if pred_class_column_name in df.columns:
        df['G'] = df[pred_class_column_name].map(
            {1: 'Unprocessed', 2: 'processed culinary', 3: 'processed', 4: 'ultra processed'})

    if color_map == "color_discrete_map":
        color_discrete_map = {'ultra processed': 'rgb(232, 139, 196)',
                              'processed': 'rgb(142, 161, 204)',
                              'processed culinary': 'rgb(254, 142, 99)',
                              'Unprocessed': 'rgb(104, 200, 163)'}

        fig = px.bar(df, x=df[x_column_name].map(
            lambda x: x[0:limit_num_chars_x_axis] + '...' if len(x) > limit_num_chars_x_axis else x),
                     y=y_column_name, hover_data=hover_data, range_color=range_color,
                     color="G", height=height, width=width,
                     color_discrete_map=color_discrete_map,
                     ).update_xaxes(categoryorder=categoryorder)

    elif color_map == "plain":
        fig = px.bar(df, x=df[x_column_name].map(
            lambda x: x[0:limit_num_chars_x_axis] + '...' if len(x) > limit_num_chars_x_axis else x),
                     color_continuous_scale=color_continuous_scale, range_color=range_color,
                     y=y_column_name, hover_data=hover_data, height=height, width=width,
                     ).update_xaxes(categoryorder=categoryorder)

    # elif color_map == "plain_horizontal":
    #     fig = px.bar(
    #         df,
    #         y=df[x_column_name].map(
    #             lambda x: x[0:limit_num_chars_x_axis] + '...' if len(x) > limit_num_chars_x_axis else x),
    #         x=y_column_name, orientation='h', text="subCategory",
    #         color_continuous_scale=color_continuous_scale,
    #         # color_continuous_scale=p4_gradient_continues_scale,
    #         range_color=range_color, color=color_value_column,
    #         hover_data=hover_data, height=height, width=width
    #     ).update_yaxes(categoryorder=categoryorder)
    #     pass

    elif color_map == "p4_gradient":
        print("x")
        fig = px.bar(
            df,
            x=df[x_column_name].map(
                lambda x: x[0:limit_num_chars_x_axis] + '...' if len(x) > limit_num_chars_x_axis else x),
            y=y_column_name,
            color=color_value_column,
            #                      color_continuous_scale=px.colors.sequential.YlOrBr,
            color_continuous_scale=color_continuous_scale, range_color=range_color,
            hover_data=hover_data, height=height, width=width
        ).update_xaxes(categoryorder=categoryorder)

    elif color_map in ["p4_gradient_horizontal", "horizontal"]:
        fig = px.bar(
            df,
            y=df[x_column_name].map(
                lambda x: x[0:limit_num_chars_x_axis] + '...' if len(x) > limit_num_chars_x_axis else x),
            x=y_column_name, orientation='h', text=text_of_bar_column_name,
            color=color_value_column,
            #                      color_continuous_scale=px.colors.sequential.YlOrBr,
            color_continuous_scale=color_continuous_scale, range_color=range_color,
            hover_data=hover_data, height=height, width=width
        ).update_yaxes(categoryorder=categoryorder)

        fig.update_traces(**update_traces_args)

    elif color_map == "custom_fixed_color":
        df['C_fixed'] = 'x'

        fig = px.bar(df, x=df['Main food description'].map(
            lambda x: x[0:limit_num_chars_x_axis] + '...' if len(x) > limit_num_chars_x_axis else x),
                     color="C_fixed", y=y_column_name, hover_data=hover_data, range_color=range_color,
                     height=height, width=width, color_discrete_map={'x': fixed_color}
                     ).update_xaxes(categoryorder=categoryorder)

        pass

    fig.update_layout({
        #         "plot_bgcolor": "rgb(0, 0, 0)",
        'paper_bgcolor': 'rgb(255,255,255)',
        'plot_bgcolor': 'rgba(0,0,0, 0)',  # 'rgb(236,232,232)',
        "xaxis": {
            "linecolor": "black",
            'showline': True,
            'linewidth': 1,
            "gridcolor": "rgb(236,232,232)",
            # "zerolinecolor": "rgb(74, 134, 232)"
        },
        "yaxis": {
            "linecolor": "black",
            'showline': True,
            'linewidth': 1,
            # "gridcolor": "rgb(159, 197, 232)",
            # "zerolinecolor": "rgb(74, 134, 232)"
        },
        "coloraxis": {
            "showscale": showscale
        }
    }
        #         margin=dict(b=400) # r=20
    )

    fig.update_layout(font_family="Times New Roman", font_color="black", **plotly_layout_args)

    #     fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='rgb(120, 120, 120)', size=18))
    #     fig.update_yaxes(tickfont=dict(family='Rockwell', color='rgb(120, 120, 120)', size=18))
    fig.update_xaxes(tickangle=x_axis_tick_angel, tickfont=dict(color='black', size=x_axis_label_font))
    fig.update_yaxes(tickfont=dict(color='black', size=y_axis_label_font))

    fig.update_traces(textposition='inside', textfont_size=textfont_size)

    if y_column_title is None:
        y_column_title = y_column_name

    if remove_all_labels is True:
        fig.update_layout({"title": ''})
        fig.update_xaxes(title_text='')
        fig.update_yaxes(title_text='')
        pass
    else:
        fig.update_layout({"title": chart_title})
        fig.update_xaxes(title_text=x_column_title)
        fig.update_yaxes(title_text=y_column_title)
        pass

    if save_png:
        file_path = "temp\{}_{}.png".format(chart_title, color_map).replace(' ', '_')
        # fig.write_image(file_path, dpi=150)
        pio.write_image(fig, file_path, scale=5)
        print(file_path)

    if save_html:
        file_path = "temp\{}_{}.html".format(chart_title, color_map).replace(' ', '_')
        plotly.offline.plot(fig, filename="temp\{}_{}.html".format(chart_title, color_map), auto_open=auto_open_html)
        print(file_path)

    return fig


# region GroceryStore Data
import collections

# UNP_UP_eigen_values_vectors_FNDDS_2001_2016
UNP_UP_eigen_values_vectors = {
    "FNDDS_2001_2016": {
        'UNP_eigenvalues': np.array([5.24904112e-02, 8.70651614e-03, 1.19995080e-03, 2.13373805e-33]),
        'UNP_eigenvectors': np.array([[-0.83243864, 0.07331732, 0.29728702, 0.46183429],
                                      [-0.07390833, -0.0827349, 0.7766854, -0.62004217],
                                      [-0.22711996, 0.85894092, -0.24161999, -0.39020097],
                                      [0.5, 0.5, 0.5, 0.5]]),
        'UP_eigenvalues': np.array([3.67295262e-02, 4.58553962e-03, 9.41871310e-04, 1.74038685e-33]),
        'UP_eigenvectors': np.array([[0.14263652, 0.03955661, 0.60226226, -0.7844554],
                                     [0.68147913, 0.25063322, -0.60453444, -0.32757791],
                                     [-0.51501555, 0.82802069, -0.14771012, -0.16529502],
                                     [-0.5, -0.5, -0.5, -0.5]])},
    "FNDDS_2001_2018": {
        'UNP_eigenvalues': np.array([5.14199072e-02, 1.09872633e-02, 1.16273066e-03, 6.24999651e-33]),
        'UNP_eigenvectors': np.array([[-0.83535703, 0.06589981, 0.35469835, 0.41475887],
                                      [-0.00933695, -0.09647115, 0.75469051, -0.6488824],
                                      [-0.22823551, 0.85810869, -0.23373343, -0.39613975],
                                      [-0.5, -0.5, -0.5, -0.5]]),
        'UP_eigenvalues': np.array([3.82473227e-02, 4.54454121e-03, 1.02371672e-03, 1.88347168e-33]),
        'UP_eigenvectors': np.array([[0.14084882, 0.03940318, 0.60354669, -0.78379869],
                                     [0.67119824, 0.26748316, -0.60605992, -0.33262148],
                                     [-0.52882372, 0.82273942, -0.13573051, -0.15818519],
                                     [-0.5, -0.5, -0.5, -0.5]])}
}

nutrient_panels_grocery_store_data = {
    "12P": [
        'protein', 'totalFat', 'carbohydrates', 'sugar', 'fiber', 'calcium', 'iron',
        'sodium', 'vitaminC', 'cholesterol', 'saturatedFat', 'vitaminA'
    ],
    "11P": [
        'protein', 'totalFat', 'carbohydrates', 'sugar', 'fiber', 'calcium', 'iron',
        'sodium', 'vitaminC', 'cholesterol', 'saturatedFat'
    ],
    '10P': ['protein', 'totalFat', 'carbohydrates', 'sugar', 'fiber', 'calcium',
            'iron', 'sodium', 'cholesterol', 'saturatedFat'],
    #      ['Protein', 'Total Fat', 'Carbohydrate', 'Sugars, total', ' Fiber, total dietary', 'Calcium',
    #       'Iron', 'Sodium', 'Cholesterol', 'Fatty acids, total saturated']
}


def add_fraction_nuts_panel(WH_products_df, columns_panel_12nuts, new_col_name):
    def fraction_12_nuts(row):
        total = len(columns_panel_12nuts)

        have_nuts = 0

        for nut in columns_panel_12nuts:
            if row[nut] is not None and bool(np.isnan(row[nut])) is False:
                have_nuts += 1

        return have_nuts / total

    WH_products_df[new_col_name] = WH_products_df.apply(fraction_12_nuts, axis=1)
    return WH_products_df


def prep_for_fig_ranks(WF_products_cat_seperated_preds, mode, ignore_brands, store,
                       nut_panel, do_nav_bar_for_walmart):
    """
    """

    brands_ignore = []
    if ignore_brands is True:
        brands_ignore = ['365']

    def create_cat_sub_cat(row):
        try:
            if store == "Walmart":
                res = "{} [{}]".format(
                    ", ".join([c for c in row["subCategory"] if
                               c.lower().startswith('nsc ') is do_nav_bar_for_walmart]),
                    ", ".join([c for c in row["category"] if
                               c not in brands_ignore and c.lower().startswith('nc ') is do_nav_bar_for_walmart])
                ).title().replace('-', ' ')
            else:
                res = "{} [{}]".format(
                    ", ".join([c for c in row["subCategory"]]),
                    ", ".join([c for c in row["category"] if c not in brands_ignore])
                ).title().replace('-', ' ')
        except:
            print(row)
            raise Exception('')
        return res

    WF_products_cat_seperated_preds["cat_sub_cat"] = WF_products_cat_seperated_preds.apply(
        lambda r: create_cat_sub_cat(r), axis=1)

    rename_cols = {"id": "count"}

    #     if "p4" in WF_products_cat_seperated_preds.columns:
    if mode == "preds":
        # agg_cols = {"id": np.size, "p1": np.mean, "p2": np.mean, "p3": np.mean, "p4": np.mean}
        agg_cols = {nut_panel + " " + p: np.mean for p in ["p1", "p2", "p3", "p4"]}
        agg_cols["id"] = np.size

        agg_cols.update({
            col: np.mean for col in WF_products_cat_seperated_preds.columns if col.startswith("PI ")
        })

        # rename_cols = {"id": "count", "p1": "<p1>", "p2": "<p2>", "p3": "<p3>", "p4": "<p4>"}

        rename_cols.update({
            nut_panel + " " + p: "<" + nut_panel + " " + p + ">" for p in ["p1", "p2", "p3", "p4"]
        })

        rename_cols.update({
            col: f"<{col}>" for col in WF_products_cat_seperated_preds.columns if col.startswith("PI ")
        })

    elif mode == "frac of panel nuts":
        agg_cols = {"id": np.size, f"frac_{nut_panel}_nuts": np.mean}
    else:
        agg_cols = {"id": np.size}
        pass

    # print("\n\n\nrename_cols:", rename_cols)
    # print("DF cols:", str(list(WF_products_cat_seperated_preds.columns)))

    WF_products_cat_seperated_preds_12_nuts = (
        WF_products_cat_seperated_preds
            .groupby(["cat_sub_cat"])
            .agg(agg_cols)
            .reset_index()
            .rename(columns=rename_cols)
            .sort_values(by="count", ascending=False)
    )

    WF_products_cat_seperated_preds_12_nuts = WF_products_cat_seperated_preds_12_nuts.rename(columns=rename_cols)
    # print("-------\nDF cols AFTER RENAME:", str(list(WF_products_cat_seperated_preds.columns)))

    WF_products_cat_seperated_preds_12_nuts["subCategory"] = WF_products_cat_seperated_preds_12_nuts[
        "cat_sub_cat"].apply(
        lambda x: x.split(" [")[0].title().replace("-", " ")
    )
    WF_products_cat_seperated_preds_12_nuts["category"] = WF_products_cat_seperated_preds_12_nuts["cat_sub_cat"].apply(
        lambda x: x.split(" [")[1][0:-1].title().replace("-", " ")
    )

    WF_products_cat_seperated_preds_12_nuts["subCategoryShort"] = WF_products_cat_seperated_preds_12_nuts[
        "subCategory"].apply(
        lambda x: x[0:6].strip() + ".." + x[-6:].strip() if len(x) > 13 else x
    )

    if mode == "preds":
        WF_products_cat_seperated_preds_12_nuts["c"] = \
            WF_products_cat_seperated_preds_12_nuts["cat_sub_cat"] + \
            "<br /><p1>:" + WF_products_cat_seperated_preds_12_nuts[f"<{nut_panel} p1>"].round(3).astype(str) + \
            "  <p2>:" + WF_products_cat_seperated_preds_12_nuts[f"<{nut_panel} p2>"].round(3).astype(str) + \
            "  <p3>:" + WF_products_cat_seperated_preds_12_nuts[f"<{nut_panel} p3>"].round(3).astype(str) + \
            "  <p4>:" + WF_products_cat_seperated_preds_12_nuts[f"<{nut_panel} p4>"].round(3).astype(str)
    #     elif mode == "frac of panel nuts":
    #         agg_cols = {"id": np.size, f"frac_{nut_panel}_nuts": np.mean}
    #         rename_cols = { "id": "count"}
    else:
        WF_products_cat_seperated_preds_12_nuts["c"] = WF_products_cat_seperated_preds_12_nuts["cat_sub_cat"]
        pass
    # WF_products_cat_seperated_preds_12_nuts
    return WF_products_cat_seperated_preds_12_nuts
    pass


# endregion


if __name__ == "__main__":
    print(NOVA_predictions_colors_dict)
    raise Exception()
    from scipy.cluster.hierarchy import dendrogram, linkage

    # flav = FNDDS("2007_2010", "flavonoid")
    # # flav.load_data(dataset_path='D:/GoogleDrive/Research/Foodome/FNDDS/2007-2010/Flavonoid_Database_0710.mdb',
    # #                FNDDS_tables=["MainFoodDesc", "FlavVal", "FlavDesc"])
    #
    # f16 = FNDDS("2015_2016", "FNDDS")
    # f16.load_data(dataset_path='D:/GoogleDrive/Research/Foodome/FNDDS/2015-2016/FNDDS_2015_2016.mdb',
    #               FNDDS_tables=["MainFoodDesc", "FNDDSNutVal", "NutDesc", "FNDDSIngred"])

    # clf = get_giulia_classifier("nutritionfacts")

    # from plotly.validators.scatter.marker import SymbolValidator
    #
    # print(SymbolValidator().values)

    # data = pd.read_csv("data/RFFNDDS_2009_10_recom_Giulia_HR_Beans_Raws_salt_12_nuts.csv", index_col=0)
    data = pd.read_csv("data/FNDDS/predictions/RFFNDDS_2009_10_recom_Giulia_Raws_12_nuts.csv", index_col=0)

    clusters = do_clusters(
        dynamic_cut_args={"deepSplit": 2},  # 'minAbsGap': 5.9, 'maxAbsCoreScatter': 0.04
        data=data, features=["p1", "p2", "p3", "p4"], NOVA_class_column="class",
        name="probs", description_column="Description", fontfamily='Times New Roman',
        save_path="temp", save_excel_and_cluster_map=False, open_excel=False,
        remove_heat_map_legend=True,
        # lut_cat_dynamic_tree={1: "#ffffff"}
    )

    # print(clusters)
    if False:
        name = "home recipe"
        # name = "chili"
        # name = "beef"
        # name = "pizza"
        # name = "sandwich"

        f16 = FNDDS("2015_2016", "FNDDS")
        f16.load_data(dataset_path='D:/GoogleDrive/Research/Foodome/FNDDS/2015-2016/FNDDS_2015_2016.mdb',
                      FNDDS_tables=["MainFoodDesc", "FNDDSNutVal", "NutDesc", "FNDDSIngred",
                                    "INGREDNutVal", "MoistAdjust"
                                    ])

        f = f16.find_food(name)
        # f = f16.find_food("chili", category="dish")
        # f = f16.find_food(keywords="", category="beef")

        # f = f[f["Main food description"].str.match("Chi")]
        # f = f[f["Food code"].isin([27111440, 27111430])]
        # f = f[:20]
        NOVA_classification = pd.read_csv("data/RFFNDDSpredS_cleaned_withsalt FNDDS 2009-2010.csv")

        FI_nodes, FI_links, FI_g, fig_pie = f16.query_network_food_ingredients(
            food_df=f, classification_df=NOVA_classification, WWEIA_pie_chart=True,
            WWEIA_pie_chart_auto_open_html=False,
            normalize_weights=False, labels_length=20, path_save="temp/networks/FI", name=name)

        if False:
            FF_nodes, FF_links, FF_g = f16.query_network_food_food(FI_nodes, FI_links, name=name,
                                                                   path_save="temp/networks/FF",
                                                                   weights_attr_to_sum="Food ingredient weight"
                                                                   #                                                     weights_attr_to_sum="Food ingredient weight normalized"
                                                                   )
            FFCD_nodes, FFCD_links, FFCD_distance_A = f16.query_food_food_cosine_similarity(
                food_ingredient_weight_nx_graph=FI_g, edge_attr_weights="Food ingredient weight",
                FI_nodes_df=FI_nodes, path_save="temp/networks/FFCD", name=name)
            pass

        # fig_pie.show()
        pass
    pass


def get_consumed_items_both_days(all_nhanes, diet_all, SEQN, year, FPro_sort=None, ascending_FPro=False):
    FPro_columns = ['Processing index J1', 'ens_FPS', 'ens_min_FPS']
    iFPro_columns = ['FPro WFDPI mean of both days sum',
                     'FPro RW WFDPI mean of both days sum',
                     'FPro WCDPI mean of both days sum',
                     'ens_FPro WFDPI mean of both days sum',
                     'ens_FPro RW WFDPI mean of both days sum',
                     'ens_FPro WCDPI mean of both days sum',
                     'ens_min_FPro WFDPI mean of both days sum',
                     'ens_min_FPro RW WFDPI mean of both days sum',
                     'ens_min_FPro WCDPI mean of both days sum']

    print('Warning! all FPro are from 58 nuts panel!')
    if year == 1999:
        fndds_df = all_nhanes[2001].fndds.NOVA_58_nuts_classified
    else:
        fndds_df = all_nhanes[year].fndds.NOVA_58_nuts_classified

    day1 = (
        all_nhanes[year].dr1iff_diet[
            all_nhanes[year].dr1iff_diet["SEQN"] == SEQN
            ][
            ["SEQN", "Food code", "Main food description", "DR1IGRMS", "DR1IKCAL",
             "Total gram consumed", "Consumption fraction grams",
             "Total kcal consumed", "Consumption fraction calories",
             ]]
            .rename(columns={"DR1IGRMS": "Grams", "DR1IKCAL": "Calories kcal"})
    )

    day1 = pd.merge(day1, fndds_df[['Food code'] + FPro_columns], on='Food code', how='left')

    day1["DR2IGRMS"] = None
    day1["DR2IKCAL"] = None

    day1['Day'] = 1

    both_days = [day1]

    if all_nhanes[year].dr2iff_diet is not None:
        day2 = all_nhanes[year].dr2iff_diet[all_nhanes[year].dr2iff_diet["SEQN"] == SEQN][
            ["SEQN", "Food code", "Main food description", "DR2IGRMS", "DR2IKCAL",
             "Total gram consumed", "Consumption fraction grams",
             "Total kcal consumed", "Consumption fraction calories",
             ]].rename(columns={"DR2IGRMS": "Grams", "DR2IKCAL": "Calories kcal"})

        day2 = pd.merge(day2, fndds_df[['Food code'] + FPro_columns], on='Food code', how='left')
        day2['Day'] = 2

        both_days.append(day2)
        pass

    consumed = pd.concat(both_days)
    #     return consumed
    consumed = pd.merge(consumed, diet_all[["SEQN", "RIAGENDR", "RIDAGEYR", "BMXBMI"] + iFPro_columns], on="SEQN")

    print(
        consumed.drop_duplicates(['Food code']).groupby('Day').agg(num_unique_dishes=('Food code', np.size))
    )

    consumed = consumed.rename(columns={'Processing index J1': 'FPS'})

    FPro_columns_renamed = ['FPS', 'ens_FPS', 'ens_min_FPS']

    consumed = (
        consumed[[
                     "SEQN", "Food code", "Day", "Main food description",
                     'FPS', 'ens_FPS', 'ens_min_FPS',
                     "Grams", "Calories kcal",
                     "Consumption fraction grams", "Total gram consumed",
                     "Consumption fraction calories", "Total kcal consumed",
                     "RIAGENDR", "RIDAGEYR", "BMXBMI"
                 ] + iFPro_columns]
            .sort_values(by='Day', ascending=True)
    )

    if FPro_sort is not None:
        consumed = consumed.sort_values(by=['Day', FPro_sort], ascending=[True, ascending_FPro])
        pass

    return consumed
