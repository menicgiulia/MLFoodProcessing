import os
from os.path import isfile, join
# import pyodbc
import pandas as pd
# import joblib
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# import matplotlib._color_data as mcd
# import matplotlib.patches as mpatch
################ For Volcano
from matplotlib.colors import ListedColormap
from matplotlib.ticker import Locator
from bioinfokit import analys, visuz
import matplotlib.patches as mpatch
from collections import OrderedDict


################ For Volcano

# import plotly.express as px
# import plotly
# import plotly.figure_factory as ff
# import plotly.graph_objs as go
# import plotly.io as pio
#
# from sklearn.decomposition import PCA
# from sklearn import preprocessing
# from sklearn.manifold import TSNE
# import umap
# import json
#
# import scipy.spatial as sp
# from collections import OrderedDict

# import networkx as nx
# import re


def draw_color_text_box(overlap, figsize, title, title_font_size=16, color_fontsize=12,
                        title_font_y=1.1, fontfamily=None):
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
        txt = ax.text(figsize[0] / 2, j + 0.5, '  ' + key, va='center', fontsize=color_fontsize,
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


"""

        if row["p_val"] == 0:
            star = '***'
        elif row["p_val"] <= 0.001:
            star = '**'
        elif row["p_val"] <= 0.01:
            star = '**'
        elif row["p_val"] <= 0.05:
            star = '*'
        elif row["p_val"] <= 0.1:
            star = '.'"""


def p_val_sig_level_symbol(p_val):
    star = ""
    #     0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    if p_val == 0:
        star = '***'
    elif p_val <= 0.001:
        star = '**'
    elif p_val <= 0.01:
        star = '**'
    elif p_val <= 0.05:
        star = '*'
    elif p_val <= 0.1:
        star = '.'
    return star  # x


def desc_reg(r_df, columns_reg_desc, rename_covars_dicts):
    def describe_coef(row):
        star = p_val_sig_level_symbol(row["p_val"])

        return "{} ({} {})".format(
            round(row["coef"], 3),
            round(row["p_val"], 3),
            star
        )
        pass

    r_df['code_desc'] = r_df.apply(lambda row: describe_coef(row), axis=1)

    if len(r_df[~r_df["resp_var"].isnull()]) == 0:
        # print("[Ignored Tab] empty r_df is given.")
        return None

    # resp_var_type

    spl_char = '$!@'

    if True:
        r_df['resp_var'] = r_df['resp_var'] + \
                           spl_char + r_df["resp_var_type"].astype(str) + \
                           spl_char + r_df["N"].astype(str) + \
                           spl_char + r_df["NA_count"].astype(str) + \
                           spl_char + r_df["num_covars"].astype(str) + \
                           spl_char + r_df["unique_val_counts"].astype(str) + \
                           spl_char + r_df["value_counts"].astype(str)

        q = r_df.pivot(index='resp_var', columns='covariate', values=['code_desc', "coef", "p_val", "summary"])
    else:
        q = r_df.pivot_table(index=['resp_var', 'resp_var_type'], columns='covariate',
                             values=['code_desc', "coef", "p_val", "summary"], aggfunc=lambda x: x)

    q = q.rename(columns=rename_covars_dicts)

    q.columns = q.columns.map('|'.join).str.strip('|')

    rename_cols_dicts = {}

    rename_cols_dicts.update({f'code_desc|{c}': c for c in columns_reg_desc})
    rename_cols_dicts.update({f'coef|{c}': f'{c} coef' for c in columns_reg_desc})
    rename_cols_dicts.update({f'p_val|{c}': f'{c} p_val' for c in columns_reg_desc})
    rename_cols_dicts.update({f'summary|{c}': f'{c} summary' for c in columns_reg_desc})

    q = q.rename(columns=rename_cols_dicts)

    if False:
        q = q.rename(columns={
            'code_desc|HEI-15': 'HEI-15', 'code_desc|RW.WFDPI': 'RW.WFDPI', 'code_desc|WCDPI': 'WCDPI',
            'code_desc|WFDPI': 'WFDPI',
            #
            'coef|HEI-15': 'HEI-15 coef', 'coef|RW.WFDPI': 'RW.WFDPI coef', 'coef|WCDPI': 'WCDPI coef',
            'coef|WFDPI': 'WFDPI coef',
            #
            'p_val|HEI-15': 'HEI-15 p_val', 'p_val|RW.WFDPI': 'RW.WFDPI p_val', 'p_val|WCDPI': 'WCDPI p_val',
            'p_val|WFDPI': 'WFDPI p_val',
            #
            'summary|HEI-15': 'HEI-15 summary', 'summary|RW.WFDPI': 'RW.WFDPI summary',
            'summary|WFDPI': 'WFDPI summary',
            'summary|WCDPI': 'WCDPI summary'
        })
        pass

    q = q.reset_index()

    def restore_static_columns(row):
        values_spl = row["resp_var"].split(spl_char)

        row["resp_var"] = values_spl[0]
        row["resp_var_type"] = int(values_spl[1])
        row["N"] = int(values_spl[2])
        row['NA_count'] = int(values_spl[3])
        row['num_covars'] = int(values_spl[4])
        row['unique_val_counts'] = int(values_spl[5])

        if values_spl[6] == 'nan':
            row['value_counts'] = None
        else:
            row['value_counts'] = values_spl[6]
        return row

    q = q.apply(restore_static_columns, axis=1)

    # q["resp_var_type"] = q["resp_var"].apply(lambda x: int(x.split(spl_char)[1]))
    # q["N"] = q["resp_var"].apply(lambda x: x.split(spl_char)[2])
    # q['NA_count'] = q['resp_var'].apply(lambda x: int(x.split(spl_char)[3]))
    # q['num_covars'] = q['resp_var'].apply(lambda x: int(x.split(spl_char)[4]))
    # q['unique_val_counts'] = q['resp_var'].apply(lambda x: int(x.split(spl_char)[5]))
    # q['value_counts'] = q['resp_var'].apply(lambda x: x.split(spl_char)[6])
    # # This must be last!
    # q["resp_var"] = q["resp_var"].apply(lambda x: x.split(spl_char)[0])

    if False:
        print('@@@@@@@@', q.columns)

        print('+++++++', ["resp_var", "resp_var_type", "N",
                          'NA_count', 'num_covars', 'unique_val_counts', 'value_counts'] +
              [col for col_list in [(f'{m}', f'{m} coef', f'{m} p_val', f'{m} summary') for m in columns_reg_desc] for
               col
               in
               col_list])
        pass

    q = q[
        ["resp_var", "resp_var_type", "N",
         'NA_count', 'num_covars', 'unique_val_counts', 'value_counts'] +
        [col for col_list in [(f'{m}', f'{m} coef', f'{m} p_val', f'{m} summary') for m in columns_reg_desc] for col in
         col_list]
        ]
    # LBXSCH - Cholesterol (mg/dL) #'HEI-15 Both', 'HEI-15 D1'

    return q

    # return q[
    #     ["resp_var", "resp_var_type", "N",
    #      'NA_count', 'num_covars', 'unique_val_counts', 'value_counts'] +
    #     [col for col_list in [(f'{m}', f'{m} coef', f'{m} p_val') for m in columns_reg_desc] for col in col_list]
    #     # 'HEI-15', 'HEI-15 coef', 'HEI-15 p_val'
    #     # 'RW.WFDPI', 'RW.WFDPI coef', 'RW.WFDPI p_val',
    #     # 'WCDPI', 'WCDPI coef', 'WCDPI p_val',
    #     # 'WFDPI', 'WFDPI coef', 'WFDPI p_val',
    #     # 'HEI-15 summary', 'RW.WFDPI summary', 'WFDPI summary', 'WCDPI summary'
    # ]


def count_sig(path, columns_reg_desc, rename_covars_dicts, encoding="ISO-8859-1"):
    # print("XX ", path)
    desc_df = desc_reg(
        r_df=pd.read_csv(path, index_col=0, encoding=encoding),
        columns_reg_desc=columns_reg_desc,
        rename_covars_dicts=rename_covars_dicts
    )

    if desc_df is None:
        return None, None, None

    for col in columns_reg_desc:
        desc_df[col + " sig"] = desc_df[col].apply(lambda r: 1 if '*' in r else 0)
        pass

    file_name = path.split('/')[-1].replace('reg_analysis_boxcox_', '').replace('_', ' ').title()[:-4]

    stat1 = [
        (file_name, "Num vars", len(desc_df)),
    ]

    stat2 = {
        "Module": file_name,
        "Num Vars": len(desc_df)
    }

    desc_df.insert(loc=0, column='module', value=file_name)

    return desc_df, stat1, stat2


def helped_is_binary(x):
    if str(x) == "nan":
        return None

    if (len(x)) != 1:
        raise Exception()

    return list(x)[0]


# print("q2", len(reg_analysis_boxcox_all[reg_analysis_boxcox_all['var'] == 'LBXV1A']))
def is_categorical(row):
    ret = 0

    if row['HEI-15 coef'] is None or np.isnan(row['HEI-15 coef']):
        ret = None

    if row['is_categorical tested'] == 1:
        ret = 1

    if row['is_binary'] == 1 and row['is_categorical tested'] == 0:
        ret = 1
        #         if '/mL' not in row['var_desc']:
        #             ret = 1
        #         else:
        #             print("Var {} [{}] is not binary!!".format(row['var'], row['var_desc']))
        pass

    if str(row['is_ordinal']) == 'nan' or str(row['categorical_levels']) == 'nan':
        ret = None

    if str(row['is_ordinal']) != 'nan':
        if len(row['is_ordinal']) > 1:
            print(row)
            raise Exception('len[is_ordinal] > 1')

        is_ordinal = list(row['is_ordinal'])
        if len(is_ordinal) == 1 and is_ordinal[0] != 0:
            ret = 1
    if str(row['categorical_levels']) != 'nan':
        if len(row['categorical_levels']) > 1:
            print(row)
            raise Exception('len[categorical_levels] > 1')

        if len(row['categorical_levels']) > 0:
            ret = 1

    if str(row['category']) != 'nan':
        if len(row["category"]) > 1:
            print(row)
            raise Exception('len[category] > 1')
    if str(row['sub_category']) != 'nan':
        if len(row["sub_category"]) > 1:
            print(row['var'], ": has more than one sub category ", row["sub_category"])

    if str(row['is_ordinal']) != 'nan':
        if len(row["is_ordinal"]) > 1:
            print(row)
            raise Exception('len[is_ordinal] > 1')

    if str(row["N Patel"]) != 'nan':
        if len(row["N Patel"]) > 1:
            print(row)
            raise Exception('len[N Patel] > 1')

    return ret


def merge_all_regs(VarDescription, survey_year, survey_year_code_to_direction_name,
                   path_reg_analysis,
                   columns_reg_desc,
                   rename_covars_dicts,
                   only_work_on_module=None):
    """

    """
    path = os.path.abspath("{}/{}".format(path_reg_analysis, survey_year_code_to_direction_name[survey_year]))

    if os.path.exists(path) == False:
        print('Path does not exists: ', path)
        return None

    onlyfiles = [f for f in os.listdir(path) if isfile(join(path, f)) and f.startswith("reg_analysis_boxcox_")]

    stats1 = []
    stats2 = []

    all_desc_df = []

    for box_cox_analysis in onlyfiles:
        if only_work_on_module is not None:
            if only_work_on_module.lower() not in box_cox_analysis.lower():
                continue
        #     print("Working on {}".format(box_cox_analysis))
        #     if "Dioxins".lower() not in box_cox_analysis.lower():
        #         continue

        desc_df, stat1, stat2 = count_sig(path=path + '/' + box_cox_analysis,
                                          columns_reg_desc=columns_reg_desc,
                                          rename_covars_dicts=rename_covars_dicts
                                          )

        if desc_df is None:
            # print("File {} is empty!".format(
            #     path + '/' + box_cox_analysis
            # ))
            continue

        all_desc_df.append(desc_df)

        #     if 'LBXV1A' in list(desc_df.index):
        #         print(box_cox_analysis)
        #         q1 = desc_df.reset_index()
        #         print(len(q1[q1['resp_var'] == 'LBXV1A']))
        #         pass

        stats1 += stat1
        stats2.append(stat2)

        #         break
        pass

    stats1_df = pd.DataFrame(stats1, columns=["Module", 'Index', 'Count'])
    stats2_df = pd.DataFrame(stats2)

    # stats1_df = stats1_df[stats1_df["Index"].isin(["Num vars", "DPI Any sig", "HEI-15 sig"])]
    #     stats1_df = stats1_df[~stats1_df["Module"].isin(["Custom"])]
    #     sns.set(font_scale=1.5)

    """Merge All"""

    reg_analysis_boxcox_all = pd.concat(all_desc_df).reset_index(drop=True)
    reg_analysis_boxcox_all = reg_analysis_boxcox_all.rename(
        columns={"resp_var": "var", "resp_var_type": "var_type"}
    )

    # print("q1", len(reg_analysis_boxcox_all[reg_analysis_boxcox_all['var'] == 'LBXV1A']))
    """
    Add variable descriptions and save results for all modules.
    """

    def organize_series(s):
        res = [v for v in s]
        res.sort()
        return res

    # this will fix inconsistant values for a variables
    VarDescription_category = VarDescription[
        ["var", "var_desc", "category", "sub_category", "is_binary", "is_ordinal", "categorical_levels", "N", "series"]
    ].groupby(["var", "var_desc"]).agg({
        "category": lambda x: set([v for v in x if str(v) != "nan"]),
        "sub_category": lambda x: set([v for v in x if str(v) != "nan"]),
        "is_binary": lambda x: set([v for v in x if str(v) != "nan"]),
        "is_ordinal": lambda x: set([v for v in x if str(v) != "nan"]),
        "categorical_levels": lambda x: set([v for v in x if str(v) != "nan"]),
        # avariable can exist in multiple years but iassigned to different modules! like URXOP4
        #     "series": lambda x: organize_series(x),
        "N": lambda x: set([v for v in x if str(v) != "nan"])
    }).reset_index()

    VarDescription_category = VarDescription_category.rename(columns={"N": "N Patel"})

    def find_var_series_in_all_modules(var):
        # avariable can exist in multiple years but iassigned to different modules! like URXOP4

        var_series = VarDescription[VarDescription["var"] == var][["var", "series"]].groupby("var").agg({
            "series": lambda x: organize_series(x)
        })

        return var_series["series"].values[0]

    VarDescription_category["series"] = VarDescription_category["var"].apply(
        lambda var: find_var_series_in_all_modules(var))
    VarDescription_category["num series"] = VarDescription_category["series"].apply(lambda x: len(x))

    reg_analysis_boxcox_all = pd.merge(
        reg_analysis_boxcox_all,
        VarDescription_category,
        on="var", how="left")

    reg_analysis_boxcox_all['is_binary'] = reg_analysis_boxcox_all['is_binary'].apply(helped_is_binary)

    print('========', reg_analysis_boxcox_all.columns)

    reg_analysis_boxcox_all['is_categorical tested'] = (
        reg_analysis_boxcox_all['HEI-15 summary'].str.startswith("factor")
            .apply(lambda x: None if x is None else (1.0 if x is True else 0.0))
    )

    # print("q2", len(reg_analysis_boxcox_all[reg_analysis_boxcox_all['var'] == 'LBXV1A']))

    reg_analysis_boxcox_all['is_categorical'] = reg_analysis_boxcox_all.apply(is_categorical, axis=1)

    def helper_remove_set(x):
        if str(x) == 'nan':
            return ''

        if len(x) > 1:
            return ', '.join(list(x))
        elif len(x) == 1:
            return list(x)[0]
        else:
            return ''

    reg_analysis_boxcox_all["category"] = reg_analysis_boxcox_all["category"].apply(helper_remove_set)
    reg_analysis_boxcox_all["sub_category"] = reg_analysis_boxcox_all["sub_category"].apply(helper_remove_set)
    reg_analysis_boxcox_all["categorical_levels"] = reg_analysis_boxcox_all["categorical_levels"].apply(
        helper_remove_set)
    reg_analysis_boxcox_all["categorical_levels"] = reg_analysis_boxcox_all["categorical_levels"].apply(
        helper_remove_set)
    reg_analysis_boxcox_all["N Patel"] = reg_analysis_boxcox_all["N Patel"].apply(helper_remove_set)
    # reg_analysis_boxcox_all["N"] = reg_analysis_boxcox_all["N"].apply(helper_remove_set)

    if 'series' not in reg_analysis_boxcox_all.columns:
        reg_analysis_boxcox_all.insert(1, 'series', survey_year)
    else:
        reg_analysis_boxcox_all['series'] = survey_year

    return {
        "all_merged": reg_analysis_boxcox_all,
        "stats1_df": stats1_df,
        "stats2_df": stats2_df,
        "survey_year": survey_year
    }


##################### VOLCANO


class MinorSymLogLocator(Locator):
    """
    https://stackoverflow.com/questions/20470892/how-to-place-minor-ticks-on-symlog-scale

    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """

    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i - 1]
            if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))


# # https://reneshbedre.github.io/blog/volcano.html
def volcano(df="dataframe", lfc=None, pv=None, lfc_thr=1, pv_thr=0.05, color=("green", "grey", "red"), valpha=1,
            geneid=None, genenames=None, gfont=8, dim=(5, 5), r=300, ar=90, dotsize=8, markerdot="o",
            sign_line=False, gstyle=1, show=False, figtype='png', axtickfontsize=9,
            axtickfontname="Arial", axlabelfontsize=9, axlabelfontname="Arial", axxlabel=None,
            axylabel=None, xlm=None, ylm=None, plotlegend=False, legendpos='best',
            figname='volcano', legendanchor=None,
            legendlabels=['significant up', 'not significant', 'significant down'],
            x_log_scale=False, xlim=None, custom_color_column=None):
    _x = r'$ log_{2}(Fold Change)$'
    _y = r'$ -log_{10}(P_{value})$'
    color = color
    if custom_color_column is not None:
        color = list(df[custom_color_column].unique())
        # for i in range(0, 3 - len(color)):
        #     color.append('red')
        # print(color)

    # check if dataframe contains any non-numeric character
    assert visuz.general.check_for_nonnumeric(df[lfc]) == 0, 'dataframe contains non-numeric values in lfc column'
    assert visuz.general.check_for_nonnumeric(df[pv]) == 0, 'dataframe contains non-numeric values in pv column'
    # this is important to check if color or logpv exists and drop them as if you run multiple times same command
    # it may update old instance of df
    df = df.drop(['color_add_axy', 'logpv_add_axy'], axis=1, errors='ignore')

    # allow having less than 3 colors
    if custom_color_column is None:
        assert len(set(color)) == 3, 'unique color must be size ometrics_sings_not_changed_cohortsf 3'

        df.loc[(df[lfc] >= lfc_thr) & (df[pv] < pv_thr), 'color_add_axy'] = color[0]  # upregulated
        df.loc[(df[lfc] <= -lfc_thr) & (df[pv] < pv_thr), 'color_add_axy'] = color[2]  # downregulated
        df['color_add_axy'].fillna(color[1], inplace=True)  # intermediate
        pass
    else:
        df['color_add_axy'] = df[custom_color_column]
        pass

    df['logpv_add_axy'] = -(np.log10(df[pv]))
    # plot

    assign_values = {col: i for i, col in enumerate(color)}
    color_result_num = [assign_values[i] for i in df['color_add_axy']]

    # Allow having two colors!
    if False:
        assert len(set(
            color_result_num)) == 3, 'either significant or non-significant genes are missing; try to change lfc_thr or ' \
                                     'pv_thr to include  both significant and non-significant genes'

    fig = plt.figure()

    plt.subplots(figsize=dim)
    if plotlegend:
        s = plt.scatter(df[lfc], df['logpv_add_axy'], c=color_result_num, cmap=ListedColormap(color), alpha=valpha,
                        s=dotsize,
                        marker=markerdot)
        assert len(legendlabels) == 3, 'legendlabels must be size of 3'
        plt.legend(handles=s.legend_elements()[0], labels=legendlabels, loc=legendpos,
                   bbox_to_anchor=legendanchor)
    else:
        plt.scatter(df[lfc], df['logpv_add_axy'], c=color_result_num, cmap=ListedColormap(color), alpha=valpha,
                    s=dotsize,
                    marker=markerdot)
    #     plt.axes().set_xscale('log')

    if sign_line:
        plt.axhline(y=-np.log10(pv_thr), linestyle='--', color='#7d7d7d', linewidth=1)
        plt.axvline(x=lfc_thr, linestyle='--', color='#7d7d7d', linewidth=1)
        plt.axvline(x=-lfc_thr, linestyle='--', color='#7d7d7d', linewidth=1)

    # visuz.gene_exp.geneplot(df, geneid, lfc, lfc_thr, pv_thr, genenames, gfont, pv, gstyle)
    gene_exp.geneplot(df, geneid, lfc, lfc_thr, pv_thr, genenames, gfont, pv, gstyle)

    if x_log_scale is True:
        # https://stackoverflow.com/questions/43372499/plot-negative-values-on-a-log-scale
        #      Here's a function to make range split into bins for symlog scale: gist.github.com/artoby/0bcf790cfebed5805fbbb6a9853fe5d5. – artoby Jul 15 at 20:53
        plt.xscale("symlog", linthreshy=1e-1)

        if xlim is not None:
            plt.xlim(xlim)

        """
        Just to add minor ticks
        https://stackoverflow.com/questions/20470892/how-to-place-minor-ticks-on-symlog-scale
        """
        xaxis = plt.gca().xaxis
        xaxis.set_minor_locator(MinorSymLogLocator(1e-1))
        """"""
        pass

    if axxlabel:
        _x = axxlabel
    if axylabel:
        _y = axylabel
    # visuz.general.axis_labels(_x, _y, axlabelfontsize, axlabelfontname)
    # visuz.general.axis_ticks(xlm, ylm, axtickfontsize, axtickfontname, ar)
    # visuz.general.get_figure(show, r, figtype, figname)
    general.axis_labels(_x, _y, axlabelfontsize, axlabelfontname)
    general.axis_ticks(xlm, ylm, axtickfontsize, axtickfontname, ar)
    general.get_figure(show, r, figtype, figname)

    return fig


def select_best_coef_pval(metric_coef_pval, strategy):
    if metric_coef_pval is None or str(metric_coef_pval) == 'nan':
        return None

    coef_pval_list = [cp for cpl in metric_coef_pval.values() for cp in cpl]

    if strategy == 'coef largest':
        coef_pval_list = sorted(coef_pval_list, key=lambda x: abs(x[0]), reverse=True)
    elif strategy == 'pvalue smalles':
        coef_pval_list = sorted(coef_pval_list, key=lambda x: abs(x[1]), reverse=False)
    else:
        raise Exception('bad strategy')
    #     print(coef_pval_list)
    if len(coef_pval_list) > 0:
        return coef_pval_list[0]
    else:
        return None


# import pandas as pd
# import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
import matplotlib.cm as cmc
# import seaborn as sns
from matplotlib_venn import venn3, venn2
from random import sample
from functools import reduce
import sys
from matplotlib.colors import ListedColormap
from adjustText import adjust_text


def involcano(table="dataset_file", lfc="logFC", pv="p_values", lfc_thr=1, pv_thr=0.05, color=("green", "red"),
              valpha=1,
              geneid=None, genenames=None, gfont=8):
    general.depr_mes("bioinfokit.visuz.gene_exp.involcano")


def ma(table="dataset_file", lfc="logFC", ct_count="value1", st_count="value2", lfc_thr=1):
    general.depr_mes("bioinfokit.visuz.gene_exp.ma")


def corr_mat(table="p_df", corm="pearson"):
    general.depr_mes("bioinfokit.visuz.stat.corr_mat")


def screeplot(obj="pcascree"):
    y = [x * 100 for x in obj[1]]
    plt.bar(obj[0], y)
    plt.xlabel('PCs', fontsize=12, fontname="sans-serif")
    plt.ylabel('Proportion of variance (%)', fontsize=12, fontname="sans-serif")
    plt.xticks(fontsize=7, rotation=70)
    plt.savefig('screeplot.png', format='png', bbox_inches='tight', dpi=300)
    plt.close()


def pcaplot(x="x", y="y", z="z", labels="d_cols", var1="var1", var2="var2", var3="var3"):
    for i, varnames in enumerate(labels):
        plt.scatter(x[i], y[i])
        plt.text(x[i], y[i], varnames, fontsize=10)
    plt.xlabel("PC1 ({}%)".format(var1), fontsize=12, fontname="sans-serif")
    plt.ylabel("PC2 ({}%)".format(var2), fontsize=12, fontname="sans-serif")
    plt.tight_layout()
    plt.savefig('pcaplot_2d.png', format='png', bbox_inches='tight', dpi=300)
    plt.close()

    # for 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, varnames in enumerate(labels):
        ax.scatter(x[i], y[i], z[i])
        ax.text(x[i], y[i], z[i], varnames, fontsize=10)
    ax.set_xlabel("PC1 ({}%)".format(var1), fontsize=12, fontname="sans-serif")
    ax.set_ylabel("PC2 ({}%)".format(var2), fontsize=12, fontname="sans-serif")
    ax.set_zlabel("PC3 ({}%)".format(var3), fontsize=12, fontname="sans-serif")
    plt.tight_layout()
    plt.savefig('pcaplot_3d.png', format='png', bbox_inches='tight', dpi=300)
    plt.close()


def hmap(table="dataset_file", cmap="seismic", scale=True, dim=(4, 6), clus=True, zscore=None, xlabel=True, ylabel=True,
         tickfont=(10, 10)):
    general.depr_mes("bioinfokit.visuz.gene_exp.hmap")


def venn(vennset=(1, 1, 1, 1, 1, 1, 1), venncolor=('#00909e', '#f67280', '#ff971d'), vennalpha=0.5,
         vennlabel=('A', 'B', 'C')):
    fig = plt.figure()
    if len(vennset) == 7:
        venn3(subsets=vennset, set_labels=vennlabel, set_colors=venncolor, alpha=vennalpha)
        plt.savefig('venn3.png', format='png', bbox_inches='tight', dpi=300)
    elif len(vennset) == 3:
        venn2(subsets=vennset, set_labels=vennlabel, set_colors=venncolor, alpha=vennalpha)
        plt.savefig('venn2.png', format='png', bbox_inches='tight', dpi=300)
    else:
        print("Error: check the set dataset")


class gene_exp:

    def __init__(self):
        pass

    def geneplot(d, geneid, lfc, lfc_thr, pv_thr, genenames, gfont, pv, gstyle):

        if genenames is not None and genenames == "deg":
            for i in d[geneid].unique():
                if (d.loc[d[geneid] == i, lfc].iloc[0] >= lfc_thr and d.loc[d[geneid] == i, pv].iloc[0] < pv_thr) or \
                        (d.loc[d[geneid] == i, lfc].iloc[0] <= -lfc_thr and d.loc[d[geneid] == i, pv].iloc[0] < pv_thr):
                    if gstyle == 1:
                        plt.text(d.loc[d[geneid] == i, lfc].iloc[0], d.loc[d[geneid] == i, 'logpv_add_axy'].iloc[0], i,
                                 fontsize=gfont)
                    elif gstyle == 2:
                        plt.annotate(i, xy=(
                            d.loc[d[geneid] == i, lfc].iloc[0], d.loc[d[geneid] == i, 'logpv_add_axy'].iloc[0]),
                                     xycoords='data', xytext=(5, -15), textcoords='offset points', size=gfont,
                                     bbox=dict(boxstyle="round", alpha=0.1),
                                     arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1, relpos=(0, 0)))
                    else:
                        print("Error: invalid gstyle choice")
                        sys.exit(1)
        elif genenames is not None and type(genenames) is tuple:
            for i in d[geneid].unique():
                if i in genenames:
                    if gstyle == 1:
                        plt.text(d.loc[d[geneid] == i, lfc].iloc[0], d.loc[d[geneid] == i, 'logpv_add_axy'].iloc[0], i,
                                 fontsize=gfont)
                    elif gstyle == 2:
                        plt.annotate(i, xy=(
                            d.loc[d[geneid] == i, lfc].iloc[0], d.loc[d[geneid] == i, 'logpv_add_axy'].iloc[0]),
                                     xycoords='data', xytext=(5, -15), textcoords='offset points', size=gfont,
                                     bbox=dict(boxstyle="round", alpha=0.1),
                                     arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1, relpos=(0, 0)))
                    else:
                        print("Error: invalid gstyle choice")
                        sys.exit(1)
        elif genenames is not None and type(genenames) is dict:
            for i in d[geneid].unique():
                if i in genenames:
                    if gstyle == 1:
                        plt.text(d.loc[d[geneid] == i, lfc].iloc[0], d.loc[d[geneid] == i, 'logpv_add_axy'].iloc[0],
                                 genenames[i], fontsize=gfont)
                    elif gstyle == 2:
                        plt.annotate(genenames[i], xy=(
                            d.loc[d[geneid] == i, lfc].iloc[0], d.loc[d[geneid] == i, 'logpv_add_axy'].iloc[0]),
                                     xycoords='data', xytext=(5 + 10, -15 + 25), textcoords='offset points', size=gfont,
                                     bbox=dict(boxstyle="round", alpha=0.1 + 0.05),
                                     arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1 + 0.5,  # arrow
                                                     relpos=(0, 0)))
                    else:
                        print("Error: invalid gstyle choice")
                        sys.exit(1)

    def volcano(df="dataframe", lfc=None, pv=None, lfc_thr=1, pv_thr=0.05, color=("green", "grey", "red"), valpha=1,
                geneid=None, genenames=None, gfont=8, dim=(5, 5), r=300, ar=90, dotsize=8, markerdot="o",
                sign_line=False, gstyle=1, show=False, figtype='png', axtickfontsize=9,
                axtickfontname="Arial", axlabelfontsize=9, axlabelfontname="Arial", axxlabel=None,
                axylabel=None, xlm=None, ylm=None, plotlegend=False, legendpos='best',
                figname='volcano', legendanchor=None,
                legendlabels=['significant up', 'not significant', 'significant down']):
        _x = r'$ log_{2}(Fold Change)$'
        _y = r'$ -log_{10}(P-value)$'
        color = color
        # check if dataframe contains any non-numeric character
        assert general.check_for_nonnumeric(df[lfc]) == 0, 'dataframe contains non-numeric values in lfc column'
        assert general.check_for_nonnumeric(df[pv]) == 0, 'dataframe contains non-numeric values in pv column'
        # this is important to check if color or logpv exists and drop them as if you run multiple times same command
        # it may update old instance of df
        df = df.drop(['color_add_axy', 'logpv_add_axy'], axis=1, errors='ignore')
        assert len(set(color)) == 3, 'unique color must be size of 3'
        df.loc[(df[lfc] >= lfc_thr) & (df[pv] < pv_thr), 'color_add_axy'] = color[0]  # upregulated
        df.loc[(df[lfc] <= -lfc_thr) & (df[pv] < pv_thr), 'color_add_axy'] = color[2]  # downregulated
        df['color_add_axy'].fillna(color[1], inplace=True)  # intermediate
        df['logpv_add_axy'] = -(np.log10(df[pv]))
        # print(df[df['color']==color[0]].count(), 'zzzz')
        # plot
        assign_values = {col: i for i, col in enumerate(color)}
        color_result_num = [assign_values[i] for i in df['color_add_axy']]
        assert len(set(
            color_result_num)) == 3, 'either significant or non-significant genes are missing; try to change lfc_thr or ' \
                                     'pv_thr to include  both significant and non-significant genes'
        plt.subplots(figsize=dim)
        if plotlegend:
            s = plt.scatter(df[lfc], df['logpv_add_axy'], c=color_result_num, cmap=ListedColormap(color), alpha=valpha,
                            s=dotsize,
                            marker=markerdot)
            assert len(legendlabels) == 3, 'legendlabels must be size of 3'
            plt.legend(handles=s.legend_elements()[0], labels=legendlabels, loc=legendpos,
                       bbox_to_anchor=legendanchor)
        else:
            plt.scatter(df[lfc], df['logpv_add_axy'], c=color_result_num, cmap=ListedColormap(color), alpha=valpha,
                        s=dotsize,
                        marker=markerdot)
        if sign_line:
            plt.axhline(y=-np.log10(pv_thr), linestyle='--', color='#7d7d7d', linewidth=1)
            plt.axvline(x=lfc_thr, linestyle='--', color='#7d7d7d', linewidth=1)
            plt.axvline(x=-lfc_thr, linestyle='--', color='#7d7d7d', linewidth=1)
        gene_exp.geneplot(df, geneid, lfc, lfc_thr, pv_thr, genenames, gfont, pv, gstyle)

        if axxlabel:
            _x = axxlabel
        if axylabel:
            _y = axylabel
        general.axis_labels(_x, _y, axlabelfontsize, axlabelfontname)
        general.axis_ticks(xlm, ylm, axtickfontsize, axtickfontname, ar)
        general.get_figure(show, r, figtype, figname)

    def involcano(df="dataframe", lfc="logFC", pv="p_values", lfc_thr=1, pv_thr=0.05, color=("green", "grey", "red"),
                  valpha=1, geneid=None, genenames=None, gfont=8, dim=(5, 5), r=300, ar=90, dotsize=8, markerdot="o",
                  sign_line=False, gstyle=1, show=False, figtype='png', axtickfontsize=9,
                  axtickfontname="Arial", axlabelfontsize=9, axlabelfontname="Arial", axxlabel=None,
                  axylabel=None, xlm=None, ylm=None, plotlegend=False, legendpos='best',
                  figname='involcano', legendanchor=None,
                  legendlabels=['significant up', 'not significant', 'significant down']):
        _x = r'$ log_{2}(Fold Change)$'
        _y = r'$ -log_{10}(P-value)$'
        color = color
        assert general.check_for_nonnumeric(df[lfc]) == 0, 'dataframe contains non-numeric values in lfc column'
        assert general.check_for_nonnumeric(df[pv]) == 0, 'dataframe contains non-numeric values in pv column'
        # this is important to check if color or logpv exists and drop them as if you run multiple times same command
        # it may update old instance of df
        df = df.drop(['color_add_axy', 'logpv_add_axy'], axis=1, errors='ignore')
        assert len(set(color)) == 3, 'unique color must be size of 3'
        df.loc[(df[lfc] >= lfc_thr) & (df[pv] < pv_thr), 'color_add_axy'] = color[0]  # upregulated
        df.loc[(df[lfc] <= -lfc_thr) & (df[pv] < pv_thr), 'color_add_axy'] = color[2]  # downregulated
        df['color_add_axy'].fillna(color[1], inplace=True)  # intermediate
        df['logpv_add_axy'] = -(np.log10(df[pv]))

        # plot
        assign_values = {col: i for i, col in enumerate(color)}
        color_result_num = [assign_values[i] for i in df['color_add_axy']]
        assert len(set(
            color_result_num)) == 3, 'either significant or non-significant genes are missing; try to change lfc_thr or ' \
                                     'pv_thr to include  both significant and non-significant genes'
        plt.subplots(figsize=dim)
        if plotlegend:
            s = plt.scatter(df[lfc], df['logpv_add_axy'], c=color_result_num, cmap=ListedColormap(color), alpha=valpha,
                            s=dotsize, marker=markerdot)
            assert len(legendlabels) == 3, 'legendlabels must be size of 3'
            plt.legend(handles=s.legend_elements()[0], labels=legendlabels, loc=legendpos,
                       bbox_to_anchor=legendanchor)
        else:
            plt.scatter(df[lfc], df['logpv_add_axy'], c=color_result_num, cmap=ListedColormap(color), alpha=valpha,
                        s=dotsize, marker=markerdot)
        gene_exp.geneplot(df, geneid, lfc, lfc_thr, pv_thr, genenames, gfont, pv, gstyle)
        plt.gca().invert_yaxis()
        if axxlabel:
            _x = axxlabel
        if axylabel:
            _y = axylabel
        general.axis_labels(_x, _y, axlabelfontsize, axlabelfontname)
        if xlm:
            print('Error: xlm not compatible with involcano')
            sys.exit(1)
        if ylm:
            print('Error: ylm not compatible with involcano')
            sys.exit(1)
        general.axis_ticks(xlm, ylm, axtickfontsize, axtickfontname, ar)
        general.get_figure(show, r, figtype, figname)

    def ma(df="dataframe", lfc="logFC", ct_count="value1", st_count="value2", lfc_thr=1, valpha=1, dotsize=8,
           markerdot="o", dim=(6, 5), r=300, show=False, color=("green", "grey", "red"), ar=90, figtype='png',
           axtickfontsize=9,
           axtickfontname="Arial", axlabelfontsize=9, axlabelfontname="Arial", axxlabel=None,
           axylabel=None, xlm=None, ylm=None, fclines=False, fclinescolor='#2660a4', legendpos='best',
           figname='ma', legendanchor=None, legendlabels=['significant up', 'not significant', 'significant down'],
           plotlegend=False):
        _x, _y = 'A', 'M'
        assert general.check_for_nonnumeric(df[lfc]) == 0, 'dataframe contains non-numeric values in lfc column'
        assert general.check_for_nonnumeric(df[ct_count]) == 0, \
            'dataframe contains non-numeric values in ct_count column'
        assert general.check_for_nonnumeric(
            df[st_count]) == 0, 'dataframe contains non-numeric values in ct_count column'
        # this is important to check if color or A exists and drop them as if you run multiple times same command
        # it may update old instance of df
        df = df.drop(['color_add_axy', 'A_add_axy'], axis=1, errors='ignore')
        assert len(set(color)) == 3, 'unique color must be size of 3'
        df.loc[(df[lfc] >= lfc_thr), 'color_add_axy'] = color[0]  # upregulated
        df.loc[(df[lfc] <= -lfc_thr), 'color_add_axy'] = color[2]  # downregulated
        df['color_add_axy'].fillna(color[1], inplace=True)  # intermediate
        df['A_add_axy'] = (np.log2(df[ct_count]) + np.log2(df[st_count])) / 2
        # plot
        assign_values = {col: i for i, col in enumerate(color)}
        color_result_num = [assign_values[i] for i in df['color_add_axy']]
        assert len(
            set(
                color_result_num)) == 3, 'either significant or non-significant genes are missing; try to change lfc_thr' \
                                         ' to include both significant and non-significant genes'
        plt.subplots(figsize=dim)
        # plt.scatter(df['A'], df[lfc], c=df['color'], alpha=valpha, s=dotsize, marker=markerdot)
        if plotlegend:
            s = plt.scatter(df['A_add_axy'], df[lfc], c=color_result_num, cmap=ListedColormap(color),
                            alpha=valpha, s=dotsize, marker=markerdot)
            assert len(legendlabels) == 3, 'legendlabels must be size of 3'
            plt.legend(handles=s.legend_elements()[0], labels=legendlabels, loc=legendpos,
                       bbox_to_anchor=legendanchor)
        else:
            plt.scatter(df['A_add_axy'], df[lfc], c=color_result_num, cmap=ListedColormap(color),
                        alpha=valpha, s=dotsize, marker=markerdot)
        # draw a central line at M=0
        plt.axhline(y=0, color='#7d7d7d', linestyle='--')
        # draw lfc threshold lines
        if fclines:
            plt.axhline(y=lfc_thr, color=fclinescolor, linestyle='--')
            plt.axhline(y=-lfc_thr, color=fclinescolor, linestyle='--')
        if axxlabel:
            _x = axxlabel
        if axylabel:
            _y = axylabel
        general.axis_labels(_x, _y, axlabelfontsize, axlabelfontname)
        general.axis_ticks(xlm, ylm, axtickfontsize, axtickfontname, ar)
        general.get_figure(show, r, figtype, figname)

    def hmap(df="dataframe", cmap="seismic", scale=True, dim=(4, 6), rowclus=True, colclus=True, zscore=None,
             xlabel=True,
             ylabel=True, tickfont=(10, 10), r=300, show=False, figtype='png', figname='heatmap'):
        # df = df.set_index(d.columns[0])
        # plot heatmap without cluster
        # more cmap: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        # dim = dim
        fig, hm = plt.subplots(figsize=dim)
        if rowclus and colclus:
            hm = sns.clustermap(df, cmap=cmap, cbar=scale, z_score=zscore, xticklabels=xlabel, yticklabels=ylabel,
                                figsize=dim)
            hm.ax_heatmap.set_xticklabels(hm.ax_heatmap.get_xmajorticklabels(), fontsize=tickfont[0])
            hm.ax_heatmap.set_yticklabels(hm.ax_heatmap.get_ymajorticklabels(), fontsize=tickfont[1])
            general.get_figure(show, r, figtype, figname)
        elif rowclus and colclus is False:
            hm = sns.clustermap(df, cmap=cmap, cbar=scale, z_score=zscore, xticklabels=xlabel, yticklabels=ylabel,
                                figsize=dim, row_cluster=True, col_cluster=False)
            hm.ax_heatmap.set_xticklabels(hm.ax_heatmap.get_xmajorticklabels(), fontsize=tickfont[0])
            hm.ax_heatmap.set_yticklabels(hm.ax_heatmap.get_ymajorticklabels(), fontsize=tickfont[1])
            general.get_figure(show, r, figtype, figname)
        elif colclus and rowclus is False:
            hm = sns.clustermap(df, cmap=cmap, cbar=scale, z_score=zscore, xticklabels=xlabel, yticklabels=ylabel,
                                figsize=dim, row_cluster=False, col_cluster=True)
            hm.ax_heatmap.set_xticklabels(hm.ax_heatmap.get_xmajorticklabels(), fontsize=tickfont[0])
            hm.ax_heatmap.set_yticklabels(hm.ax_heatmap.get_ymajorticklabels(), fontsize=tickfont[1])
            general.get_figure(show, r, figtype, figname)
        else:
            hm = sns.heatmap(df, cmap=cmap, cbar=scale, xticklabels=xlabel, yticklabels=ylabel)
            plt.xticks(fontsize=tickfont[0])
            plt.yticks(fontsize=tickfont[1])
            general.get_figure(show, r, figtype, figname)


class general:
    def __init__(self):
        pass

    rand_colors = ('#a7414a', '#282726', '#6a8a82', '#a37c27', '#563838', '#0584f2', '#f28a30', '#f05837',
                   '#6465a5', '#00743f', '#be9063', '#de8cf0', '#888c46', '#c0334d', '#270101', '#8d2f23',
                   '#ee6c81', '#65734b', '#14325c', '#704307', '#b5b3be', '#f67280', '#ffd082', '#ffd800',
                   '#ad62aa', '#21bf73', '#a0855b', '#5edfff', '#08ffc8', '#ca3e47', '#c9753d', '#6c5ce7')

    def get_figure(show, r, figtype, fig_name):
        if show:
            plt.show()
        else:
            plt.savefig(fig_name + '.' + figtype, format=figtype, bbox_inches='tight', dpi=r)
        plt.close()

    def axis_labels(x, y, axlabelfontsize=None, axlabelfontname=None):
        plt.xlabel(x, fontsize=axlabelfontsize, fontname=axlabelfontname)
        plt.ylabel(y, fontsize=axlabelfontsize, fontname=axlabelfontname)
        # plt.xticks(fontsize=9, fontname="sans-serif")
        # plt.yticks(fontsize=9, fontname="sans-serif")

    def axis_ticks(xlm=None, ylm=None, axtickfontsize=None, axtickfontname=None, ar=None):
        if xlm:
            plt.xlim(left=xlm[0], right=xlm[1])
            plt.xticks(np.arange(xlm[0], xlm[1], xlm[2]), fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)
        else:
            plt.xticks(fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)

        if ylm:
            plt.ylim(bottom=ylm[0], top=ylm[1])
            plt.yticks(np.arange(ylm[0], ylm[1], ylm[2]), fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)
        else:
            plt.yticks(fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)

    def depr_mes(func_name):
        print("This function is deprecated. Please use", func_name)
        print("Read docs at https://reneshbedre.github.io/blog/howtoinstall.html")

    def check_for_nonnumeric(pd_series=None):
        if pd.to_numeric(pd_series, errors='coerce').isna().sum() == 0:
            return 0
        else:
            return 1


class marker:

    def __init__(self):
        pass

    def geneplot_mhat(df, markeridcol, chr, pv, gwasp, markernames, gfont, gstyle, ax):
        if markeridcol is not None:
            if markernames is not None and markernames is True:
                for i in df[markeridcol].unique():
                    if df.loc[df[markeridcol] == i, pv].iloc[0] <= gwasp:
                        if gstyle == 1:
                            plt.text(df.loc[df[markeridcol] == i, 'ind'].iloc[0],
                                     df.loc[df[markeridcol] == i, 'tpval'].iloc[0],
                                     str(i), fontsize=gfont)
                        elif gstyle == 2:
                            plt.annotate(i, xy=(
                                df.loc[df[markeridcol] == i, 'ind'].iloc[0],
                                df.loc[df[markeridcol] == i, 'tpval'].iloc[0]),
                                         xycoords='data', xytext=(5, -15), textcoords='offset points', size=6,
                                         bbox=dict(boxstyle="round", alpha=0.2),
                                         arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.2, relpos=(0, 0)))
            elif markernames is not None and isinstance(markernames, (tuple, list)):
                for i in df[markeridcol].unique():
                    if i in markernames:
                        if gstyle == 1:
                            plt.text(df.loc[df[markeridcol] == i, 'ind'].iloc[0],
                                     df.loc[df[markeridcol] == i, 'tpval'].iloc[0],
                                     str(i), fontsize=gfont)
                        elif gstyle == 2:
                            plt.annotate(i, xy=(
                                df.loc[df[markeridcol] == i, 'ind'].iloc[0],
                                df.loc[df[markeridcol] == i, 'tpval'].iloc[0]),
                                         xycoords='data', xytext=(5, -15), textcoords='offset points', size=6,
                                         bbox=dict(boxstyle="round", alpha=0.2),
                                         arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.2, relpos=(0, 0)))
            elif markernames is not None and isinstance(markernames, dict):
                for i in df[markeridcol].unique():
                    if i in markernames:
                        if gstyle == 1:
                            plt.text(df.loc[df[markeridcol] == i, 'ind'].iloc[0],
                                     df.loc[df[markeridcol] == i, 'tpval'].iloc[0],
                                     markernames[i], fontsize=gfont)
                        elif gstyle == 2:
                            plt.annotate(markernames[i], xy=(
                                df.loc[df[markeridcol] == i, 'ind'].iloc[0],
                                df.loc[df[markeridcol] == i, 'tpval'].iloc[0]),
                                         xycoords='data', xytext=(5, -15), textcoords='offset points', size=6,
                                         bbox=dict(boxstyle="round", alpha=0.2),
                                         arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.2, relpos=(0, 0)))
        else:
            raise Exception("provide 'markeridcol' parameter")

    def mhat(df="dataframe", chr=None, pv=None, color=None, dim=(6, 4), r=300, ar=90, gwas_sign_line=False,
             gwasp=5E-08, dotsize=8, markeridcol=None, markernames=None, gfont=8, valpha=1, show=False, figtype='png',
             axxlabel=None, axylabel=None, axlabelfontsize=9, axlabelfontname="Arial", axtickfontsize=9,
             axtickfontname="Arial", ylm=None, gstyle=1, figname='manhatten'):

        _x, _y = 'Chromosomes', r'$ -log_{10}(P)$'
        rand_colors = ('#a7414a', '#282726', '#6a8a82', '#a37c27', '#563838', '#0584f2', '#f28a30', '#f05837',
                       '#6465a5', '#00743f', '#be9063', '#de8cf0', '#888c46', '#c0334d', '#270101', '#8d2f23',
                       '#ee6c81', '#65734b', '#14325c', '#704307', '#b5b3be', '#f67280', '#ffd082', '#ffd800',
                       '#ad62aa', '#21bf73', '#a0855b', '#5edfff', '#08ffc8', '#ca3e47', '#c9753d', '#6c5ce7',
                       '#a997df', '#513b56', '#590925', '#007fff', '#bf1363', '#f39237', '#0a3200', '#8c271e')

        # minus log10 of P-value
        df['tpval'] = -np.log10(df[pv])
        # df = df.sort_values(chr)
        # if the column contains numeric strings
        df = df.loc[pd.to_numeric(df[chr], errors='coerce').sort_values().index]
        # add indices
        df['ind'] = range(len(df))
        df_group = df.groupby(chr)
        if color is not None and len(color) == 2:
            color_1 = int(df[chr].nunique() / 2) * [color[0]]
            color_2 = int(df[chr].nunique() / 2) * [color[1]]
            if df[chr].nunique() % 2 == 0:
                color_list = list(reduce(lambda x, y: x + y, zip(color_1, color_2)))
            elif df[chr].nunique() % 2 == 1:
                color_list = list(reduce(lambda x, y: x + y, zip(color_1, color_2)))
                color_list.append(color[0])
        elif color is not None and len(color) == df[chr].nunique():
            color_list = color
        elif color is None:
            # select colors randomly from the list based in number of chr
            color_list = sample(rand_colors, df[chr].nunique())
        else:
            print("Error: in color argument")
            sys.exit(1)

        xlabels = []
        xticks = []
        fig, ax = plt.subplots(figsize=dim)
        i = 0
        for label, df1 in df.groupby(chr):
            df1.plot(kind='scatter', x='ind', y='tpval', color=color_list[i], s=dotsize, alpha=valpha, ax=ax)
            df1_max_ind = df1['ind'].iloc[-1]
            df1_min_ind = df1['ind'].iloc[0]
            xlabels.append(label)
            xticks.append((df1_max_ind - (df1_max_ind - df1_min_ind) / 2))
            i += 1

        # add GWAS significant line
        if gwas_sign_line is True:
            ax.axhline(y=-np.log10(gwasp), linestyle='--', color='#7d7d7d', linewidth=1)
        if markernames is not None:
            marker.geneplot_mhat(df, markeridcol, chr, pv, gwasp, markernames, gfont, gstyle, ax=ax)
        ax.margins(x=0)
        ax.margins(y=0)
        ax.set_xticks(xticks)
        ax.set_ylim([0, max(df['tpval'] + 1)])
        if ylm:
            ylm = np.arange(ylm[0], ylm[1], ylm[2])
        else:
            ylm = np.arange(0, max(df['tpval'] + 1), 1)
        ax.set_yticks(ylm)
        ax.set_xticklabels(xlabels, rotation=ar)
        # ax.set_yticklabels(ylm, fontsize=axtickfontsize, fontname=axtickfontname, rotation=ar)
        if axxlabel:
            _x = axxlabel
        if axylabel:
            _y = axylabel
        ax.set_xlabel(_x, fontsize=axlabelfontsize, fontname=axlabelfontname)
        ax.set_ylabel(_y, fontsize=axlabelfontsize, fontname=axlabelfontname)
        general.get_figure(show, r, figtype, figname)


class stat:
    def __init__(self):
        pass

    def bardot(df="dataframe", dim=(6, 4), bw=0.4, colorbar="#f2aa4cff", colordot=["#101820ff"], hbsize=4, r=300, ar=0,
               dotsize=6, valphabar=1, valphadot=1, markerdot="o", errorbar=True, show=False, ylm=None,
               axtickfontsize=9,
               axtickfontname="Arial", axlabelfontsize=9, axlabelfontname="Arial", yerrlw=None, yerrcw=None,
               axxlabel=None,
               axylabel=None, figtype='png'):
        # set axis labels to None
        _x = None
        _y = None
        xbar = np.arange(len(df.columns.to_numpy()))
        color_list_bar = colorbar
        color_list_dot = colordot
        if len(color_list_dot) == 1:
            color_list_dot = colordot * len(df.columns.to_numpy())
        plt.subplots(figsize=dim)
        if errorbar:
            plt.bar(x=xbar, height=df.describe().loc['mean'], yerr=df.sem(), width=bw, color=color_list_bar,
                    capsize=hbsize,
                    zorder=0, alpha=valphabar, error_kw={'elinewidth': yerrlw, 'capthick': yerrcw})
        else:
            plt.bar(x=xbar, height=df.describe().loc['mean'], width=bw, color=color_list_bar,
                    capsize=hbsize,
                    zorder=0, alpha=valphabar)

        plt.xticks(xbar, df.columns.to_numpy(), fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)
        if axxlabel:
            _x = axxlabel
        if axylabel:
            _y = axylabel
        general.axis_labels(_x, _y, axlabelfontsize, axlabelfontname)
        # ylm must be tuple of start, end, interval
        if ylm:
            plt.ylim(bottom=ylm[0], top=ylm[1])
            plt.yticks(np.arange(ylm[0], ylm[1], ylm[2]), fontsize=axtickfontsize, fontname=axtickfontname)
        plt.yticks(fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)
        # add dots
        for cols in range(len(df.columns.to_numpy())):
            # get markers from here https://matplotlib.org/3.1.1/api/markers_api.html
            plt.scatter(x=np.linspace(xbar[cols] - bw / 2, xbar[cols] + bw / 2, int(df.describe().loc['count'][cols])),
                        y=df[df.columns[cols]].dropna(), s=dotsize, color=color_list_dot[cols], zorder=1,
                        alpha=valphadot,
                        marker=markerdot)
        general.get_figure(show, r, figtype, 'bardot')

    def regplot(df="dataframe", x=None, y=None, yhat=None, dim=(6, 4), colordot='#4a4e4d', colorline='#fe8a71', r=300,
                ar=0, dotsize=6, valphaline=1, valphadot=1, linewidth=1, markerdot="o", show=False, axtickfontsize=9,
                axtickfontname="Arial", axlabelfontsize=9, axlabelfontname="Arial", ylm=None, xlm=None, axxlabel=None,
                axylabel=None, figtype='png'):
        fig, ax = plt.subplots(figsize=dim)
        plt.scatter(df[x].to_numpy(), df[y].to_numpy(), color=colordot, s=dotsize, alpha=valphadot, marker=markerdot,
                    label='Observed data')
        plt.plot(df[x].to_numpy(), df[yhat].to_numpy(), color=colorline, linewidth=linewidth, alpha=valphaline,
                 label='Regression line')
        if axxlabel:
            x = axxlabel
        if axylabel:
            y = axylabel
        general.axis_labels(x, y, axlabelfontsize, axlabelfontname)
        general.axis_ticks(xlm, ylm, axtickfontsize, axtickfontname, ar)
        plt.legend(fontsize=9)
        general.get_figure(show, r, figtype, 'reg_plot')

    def reg_resid_plot(df="dataframe", yhat=None, resid=None, stdresid=None, dim=(6, 4), colordot='#4a4e4d',
                       colorline='#2ab7ca', r=300, ar=0, dotsize=6, valphaline=1, valphadot=1, linewidth=1,
                       markerdot="o", show=False, figtype='png'):
        fig, ax = plt.subplots(figsize=dim)
        if resid is not None:
            plt.scatter(df[yhat], df[resid], color=colordot, s=dotsize, alpha=valphadot, marker=markerdot)
            plt.axhline(y=0, color=colorline, linestyle='--', linewidth=linewidth, alpha=valphaline)
            plt.xlabel("Fitted")
            plt.ylabel("Residuals")
            general.get_figure(show, r, figtype, 'resid_plot')
        else:
            print("Error: Provide residual data")
        if stdresid is not None:
            plt.scatter(df[yhat], df[stdresid], color=colordot, s=dotsize, alpha=valphadot, marker=markerdot)
            plt.axhline(y=0, color=colorline, linestyle='--', linewidth=linewidth, alpha=valphaline)
            plt.xlabel("Fitted")
            plt.ylabel("Standardized Residuals")
            general.get_figure(show, r, figtype, 'std_resid_plot')
        else:
            print("Error: Provide standardized residual data")

    def corr_mat(df="dataframe", corm="pearson", cmap="seismic", r=300, show=False, dim=(6, 5), axtickfontname="Arial",
                 axtickfontsize=7, ar=90, figtype='png'):
        d_corr = df.corr(method=corm)
        plt.subplots(figsize=dim)
        plt.matshow(d_corr, vmin=-1, vmax=1, cmap=cmap)
        plt.colorbar()
        cols = list(df)
        ticks = list(range(0, len(list(df))))
        plt.xticks(ticks, cols, fontsize=axtickfontsize, fontname=axtickfontname, rotation=ar)
        plt.yticks(ticks, cols, fontsize=axtickfontsize, fontname=axtickfontname)
        general.get_figure(show, r, figtype, 'corr_mat')

    # for data with pre-calculated mean and SE
    def multi_bar(df="dataframe", dim=(5, 4), colbar=None, colerrorbar=None, bw=0.4, colorbar=None, xbarcol=None, r=300,
                  show=False,
                  axtickfontname="Arial", axtickfontsize=9, ar=90, figtype='png', figname='multi_bar', valphabar=1,
                  legendpos='best', errorbar=False, yerrlw=None, yerrcw=None, plotlegend=False, hbsize=4, ylm=None):
        xbar = np.arange(df.shape[0])
        xbar_temp = xbar
        fig, ax = plt.subplots(figsize=dim)
        assert len(colbar) >= 2, "number of bar should be atleast 2"
        assert len(colbar) == len(colorbar), "number of color should be equivalent to number of column bars"
        if colbar is not None and isinstance(colbar, (tuple, list)):
            for i in range(len(colbar)):
                if errorbar:
                    ax.bar(x=xbar_temp, height=df[colbar[i]], yerr=df[colerrorbar[i]], width=bw, color=colorbar[i],
                           alpha=valphabar, capsize=hbsize, label=colbar[i], error_kw={'elinewidth': yerrlw,
                                                                                       'capthick': yerrcw})
                    xbar_temp = xbar_temp + bw
                else:
                    ax.bar(x=xbar_temp, height=df[colbar[i]], width=bw, color=colorbar[i], alpha=valphabar,
                           label=colbar[i])
                    xbar_temp = xbar_temp + bw
        ax.set_xticks(xbar + ((bw * (len(colbar) - 1)) / (1 + (len(colbar) - 1))))
        ax.set_xticklabels(df[xbarcol], fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)
        # ylm must be tuple of start, end, interval
        if ylm:
            plt.ylim(bottom=ylm[0], top=ylm[1])
            plt.yticks(np.arange(ylm[0], ylm[1], ylm[2]), fontsize=axtickfontsize, fontname=axtickfontname)
        if plotlegend:
            plt.legend(loc=legendpos)
        general.get_figure(show, r, figtype, figname)

    # for data with replicates
    def singlebar(df="dataframe", dim=(6, 4), bw=0.4, colorbar="#f2aa4cff", hbsize=4, r=300, ar=0,
                  valphabar=1, errorbar=True, show=False, ylm=None, axtickfontsize=9, axtickfontname="Arial",
                  axlabelfontsize=9, axlabelfontname="Arial", yerrlw=None, yerrcw=None, axxlabel=None,
                  axylabel=None, figtype='png'):
        # set axis labels to None
        _x = None
        _y = None
        xbar = np.arange(len(df.columns.to_numpy()))
        color_list_bar = colorbar
        plt.subplots(figsize=dim)
        if errorbar:
            plt.bar(x=xbar, height=df.describe().loc['mean'], yerr=df.sem(), width=bw, color=color_list_bar,
                    capsize=hbsize, alpha=valphabar, error_kw={'elinewidth': yerrlw, 'capthick': yerrcw})
        else:
            plt.bar(x=xbar, height=df.describe().loc['mean'], width=bw, color=color_list_bar,
                    capsize=hbsize, alpha=valphabar)

        plt.xticks(xbar, df.columns.to_numpy(), fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)
        if axxlabel:
            _x = axxlabel
        if axylabel:
            _y = axylabel
        general.axis_labels(_x, _y, axlabelfontsize, axlabelfontname)
        # ylm must be tuple of start, end, interval
        if ylm:
            plt.ylim(bottom=ylm[0], top=ylm[1])
            plt.yticks(np.arange(ylm[0], ylm[1], ylm[2]), fontsize=axtickfontsize, fontname=axtickfontname)
        plt.yticks(fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)
        general.get_figure(show, r, figtype, 'singlebar')


class cluster:
    def __init__(self):
        pass

    def screeplot(obj="pcascree", axlabelfontsize=9, axlabelfontname="Arial", axxlabel=None,
                  axylabel=None, figtype='png', r=300, show=False):
        y = [x * 100 for x in obj[1]]
        plt.bar(obj[0], y)
        xlab = 'PCs'
        ylab = 'Proportion of variance (%)'
        if axxlabel:
            xlab = axxlabel
        if axylabel:
            ylab = axylabel
        plt.xticks(fontsize=7, rotation=70)
        general.axis_labels(xlab, ylab, axlabelfontsize, axlabelfontname)
        general.get_figure(show, r, figtype, 'screeplot')

    def pcaplot(x=None, y=None, z=None, labels=None, var1=None, var2=None, var3=None, axlabelfontsize=9,
                axlabelfontname="Arial", figtype='png', r=300, show=False, plotlabels=True):
        if x is not None and y is not None and z is None:
            assert var1 is not None and var2 is not None and labels is not None, "var1 or var2 variable or labels are missing"
            for i, varnames in enumerate(labels):
                plt.scatter(x[i], y[i])
                if plotlabels:
                    plt.text(x[i], y[i], varnames, fontsize=10)
            general.axis_labels("PC1 ({}%)".format(var1), "PC2 ({}%)".format(var2), axlabelfontsize, axlabelfontname)
            general.get_figure(show, r, figtype, 'pcaplot_2d')
        elif x is not None and y is not None and z is not None:
            assert var1 and var2 and var3 and labels is not None, "var1 or var2 or var3 or labels are missing"
            # for 3d plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i, varnames in enumerate(labels):
                ax.scatter(x[i], y[i], z[i])
                if plotlabels:
                    ax.text(x[i], y[i], z[i], varnames, fontsize=10)
            ax.set_xlabel("PC1 ({}%)".format(var1), fontsize=axlabelfontsize, fontname=axlabelfontname)
            ax.set_ylabel("PC2 ({}%)".format(var2), fontsize=axlabelfontsize, fontname=axlabelfontname)
            ax.set_zlabel("PC3 ({}%)".format(var3), fontsize=axlabelfontsize, fontname=axlabelfontname)
            general.get_figure(show, r, figtype, 'pcaplot_3d')

    # adapted from https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot
    def biplot(cscore=None, loadings=None, labels=None, var1=None, var2=None, var3=None, axlabelfontsize=9,
               axlabelfontname="Arial",
               figtype='png', r=300, show=False, markerdot="o", dotsize=6, valphadot=1, colordot='#4a4e4d',
               arrowcolor='#fe8a71',
               valphaarrow=1, arrowlinestyle='-', arrowlinewidth=1.0, centerlines=True, colorlist=None,
               legendpos='best',
               datapoints=True):
        assert cscore is not None and loadings is not None and labels is not None and var1 is not None and var2 is not None, \
            "cscore or loadings or labels or var1 or var2 are missing"
        if var1 is not None and var2 is not None and var3 is None:
            xscale = 1.0 / (cscore[:, 0].max() - cscore[:, 0].min())
            yscale = 1.0 / (cscore[:, 1].max() - cscore[:, 1].min())
            # zscale = 1.0 / (cscore[:, 2].max() - cscore[:, 2].min())
            # colorlist is an array of classes from dataframe column
            if datapoints:
                if colorlist is not None:
                    unique_class = set(colorlist)
                    # color_dict = dict()
                    assign_values = {col: i for i, col in enumerate(unique_class)}
                    color_result_num = [assign_values[i] for i in colorlist]
                    if colordot and isinstance(colordot, (tuple, list)):
                        colour_map = ListedColormap(colordot)
                        # for i in range(len(list(unique_class))):
                        #    color_dict[list(unique_class)[i]] = colordot[i]
                        # color_result = [color_dict[i] for i in colorlist]
                        s = plt.scatter(cscore[:, 0] * xscale, cscore[:, 1] * yscale, c=color_result_num,
                                        cmap=colour_map,
                                        s=dotsize, alpha=valphadot, marker=markerdot)
                        plt.legend(handles=s.legend_elements()[0], labels=list(unique_class), loc=legendpos)
                    elif colordot and not isinstance(colordot, (tuple, list)):
                        # s = plt.scatter(cscore[:, 0] * xscale, cscore[:, 1] * yscale, color=color_result, s=dotsize,
                        #                alpha=valphadot, marker=markerdot)
                        # plt.legend(handles=s.legend_elements()[0], labels=list(unique_class))
                        s = plt.scatter(cscore[:, 0] * xscale, cscore[:, 1] * yscale, c=color_result, s=dotsize,
                                        alpha=valphadot, marker=markerdot)
                        plt.legend(handles=s.legend_elements()[0], labels=list(unique_class), loc=legendpos)
                else:
                    plt.scatter(cscore[:, 0] * xscale, cscore[:, 1] * yscale, color=colordot, s=dotsize,
                                alpha=valphadot, marker=markerdot)
            if centerlines:
                plt.axhline(y=0, linestyle='--', color='#7d7d7d', linewidth=1)
                plt.axvline(x=0, linestyle='--', color='#7d7d7d', linewidth=1)
            for i in range(len(loadings)):
                plt.arrow(0, 0, loadings[0][i], loadings[1][i], color=arrowcolor, alpha=valphaarrow, ls=arrowlinestyle,
                          lw=arrowlinewidth)
                plt.text(loadings[0][i], loadings[1][i], labels[i])
                # adjust_text(t)
            # plt.xlim(min(loadings[0]) - 0.1, max(loadings[0]) + 0.1)
            # plt.ylim(min(loadings[1]) - 0.1, max(loadings[1]) + 0.1)
            xlimit_max = np.max([np.max(cscore[:, 0] * xscale), np.max(loadings[0])])
            xlimit_min = np.min([np.min(cscore[:, 0] * xscale), np.min(loadings[0])])
            ylimit_max = np.max([np.max(cscore[:, 1] * yscale), np.max(loadings[1])])
            ylimit_min = np.min([np.min(cscore[:, 1] * xscale), np.min(loadings[1])])
            plt.xlim(xlimit_min - 0.2, xlimit_max + 0.2)
            plt.ylim(ylimit_min - 0.2, ylimit_max + 0.2)
            general.axis_labels("PC1 ({}%)".format(var1), "PC2 ({}%)".format(var2), axlabelfontsize, axlabelfontname)
            general.get_figure(show, r, figtype, 'biplot_2d')
        # 3D
        if var1 is not None and var2 is not None and var3 is not None:
            xscale = 1.0 / (cscore[:, 0].max() - cscore[:, 0].min())
            yscale = 1.0 / (cscore[:, 1].max() - cscore[:, 1].min())
            zscale = 1.0 / (cscore[:, 2].max() - cscore[:, 2].min())
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if datapoints:
                if colorlist is not None:
                    unique_class = set(colorlist)
                    assign_values = {col: i for i, col in enumerate(unique_class)}
                    color_result_num = [assign_values[i] for i in colorlist]
                    if colordot and isinstance(colordot, (tuple, list)):
                        colour_map = ListedColormap(colordot)
                        s = ax.scatter(cscore[:, 0] * xscale, cscore[:, 1] * yscale, cscore[:, 2] * zscale,
                                       c=color_result_num,
                                       cmap=colour_map, s=dotsize, alpha=valphadot, marker=markerdot)
                        plt.legend(handles=s.legend_elements()[0], labels=list(unique_class), loc=legendpos)
                    elif colordot and not isinstance(colordot, (tuple, list)):
                        s = plt.scatter(cscore[:, 0] * xscale, cscore[:, 1] * yscale, cscore[:, 2] * zscale,
                                        c=color_result_num,
                                        s=dotsize, alpha=valphadot, marker=markerdot)
                        plt.legend(handles=s.legend_elements()[0], labels=list(unique_class), loc=legendpos)
                else:
                    ax.scatter(cscore[:, 0] * xscale, cscore[:, 1] * yscale, cscore[:, 2] * zscale, color=colordot,
                               s=dotsize, alpha=valphadot, marker=markerdot)
            for i in range(len(loadings)):
                ax.quiver(0, 0, 0, loadings[0][i], loadings[1][i], loadings[2][i], color=arrowcolor, alpha=valphaarrow,
                          ls=arrowlinestyle, lw=arrowlinewidth)
                ax.text(loadings[0][i], loadings[1][i], loadings[2][i], labels[i])

            xlimit_max = np.max([np.max(cscore[:, 0] * xscale), np.max(loadings[0])])
            xlimit_min = np.min([np.min(cscore[:, 0] * xscale), np.min(loadings[0])])
            ylimit_max = np.max([np.max(cscore[:, 1] * yscale), np.max(loadings[1])])
            ylimit_min = np.min([np.min(cscore[:, 1] * xscale), np.min(loadings[1])])
            zlimit_max = np.max([np.max(cscore[:, 2] * zscale), np.max(loadings[2])])
            zlimit_min = np.min([np.min(cscore[:, 2] * zscale), np.min(loadings[2])])
            # ax.set_xlim(min(loadings[0])-0.1, max(loadings[0])+0.1)
            # ax.set_ylim(min(loadings[1])-0.1, max(loadings[1])+0.1)
            # ax.set_zlim(min(loadings[2])-0.1, max(loadings[2])+0.1)
            ax.set_xlim(xlimit_min - 0.2, xlimit_max + 0.2)
            ax.set_ylim(ylimit_min - 0.2, ylimit_max + 0.2)
            ax.set_zlim(zlimit_min - 0.2, zlimit_max + 0.2)
            ax.set_xlabel("PC1 ({}%)".format(var1), fontsize=axlabelfontsize, fontname=axlabelfontname)
            ax.set_ylabel("PC2 ({}%)".format(var2), fontsize=axlabelfontsize, fontname=axlabelfontname)
            ax.set_zlabel("PC3 ({}%)".format(var3), fontsize=axlabelfontsize, fontname=axlabelfontname)
            general.get_figure(show, r, figtype, 'biplot_3d')

    def tsneplot(score=None, axlabelfontsize=9, axlabelfontname="Arial", figtype='png', r=300, show=False,
                 markerdot="o", dotsize=6, valphadot=1, colordot='#4a4e4d', colorlist=None, legendpos='best',
                 figname='tsne_2d', dim=(6, 4), legendanchor=None):
        assert score is not None, "score are missing"
        plt.subplots(figsize=dim)
        if colorlist is not None:
            unique_class = set(colorlist)
            # color_dict = dict()
            assign_values = {col: i for i, col in enumerate(unique_class)}
            color_result_num = [assign_values[i] for i in colorlist]
            if colordot and isinstance(colordot, (tuple, list)):
                colour_map = ListedColormap(colordot)
                s = plt.scatter(score[:, 0], score[:, 1], c=color_result_num, cmap=colour_map,
                                s=dotsize, alpha=valphadot, marker=markerdot)
                plt.legend(handles=s.legend_elements()[0], labels=list(unique_class), loc=legendpos,
                           bbox_to_anchor=legendanchor)
            elif colordot and not isinstance(colordot, (tuple, list)):
                s = plt.scatter(score[:, 0], score[:, 1], c=color_result_num,
                                s=dotsize, alpha=valphadot, marker=markerdot)
                plt.legend(handles=s.legend_elements()[0], labels=list(unique_class), loc=legendpos,
                           bbox_to_anchor=legendanchor)
        else:
            plt.scatter(score[:, 0], score[:, 1], color=colordot,
                        s=dotsize, alpha=valphadot, marker=markerdot)
        plt.xlabel("t-SNE-1", fontsize=axlabelfontsize, fontname=axlabelfontname)
        plt.ylabel("t-SNE-2", fontsize=axlabelfontsize, fontname=axlabelfontname)
        general.get_figure(show, r, figtype, figname)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

survey_year_code_to_direction_name = OrderedDict(
    {1: "1", 2: "2", 3: "3", 4: "4", 234: '234', 0: "all"}
)


def fix_coef_with_no_pvalue(row):
    has_coef_no_pvalue = False

    regs_with_no_pvalue = []

    for y in survey_year_code_to_direction_name.keys():
        if row[f"B{y}"] is None or np.isnan(row[f"B{y}"]):
            continue

        if row[f"P{y}"] is None or np.isnan(row[f"P{y}"]):
            regs_with_no_pvalue.append({f"B{y}": row[f"B{y}"], f"P{y}": row[f"P{y}"], f"N{y}": row[f"N{y}"]})
            row[f"B{y}"] = np.nan
            row[f"N{y}"] = np.nan
        pass

    row['reg_no_pvalue'] = regs_with_no_pvalue

    return row


def get_metric_table(vars_selected, metric, regs_no_null_coef_pval):
    series_dict = {}

    for series in survey_year_code_to_direction_name.keys():

        metric_df_cohorts = (
            regs_no_null_coef_pval[regs_no_null_coef_pval["series"] == series][
                ["var", "var_type", "series", "N", f"{metric} coef", f"{metric} p_val"]
            ].drop_duplicates()
                .groupby("var").agg({
                #     "series": lambda x: list(x),
                "var_type": lambda x: list(x),
                "N": lambda x: list(x),
                f"{metric} coef": lambda x: list(x),
                f"{metric} p_val": lambda x: list(x)
            }).reset_index().rename(columns={
                "var_type": "is categorical",
                f"{metric} coef": f"B{series}",
                f"{metric} p_val": f"P{series}",
                "N": f"N{series}"
            })
        )

        #         metric_df_cohorts["is categorical"] = metric_df_cohorts["is categorical"].apply(lambda x: 1 if x > 0 else 0)
        # metric_df_cohorts["num series has regression"] = metric_df_cohorts["has regression in series"].apply(lambda x: len(x))

        if metric_df_cohorts[f"N{series}"].apply(len).max() > 1:
            raise Exception("N")
        if metric_df_cohorts[f"B{series}"].apply(len).max() > 1:
            raise Exception("B")
        if metric_df_cohorts[f"P{series}"].apply(len).max() > 1:
            raise Exception("P")

        metric_df_cohorts[f"N{series}"] = metric_df_cohorts[f"N{series}"].apply(lambda x: list(x)[0])
        metric_df_cohorts[f"B{series}"] = metric_df_cohorts[f"B{series}"].apply(lambda x: list(x)[0])
        metric_df_cohorts[f"P{series}"] = metric_df_cohorts[f"P{series}"].apply(lambda x: list(x)[0])

        series_dict[series] = metric_df_cohorts
        pass

    m_df = pd.merge(
        vars_selected,
        series_dict[1],  # beign with NHANES 1999 (series ID is 1)
        on="var", how="left"
    )

    for series in list(survey_year_code_to_direction_name.keys())[1:]:
        m_df = pd.merge(
            m_df,
            series_dict[series],
            on="var", how="left"
        )

    def select_is_categorical_from_a_series_that_have_it(row):
        is_categorical_vals = []

        for col in row.index:
            if col.startswith('is categorical') is False:
                continue

            if str(row[col]) == 'nan':
                continue
            for is_categorical_val in row[col]:
                if (str(is_categorical_val) == 'nan'):
                    continue

                if isinstance(is_categorical_val, list):
                    for is_categorical in is_categorical_val:
                        is_categorical_vals.append(int(is_categorical))
                elif isinstance(is_categorical_val, int):
                    is_categorical_vals.append(is_categorical_val)
            #               is_categorical_vals.append((len(row[col]), row[col]))
            pass

        if len(is_categorical_vals) == 0:
            return None

        return max(is_categorical_vals)

    # m_df['is categorical'] = m_df.apply(select_is_categorical_from_a_series_that_have_it, axis=1)

    m_df.insert(5, 'is categorical',
                m_df.apply(select_is_categorical_from_a_series_that_have_it, axis=1)
                )

    m_df['num series present'] = m_df['present in series'].apply(lambda x: len(x))

    m_df = m_df.apply(fix_coef_with_no_pvalue, axis=1)

    m_df = m_df.drop(columns=[c for c in m_df.columns if c.startswith('is categorical_')])

    cols = list(m_df.columns)

    cols_B = [c for c in cols if c.startswith('B')]
    cols_P = [c for c in cols if c.startswith('P')]
    cols_N = [c for c in cols if c.startswith('N')]

    return m_df[
        [c for c in cols if c not in cols_B + cols_P + cols_N] + cols_B + cols_P + cols_N
        ].reset_index(drop=True).reset_index(drop=True)

    return m_df.reset_index(drop=True)

    return m_df[['var', 'present in series', 'module Patel', 'num series present', 'is categorical',
                 #           "is categorical_x", "is categorical_y",
                 'B1', 'B2', 'B3', 'B4', 'B0',
                 'P1', 'P2', 'P3', 'P4', 'P0',
                 'N1', 'N2', 'N3', 'N4', 'N0'
        , 'reg_no_pvalue'
                 ]].reset_index(drop=True)


# q = get_metric_table(vars_selected=all_vars_patel, metric='WFDPI').sort_values("module Patel")
# q


"""
Tests to ensure its working right
"""


def find_coef_with_no_pvalue(row):
    has_coef_no_pvalue = False

    for y in survey_year_code_to_direction_name.keys():
        if row[f"B{y}"] is None or np.isnan(row[f"B{y}"]):
            continue

        #         print(row[f"B{y}"], row[f"P{y}"],
        #               row[f"B{y}"] is None,
        #               np.isnan(row[f"B{y}"]),
        #               (row[f"B{y}"] is None) or (np.isnan(row[f"B{y}"]) is True)
        #              )
        if row[f"P{y}"] is None or np.isnan(row[f"P{y}"]):
            has_coef_no_pvalue = True
            break
        pass

    return has_coef_no_pvalue


def test_metric_dfs(metric_dfs, columns_reg_desc):
    # for metric in ['HEI-15', 'RW.WFDPI', 'WCDPI', 'WFDPI'][0:1]:# columns_reg_desc:
    for metric in columns_reg_desc:
        q = metric_dfs[metric]

        is_categorical_null = q[
            q["is categorical"].isnull()
            #   & q["num series present"].isnull()
        ]

        n0_null = q[q['N0'].isnull()]

        coef_with_no_pvalue = q[q.apply(find_coef_with_no_pvalue, axis=1)]

        print("{} 'is categorical' null: #{} 'N0' null: #{} coef_with_no_pvalue: #{}".format(
            metric,
            len(is_categorical_null),
            len(n0_null),
            len(coef_with_no_pvalue)
        ))

        if len(n0_null) > 0:
            break
            #         raise Exception("Bad....!!")
            pass

    # q[q.apply(find_coef_with_no_pvalue, axis=1)]
    return n0_null
# test_metric_dfs(metric_dfs)
