import os
import json
import math
from pyexpat import model
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm
from tqdm import tqdm
from scipy import stats
from goatools.obo_parser import GODag
from IPython.display import display

alpha = 0.05
min_nogs = 20

plt_style_file = "lib/plot.mplstyle"
plt.style.use(plt_style_file)
mpl.rc_file(plt_style_file)
# Scale the relevant rcParams by 2x
scale_factor = 2
for key in mpl.rcParams:
    try:
        if not "size" in key:
            continue
        if isinstance(mpl.rcParams[key], (int, float)) and not isinstance(
            mpl.rcParams[key], bool
        ):
            mpl.rcParams[key] *= scale_factor
        elif isinstance(mpl.rcParams[key], (list, tuple)):
            mpl.rcParams[key] = [
                (
                    v * scale_factor
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                    else v
                )
                for v in mpl.rcParams[key]
            ]
    except Exception as e:
        print(f"Error scaling {key} from {mpl.rcParams[key]}: {e}")
        raise e

# update rcparams
# legend box with white background and frame
mpl.rcParams["legend.facecolor"] = "white"
mpl.rcParams["legend.frameon"] = True
mpl.rcParams["legend.edgecolor"] = "black"


# read in the marker styles
with open("lib/plot_marker_styles.json", "r") as fh:
    marker_styles_dict = json.load(fh)["marker_styles_dict"]
    # this is a dictionary such that for each method it contains
    # a dict of marker styles for 'marker_pyplot', 'marker_plotly', 'marker_color', 'face_color', 'label'


def download_goslim_obo(url, data_dir):
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"{data_dir}/go-basic.obo", "wb") as file:
            file.write(response.content)
        print("GOslim OBO file downloaded successfully.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


def load_godag(data_dir):
    obo_filepath = f"{data_dir}/go-basic.obo"
    return GODag(obo_filepath)


def load_compiled_nogwise_data(compiled_res_dir):
    filepath = f"{compiled_res_dir}/compiled_transfers_across_methods.nogwise.tsv"
    df = pd.read_csv(filepath, sep="\t")
    col_rename_dict = {col: f"method:{col}" for col in df.columns if col != "nog_id"}
    df.rename(columns=col_rename_dict, inplace=True)
    return df


def load_nog_members(data_dir):
    filepath = f"{data_dir}/1236_nog_members.tsv"
    return pd.read_csv(
        filepath,
        sep="\t",
        usecols=[1, 2, 3],
        names=["nog_id", "#taxa", "#genes"],
        header=None,
    ).set_index("nog_id")


def add_nog_members_info(compiled_df, nog_members_df):
    compiled_df.loc[:, "#taxa"] = compiled_df["nog_id"].map(nog_members_df["#taxa"])
    compiled_df.loc[:, "#genes"] = compiled_df["nog_id"].map(nog_members_df["#genes"])
    compiled_df.dropna(subset=["#taxa"], inplace=True)
    return compiled_df


def load_fun_profiles(data_dir, unique_nogs_list):
    filepath = f"{data_dir}/1236.func_profiles.json"
    with open(filepath, "r") as f:
        return [
            json.loads(line) for line in f if json.loads(line)["n"] in unique_nogs_list
        ]


def create_nog_to_goslim_dict(fun_profiles_list):
    return {
        profile["n"]: [term["n"].strip() for term in profile["fprof"].get("GOslim", [])]
        for profile in fun_profiles_list
    }


def add_goslim_terms(compiled_df, nog_to_goslim_dict):
    compiled_df["GO_ID"] = compiled_df["nog_id"].map(nog_to_goslim_dict)
    return compiled_df[compiled_df["GO_ID"].notnull() & (compiled_df["GO_ID"] != "")]


def create_meta_category_to_terms_dict(go):
    keywords_dict = {
        "metabolism": ["metabolic"],
        # "operational": [
        #     "macromolecule biosynthetic process",
        #     "cell envelope",
        #     "cellular metabolic process",
        #     "phospholipid biosynthetic process",
        #     "nucleotide biosynthetic process",
        #     "regulatory",
        #     "regulation",
        # ],
        "operational": [
            "amino acid biosynthesis",
            "amino acid biosynthetic process",
            "cofactor biosynthetic process",
            "cell envelope",
            "intermediary metabolism",
            "fatty acid biosynthetic process",
            "phospholipid biosynthetic process",
            "nucleotide biosynthetic process",
            "regulatory",
            "regulation",
        ],
        "information_non_transcription": [
            "translation",
            "gene expression",
            "replication",
            "repair",
            "recombination",
            "translocation",
        ],
        "informational": [
            "translation",
            "transcription",
            "GTPase",
            "tRNA synthetase",
        ],
        "transcription": ["transcription"],
        "transport": ["transport"],
        "response": ["response", "defense", "detoxification"],
    }
    return {
        meta_category: {
            go_id: term
            for go_id, term in go.items()
            if any(keyword in term.name for keyword in keywords)
        }
        for meta_category, keywords in keywords_dict.items()
    }


def get_meta_category(go_ids_list, meta_category_to_terms_dict):
    meta_categories_list = [
        meta_category
        for go_id in go_ids_list
        for meta_category, terms in meta_category_to_terms_dict.items()
        if go_id in terms
    ]
    return meta_categories_list


def add_meta_category(compiled_df, meta_category_to_terms_dict):
    compiled_df["meta-category"] = compiled_df["GO_ID"].apply(
        lambda go_ids: get_meta_category(go_ids, meta_category_to_terms_dict)
    )
    # remove "informational" and "transcription" if "operational" is present
    compiled_df["meta-category"] = compiled_df["meta-category"].apply(
        lambda meta_categories: (
            [
                meta_category
                for meta_category in meta_categories
                if meta_category not in ["informational", "transcription"]
            ]
            if "operational" in meta_categories
            else meta_categories
        )
    )
    compiled_df["meta-category"] = compiled_df["meta-category"].apply(
        lambda meta_categories: ",".join(meta_categories)
    )
    return compiled_df


def add_go_term_names(compiled_df, go):
    compiled_df["GO term names"] = compiled_df["GO_ID"].apply(
        lambda go_ids: ",".join([go[go_id].name for go_id in go_ids if go_id in go])
    )
    return compiled_df


def compile_transfers_across_methods(url, data_dir, compiled_res_dir):
    download_goslim_obo(url, data_dir)
    go = load_godag(data_dir)
    compiled_df = load_compiled_nogwise_data(compiled_res_dir)
    nog_members_df = load_nog_members(data_dir)
    compiled_df = add_nog_members_info(compiled_df, nog_members_df)
    unique_nogs_list = compiled_df["nog_id"].unique().tolist()
    fun_profiles_list = load_fun_profiles(data_dir, unique_nogs_list)
    nog_to_goslim_dict = create_nog_to_goslim_dict(fun_profiles_list)
    compiled_df = add_goslim_terms(compiled_df, nog_to_goslim_dict)
    meta_category_to_terms_dict = create_meta_category_to_terms_dict(go)
    compiled_df = add_meta_category(compiled_df, meta_category_to_terms_dict)
    compiled_df = add_go_term_names(compiled_df, go)
    return compiled_df


def compile_nogwise_go_info(url, data_dir, compiled_res_dir):
    download_goslim_obo(url, data_dir)
    go = load_godag(data_dir)
    nog_members_df = load_nog_members(data_dir)
    # set the index of the nog_members_df to 'nog_id'
    nog_members_df.reset_index(inplace=True)
    unique_nogs_list = nog_members_df["nog_id"].unique().tolist()
    fun_profiles_list = load_fun_profiles(data_dir, unique_nogs_list)
    nog_to_goslim_dict = create_nog_to_goslim_dict(fun_profiles_list)
    compiled_df = add_goslim_terms(nog_members_df, nog_to_goslim_dict)
    meta_category_to_terms_dict = create_meta_category_to_terms_dict(go)
    compiled_df = add_meta_category(compiled_df, meta_category_to_terms_dict)
    compiled_df = add_go_term_names(compiled_df, go)
    return compiled_df


#############################################################################################################


def prepare_dataframe(df):
    # Copy the dataframe
    df_copy = df.copy()

    # Filter rows where meta-category has only a single value (no commas)
    df_single_meta_category = df_copy[df_copy["meta-category"].str.count(",") == 0]

    # Split the `meta-category` column to have one meta-category per row
    df_single_meta_category.loc[:, "meta-category"] = df_single_meta_category[
        "meta-category"
    ].str.split(",")
    df_exploded = df_single_meta_category.explode("meta-category").reset_index(
        drop=True
    )

    # Fill blank meta-category with "others"
    df_exploded.loc[df_exploded["meta-category"] == "", "meta-category"] = "others"

    # Drop the others category
    df_exploded = df_exploded[df_exploded["meta-category"] != "others"]

    return df_exploded


def plot_scatter_plots_hgt_nogwise_vs_col(df, plots_dir, x_column, y_label):
    # Set seaborn style
    sns.set_style("whitegrid")

    # Get method columns
    method_columns = [col for col in df.columns if col.startswith("method:")]

    # Plotting
    num_subplots = len(method_columns)
    num_cols = 4
    num_rows = math.ceil(num_subplots / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))

    for i, column in enumerate(method_columns):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]
        rho, p = stats.spearmanr(df[x_column], df[column])
        ax.scatter(
            df[x_column],
            df[column],
            label=f"Spearman rho: {rho:.2f}\np-value: {p:.2e}",
            s=50,
        )
        ax.set_title(column, fontsize=20)
        ax.set_xlabel(f"{x_column} in NOG")
        ax.set_ylabel(y_label)
        ax.legend(
            facecolor="white",
            edgecolor="black",
            prop={"size": 13},
        )

        # Plot best fit line
        x = df[x_column]
        y = df[column]
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m * x + b, "r-", linewidth=2)

    # Remove the empty subplots
    for i in range(num_subplots, num_rows * num_cols):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    plt.show()

    # Save pdf
    fig.savefig(
        f"{plots_dir}/transfers_vs_{x_column}_scatterplots.pdf",
        format="pdf",
        bbox_inches="tight",
    )


def calculate_spearman_nogwise_hgt_vs_col(df, x_column):
    method_columns = [col for col in df.columns if col.startswith("method:")]
    spearman_results = []

    for column in method_columns:
        rho, p = stats.spearmanr(df[x_column], df[column])
        spearman_results.append({"method": column, "spearman_rho": rho, "p_value": p})

    spearman_df = pd.DataFrame(spearman_results)
    # sort by rho
    spearman_df.sort_values("spearman_rho", ascending=True, inplace=True)
    return spearman_df


def prepare_model_data(df, meta_categories, main_variable_name):
    # Copy the dataframe
    df_copy = df.copy()
    # debug
    # display(df_copy)

    # If meta_categories is "all", then the main_variable_name is the only meta-category
    if meta_categories == "all":
        # note that the df is an exploded version of the original df.
        # so there are duplicates of nog_id in the df
        # for the df rows without the main_variable_name in the meta-category, drop the duplicate rows along the nog_id column

        # get the rows with the main_variable_name in the meta-category
        df_main_var = df_copy[
            df_copy["meta-category"].apply(lambda x: main_variable_name in x.split(","))
        ]
        # get the rows without the main_variable_name in the meta-category
        df_not_main_var = df_copy[
            df_copy["meta-category"].apply(
                lambda x: main_variable_name not in x.split(",")
            )
        ]
        # # drop the duplicates along the nog_id column
        # df_not_main_var = df_not_main_var.drop_duplicates(
        #     subset=["nog_id"], keep="first"
        # )
        # remove nog_id values in df_main_var from df_not_main_var
        df_not_main_var = df_not_main_var[
            ~df_not_main_var["nog_id"].isin(df_main_var["nog_id"])
        ]
        # concatenate the two dataframes. There are no more duplicates along the nog_id column
        df_copy = pd.concat([df_main_var, df_not_main_var])
        # get the unique meta-categories from the meta-category column to replace the "all" value
        meta_categories = df_copy["meta-category"].unique()

    # keep only the rows where the meta-category is in the meta_categories list
    df_copy = df_copy[df_copy["meta-category"].isin(meta_categories)].dropna()
    # # keep only one of the rows if there are multiple with the same nog_id
    # df_copy = df_copy.drop_duplicates(subset=["nog_id"], keep="first")
    # keep only the rows where the string of meta-category contains elements from the meta_categories list but not main_variable_name
    # df_copy = df_copy[df_copy["meta-category"].apply(lambda x: any(
    #     meta_category in x.split(",") for meta_category in meta_categories) and main_variable_name not in x.split(","))]
    df_copy[main_variable_name] = df_copy["meta-category"].apply(
        lambda x: 1 if main_variable_name in x.split(",") else 0
    )
    return df_copy


def fit_models_and_summarize(
    df, main_variable_name, correction_variable: str  # "#taxa" or "#genes"
):
    model_results_summary = {}

    df["mean_transfers"] = df["transfers"] / df["transfers"].mean()
    # formula = mean_transfers ~ main_variable_name + correction_variable
    formula = (
        f"Q('mean_transfers') ~ Q('{main_variable_name}') + Q('{correction_variable}')"
    )
    model_data = df[[main_variable_name, correction_variable, "mean_transfers"]]
    model_fit = sm.OLS.from_formula(formula, data=model_data).fit()
    # return coeff of main_variable_name, p-value of main_variable_name, coeff of correction_variable, p-value of correction_variable
    model_results_summary[f"method_{main_variable_name}_coeff"] = model_fit.params[
        f"Q('{main_variable_name}')"
    ]
    model_results_summary[f"p-value_{main_variable_name}"] = model_fit.pvalues[
        f"Q('{main_variable_name}')"
    ]
    model_results_summary[f"method_{correction_variable}_coeff"] = model_fit.params[
        f"Q('{correction_variable}')"
    ]
    model_results_summary[f"p-value_{correction_variable}"] = model_fit.pvalues[
        f"Q('{correction_variable}')"
    ]
    # display(model_results_summary)
    return model_results_summary


def read_nogwise_branchwise_data(compiled_res_dir):
    compiled_nogwise_branchwise_filepaths = [
        f"{compiled_res_dir}/{fi}"
        for fi in os.listdir(compiled_res_dir)
        if fi.startswith("compiled_transfers.nogwise.branchwise")
    ]
    compiled_nogwise_branchwise_filepaths_dict = {
        fi.partition("compiled_transfers.nogwise.branchwise.")[2].partition(".tsv")[
            0
        ]: fi
        for fi in compiled_nogwise_branchwise_filepaths
    }
    compiled_nogwise_branchwise_dfs = {
        key: pd.read_csv(value, sep="\t")
        for key, value in compiled_nogwise_branchwise_filepaths_dict.items()
    }
    # drop 'gloome_branch_name' column if it exists in any df
    for key, cnb_df in compiled_nogwise_branchwise_dfs.items():
        if "gloome_branch_name" in cnb_df.columns:
            cnb_df.drop(columns=["gloome_branch_name"], inplace=True)
        # sort dfs by 'transfers'
        compiled_nogwise_branchwise_dfs[key] = cnb_df.sort_values(
            by="transfers", ascending=False
        )
    return compiled_nogwise_branchwise_dfs


def add_function_info(
    compiled_nogwise_branchwise_dfs, compiled_transfers_across_methods_df
):
    cols_to_add = ["#taxa", "#genes", "GO_ID", "meta-category", "GO term names"]
    for key, df in compiled_nogwise_branchwise_dfs.items():
        for col in cols_to_add:
            df.loc[:, col] = df["nog_id"].map(
                compiled_transfers_across_methods_df.set_index("nog_id")[col]
            )
            # if any of the cols are empty or empty lists, drop the row
            df.dropna(subset=[col], inplace=True)
            if col == "GO_ID":
                df = df[df[col].apply(lambda x: len(x) > 0)]
        compiled_nogwise_branchwise_dfs[key] = df
    return compiled_nogwise_branchwise_dfs


# for each method, we model the HGT events inferred by the method
# as a function of the meta-category of the NOG categorically being x and not y (where y may be a specific meta-category or 'all' meta-categories except x)
# we do this for each meta-category, and for each method we plot the coefficients of the meta-category variable as we vary the transfer threshold
def compare_meta_categories_vs_acquisitions(
    compiled_nogwise_branchwise_dfs: dict,
    meta_categories,
    main_variable_name,
):
    """
    This function compares the meta-categories of the NOGs against the number of HGT events inferred by the method, across acquisitions, corrected for the NOG size.
    """
    method_model_results_summary = []
    for method, method_df in tqdm(compiled_nogwise_branchwise_dfs.items()):
        # first prepare the method_df for modelling
        method_df = prepare_model_data(method_df, meta_categories, main_variable_name)
        # display(method_df)

        # if the 'transfers' column has more than 1 unique value
        if method_df["transfers"].nunique() != 1:
            # acquisitions_list is a list of progressively increasing number of rows from the method_df
            # with the last element being the maximum number of rows in the method_df
            num_splits = 20
            acquisitions_list = np.linspace(0, method_df.shape[0], num=num_splits)
            acquisitions_list = [
                method_df.head(int(acquisitions)).copy()
                for acquisitions in acquisitions_list
                if acquisitions > 0
            ]
            # print(f"Method: {method}, acquisitions_list: {[acquisitions.shape[0] for acquisitions in acquisitions_list]}")

        else:  # if not, then we take the unique value
            # acquisitions_list = [method_df['transfers'].unique()[0]]
            acquisitions_list = [method_df]

        # now, for each point, we want to model the HGT events inferred by the method as a function of the meta-category of the NOG, corrected for the NOG size
        # and we store that in a df for this method in method_model_results_summary
        for acquisitions_df in acquisitions_list:
            acquisitions = acquisitions_df.shape[0]
            acquisitions_results_summary = fit_models_and_summarize(
                acquisitions_df, main_variable_name, "#genes"
            )
            acquisitions_results_summary["Number of HGTs"] = acquisitions_df.shape[0]
            # min "transfers" required to get that many acquisitions
            acquisitions_results_summary["transfer_threshold"] = acquisitions_df[
                "transfers"
            ].min()
            acquisitions_results_summary["method"] = method
            # store the results in a dict
            method_model_results_summary.append(acquisitions_results_summary)

    # create a df from the list of dicts
    method_model_results_summary_df = pd.DataFrame(method_model_results_summary)
    print(
        f"For model: Number of HGTs ~ ({main_variable_name} and not {[x for x in meta_categories if x != main_variable_name]}) + #genes, across acquisitions"
    )
    # keep only the rows where the p-value of the main_variable_name is less than alpha
    method_model_results_summary_df = method_model_results_summary_df[
        method_model_results_summary_df[f"p-value_{main_variable_name}"] < alpha
    ]

    display(method_model_results_summary_df)
    # display the mean of the coefficients of the main_variable_name for each method in a new df
    mean_df = method_model_results_summary_df.groupby("method")[
        f"method_{main_variable_name}_coeff"
    ].mean()
    # rename the column
    mean_df = mean_df.reset_index().rename(
        columns={
            f"method_{main_variable_name}_coeff": f"mean {main_variable_name} coefficient"
        }
    )
    # sort in descending order of the mean coefficient
    mean_df.sort_values(
        by=f"mean {main_variable_name} coefficient", ascending=False, inplace=True
    )
    display(mean_df)

    return method_model_results_summary_df


def compare_meta_categories_vs_transfer_thresholds(
    compiled_nogwise_branchwise_dfs: dict,
    meta_categories,
    main_variable_name,
):
    """
    This function compares the meta-categories of the NOGs against the number of HGT events inferred by the method, across transfer thresholds, corrected for the NOG size.

    """
    method_model_results_summary = []
    for method, method_df in tqdm(compiled_nogwise_branchwise_dfs.items()):
        # first prepare the method_df for modelling
        method_df = prepare_model_data(method_df, meta_categories, main_variable_name)
        # if the 'transfers' column has more than 1 unique value
        if method_df["transfers"].nunique() != 1:
            min_transfers = method_df["transfers"].min()
            max_transfers = method_df["transfers"].max()
            # we split the range of transfers into num_splits equal parts
            num_splits = 20
            transfers_list = np.linspace(min_transfers, max_transfers, num=num_splits)
        else:  # if not, then we take the unique value
            transfers_list = [method_df["transfers"].unique()[0]]
        # now, for each point, we want to model the HGT events inferred by the method as a function of the meta-category of the NOG, corrected for the NOG size
        # and we store that in a df for this method in method_model_results_summary
        for transfers in transfers_list:
            transfers_df = method_df[method_df["transfers"] <= transfers]
            if transfers_df.shape[0] < 10:
                continue
            # print(f"Method: {method}, transfers: {transfers}, Number of HGTs: {transfers_df.shape[0]}")
            # display(transfers_df)
            transfers_results_summary = fit_models_and_summarize(
                transfers_df, main_variable_name, "#genes"
            )
            transfers_results_summary["Number of HGTs"] = transfers_df.shape[0]
            transfers_results_summary["transfer_threshold"] = transfers
            transfers_results_summary["method"] = method
            # store the results in a dict
            method_model_results_summary.append(transfers_results_summary)

    # create a df from the list of dicts
    method_model_results_summary_df = pd.DataFrame(method_model_results_summary)
    print(
        f"For model: Number of HGTs ~ ({main_variable_name} and not {[x for x in meta_categories if x != main_variable_name]}) + #genes, across transfer thresholds"
    )
    display(method_model_results_summary_df)

    return method_model_results_summary_df


def find_non_hashable_elements(df, column):
    non_hashable_elements = []
    for i, row in df.iterrows():
        try:
            hash(row[column])
        except TypeError:
            non_hashable_elements.append((i, column, row[column]))
    return non_hashable_elements


def read_nogwise_data(compiled_res_dir):
    compiled_nogwise_filepaths = [
        f"{compiled_res_dir}/{fi}"
        for fi in os.listdir(compiled_res_dir)
        if fi.startswith("compiled_transfers.nogwise.") and "branchwise" not in fi
    ]
    compiled_nogwise_filepaths_dict = {
        fi.partition("compiled_transfers.nogwise.")[2].partition(".tsv")[0]: fi
        for fi in compiled_nogwise_filepaths
    }
    compiled_nogwise_dfs = {
        key: pd.read_csv(value, sep="\t")
        for key, value in compiled_nogwise_filepaths_dict.items()
    }
    # drop 'gloome_branch_name' column if it exists in any df
    for key, cn_df in compiled_nogwise_dfs.items():
        if "gloome_branch_name" in cn_df.columns:
            cn_df.drop(columns=["gloome_branch_name"], inplace=True)
        # sort dfs by 'transfers'
        try:
            compiled_nogwise_dfs[key] = cn_df.sort_values(by="transfer_threshold")
        except KeyError as e:
            raise KeyError(f"KeyError for {key}: {e}")
    return compiled_nogwise_dfs


def fit_transferability_increase_model(df, main_variable_name, correction_variable):
    """
    This function models the increase in transferability of the main_variable_name against the number of HGT events inferred by the method, across transfer thresholds.
    First we model the HGT events vector (y) as a linear function of the correction variable.
    For y' being the ODE regression of y, we define z = (y - y') / y'.
    z is the relative increase in HGT events corrected for the correction variable.
    We then model z as a linear function of the categorical main_variable_name.
    """

    model_results_summary = {}

    # formula1 = transfers ~ correction_variable
    formula1 = f"transfers ~ Q('{correction_variable}')"
    model_data1 = df[[correction_variable, "transfers"]]
    model_fit1 = sm.OLS.from_formula(formula1, data=model_data1).fit()
    # residuals of the model
    df["residuals"] = model_fit1.resid
    # relative increase in HGT events
    df["relative_hgt_increase"] = df["residuals"] / model_fit1.fittedvalues

    # formula2 = relative_hgt_increase ~ main_variable_name
    formula2 = f"Q('relative_hgt_increase') ~ Q('{main_variable_name}')"
    model_data2 = df[[main_variable_name, "relative_hgt_increase"]]
    model_fit2 = sm.OLS.from_formula(formula2, data=model_data2).fit()
    # return coeff of main_variable_name, p-value of main_variable_name
    model_results_summary[f"Increase in inferred transferability"] = model_fit2.params[
        f"Q('{main_variable_name}')"
    ]
    model_results_summary["p-value"] = model_fit2.pvalues[f"Q('{main_variable_name}')"]

    # also return the parameters and p-values of model1
    model_results_summary[f"Coefficient of {correction_variable}"] = model_fit1.params[
        f"Q('{correction_variable}')"
    ]
    model_results_summary[f"p-value of coefficient of {correction_variable}"] = (
        model_fit1.pvalues[f"Q('{correction_variable}')"]
    )

    # find the spearman rho and p-value of the transfers vs correction_variable
    rho, p = stats.spearmanr(df[correction_variable], df["transfers"])
    model_results_summary[f"rho {correction_variable} vs HGT"] = rho
    model_results_summary["p-value rho"] = p

    # how many of the main_variable_name in the column, are non-zero?
    num_main_var_name = df[df[main_variable_name] != 0].shape[0]
    model_results_summary[f"num {main_variable_name}"] = num_main_var_name

    return model_results_summary


def fit_transferability_model(
    df,
    main_variable_name,
    correction_variable,
):
    """
    This function models the increase in transferability of the main_variable_name against the number of HGT events inferred by the method, across transfer thresholds.
    First we model the HGT events vector (y) as a linear function of the correction variable.
    For y' being the ODE regression of y, we define r = (y - y') as the residuals of the model.
    We then model r as a linear function of the categorical main_variable_name.
    """

    model_results_summary = {}

    # find non-hashable elements in the 'main_variable_name' column
    non_hashable_elements = find_non_hashable_elements(df, main_variable_name)
    if non_hashable_elements:
        print(f"Non-hashable elements in the 'main_variable_name' column:")
        # display(non_hashable_elements)

    # formula1 = transfers ~ correction_variable
    formula1 = f"transfers ~ Q('{correction_variable}')"
    model_data1 = df[[correction_variable, "transfers"]]
    model_fit1 = sm.OLS.from_formula(formula1, data=model_data1).fit()
    # residuals of the model
    df["residuals"] = model_fit1.resid
    df["residuals"] = df["residuals"].astype(float)

    # formula2 = residuals ~ main_variable_name
    formula2 = f"residuals ~ Q('{main_variable_name}')"
    model_data2 = df[[main_variable_name, "residuals"]]
    model_fit2 = sm.OLS.from_formula(formula2, data=model_data2).fit()
    # return coeff of main_variable_name, p-value of main_variable_name
    model_results_summary[f"Increase in inferred transferability"] = model_fit2.params[
        f"Q('{main_variable_name}')"
    ]
    model_results_summary["p-value"] = model_fit2.pvalues[f"Q('{main_variable_name}')"]

    # also return the parameters and p-values of model1
    model_results_summary[f"Coefficient of {correction_variable}"] = model_fit1.params[
        f"Q('{correction_variable}')"
    ]
    model_results_summary[f"p-value of coefficient of {correction_variable}"] = (
        model_fit1.pvalues[f"Q('{correction_variable}')"]
    )

    # find the spearman rho and p-value of the transfers vs correction_variable
    rho, p = stats.spearmanr(df[correction_variable], df["transfers"])
    model_results_summary[f"rho {correction_variable} vs HGT"] = rho
    model_results_summary["p-value rho"] = p

    return model_results_summary


def compare_meta_category_transferability_vs_transfer_thresholds(
    compiled_nogwise_dfs: dict,
    main_variable_name,
    meta_categories,
    correction_variable,
    model_type="relative increase",
):
    """
    This function compares the increase in transferability of meta-categories against the number of NOGwise HGT events inferred by the method, across transfer thresholds.
    The transferability is the relative increase in HGT events inferred by the method, relative to the ODE regression of the HGT events on the correction variable which is the NOG size.

    Args:
    compiled_nogwise_dfs (dict): A dictionary of dataframes containing the compiled NOG-wise data for each method.
                                    Each file must contain the columns 'nog_id', 'transfer_threshold', 'transfers',
                                    where 'transfers' is the number of HGT events inferred by the method for the 'nog_id' at the 'transfer_threshold'.
    main_variable_name (str): The main variable name to compare against the number of HGT events.
    meta_categories (list): A list of meta-categories to compare against the number of HGT events, including the main_variable_name. This should be 'all' if all meta-categories are to be compared.
    correction_variable (str): The variable to correct the number of HGT events for. This should be either '#taxa' or '#genes'.

    Returns:
    method_model_results_summary_df (pd.DataFrame): A dataframe containing the results of the models for each method across transfer thresholds.
    """
    if meta_categories == "all":
        vs_categories = "others"
    else:
        vs_categories = [x for x in meta_categories if x != main_variable_name]

    method_model_results_summary = []
    for method, method_df in tqdm(compiled_nogwise_dfs.items()):
        # first prepare the method_df for modelling
        method_df = prepare_model_data(method_df, meta_categories, main_variable_name)
        # display(method_df)
        # check if main_variable_name is only a single value
        if method_df[main_variable_name].nunique() == 1:
            print(
                f"Skipping method: {method}, main_variable_name: {main_variable_name} is {method_df[main_variable_name].unique()[0]} for all NOGs"
            )
            print(f"List of NOGs: {method_df['nog_id'].unique()}")
            continue
        # if the 'transfer_threshold' column has more than 1 unique value
        if method_df["transfer_threshold"].nunique() != 1:
            min_transfer_threshold = method_df["transfer_threshold"].min()
            max_transfer_threshold = method_df["transfer_threshold"].max()

            # we split the range of transfer_thresholds into num_splits equal parts
            num_splits = 20
            transfer_thresholds_list = np.linspace(
                min_transfer_threshold, max_transfer_threshold, num=num_splits
            )
        else:  # if not, then we take the unique value
            transfer_thresholds_list = method_df["transfer_threshold"].unique().tolist()

        # print(f"Method: {method}, transfer_thresholds_list: {transfer_thresholds_list}")
        # now, for each point, we want to model the HGT events inferred by the method as a function of the meta-category of the NOG, corrected for the NOG size
        # and we store that in a df for this method in method_model_results_summary
        for transfer_threshold in transfer_thresholds_list:
            transfers_df = method_df[
                method_df["transfer_threshold"] >= transfer_threshold
            ]
            transfers_df = transfers_df.dropna(subset=[main_variable_name])
            # check that the df is not empty
            if transfers_df.empty or transfers_df.shape[0] < min_nogs:
                continue
            # # check that the df's main_variable_name column is not all zeros or all ones
            # if transfers_df[main_variable_name].nunique() == 1:
            #     print(
            #         f"Method: {method}, transfer_threshold: {transfer_threshold}, {main_variable_name} column is {transfers_df[main_variable_name].unique()[0]}"
            #     )
            #     continue
            # sort the df by 'transfer_threshold' in ascending order
            transfers_df = transfers_df.sort_values(
                by="transfer_threshold", ascending=True
            )
            # this df now contains nogwise HGT info for each NOG at thresholds greater than or equal to transfer_threshold. We want to keep the lowest transfer_thresholds per NOG
            transfers_df = transfers_df.groupby("nog_id").first().reset_index()

            # if transfer_threshold == transfer_thresholds_list[0]:
            #     print(
            #         f"Method: {method}, transfer_threshold: {transfer_threshold}, #NOGs: {transfers_df.shape[0]}"
            #     )
            #     display(transfers_df)
            if model_type == "relative increase":
                transfers_results_summary = fit_transferability_increase_model(
                    transfers_df, main_variable_name, correction_variable
                )
            elif model_type == "residual":
                transfers_results_summary = fit_transferability_model(
                    transfers_df, main_variable_name, correction_variable
                )
            else:
                raise ValueError(
                    f"model_type must be one of: 'relative increase', 'residual'"
                )
            transfers_results_summary["#NOGs"] = transfers_df.shape[0]
            transfers_results_summary["transfer_threshold"] = transfer_threshold
            transfers_results_summary["Number of HGTs"] = transfers_df["transfers"].sum()
            transfers_results_summary["method"] = method
            # store the results
            method_model_results_summary.append(transfers_results_summary)

    # if all methods were skipped, return None
    if not method_model_results_summary:
        return None

    # create a df from the list of dicts
    method_model_results_summary_df = pd.DataFrame(method_model_results_summary)
    # sort by method and then by transfer_threshold
    method_model_results_summary_df.sort_values(
        by=["method", "transfer_threshold"], inplace=True
    )

    # # # keep only the rows where the p-value of the main_variable_name is less than alpha
    # method_model_results_summary_df = method_model_results_summary_df[
    #     method_model_results_summary_df["p-value"] < alpha
    # ]

    return method_model_results_summary_df


def plot_model_summary_pyplot(
    method_model_results_summary,
    main_variable_name,
    vs_variable_name,
    x_variable,
    y_variable="Increase in inferred transferability",
    figsave_filepath=None,
    legend=False,
):
    # single figure, multiple line plots, one for each method
    print(
        f"Plotting {main_variable_name} vs {vs_variable_name} genes across {x_variable}"
    )
    fig, ax = plt.subplots(figsize=(16, 10))
    for method, method_df in method_model_results_summary.groupby("method"):
        ax.plot(
            method_df[x_variable],
            method_df[y_variable],
            label=marker_styles_dict[method]["label"],
            marker=marker_styles_dict[method]["marker_pyplot"],
            color=marker_styles_dict[method]["marker_color"],
            markerfacecolor=marker_styles_dict[method]["face_color"],
            markersize=10,
            linestyle="",
            linewidth=2,
        )

    ax.set_xlabel(x_variable)
    ax.set_ylabel(y_variable)
    ax.set_xscale("log")

    # increase xlim such that last point includes 2x the next power of 10
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], 2 * 10 ** math.ceil(math.log10(xlim[1])))

    # set title
    # ax.set_title(
    #     f"{main_variable_name.capitalize()} vs {vs_variable_name.capitalize()} genes"
    # )
    if legend:
        ax.legend(
        # fontsize=12
        )
    # make sure there's only unique values in legend
    # ax.get_legend().remove()
    # ax.legend(
    #     # fontsize=12
    # )

    # add a grid
    ax.grid(True)

    if figsave_filepath:
        plt.savefig(figsave_filepath, format="png", bbox_inches="tight")

    plt.show()


def plot_model_summary_plotly(
    method_model_results_summary,
    main_variable_name,
    vs_variable_name,
    x_variable,
    y_variable,
):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    print(
        f"Plotting {main_variable_name} vs {vs_variable_name} genes across {x_variable}"
    )
    fig = make_subplots(
        rows=1, cols=1, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.1
    )

    for method, method_df in method_model_results_summary.groupby("method"):
        fig.add_trace(
            go.Scatter(
                x=method_df[x_variable],
                y=method_df[y_variable],
                mode="markers+lines",
                name=marker_styles_dict[method]["label"],
                line=dict(color=marker_styles_dict[method]["marker_color"], width=2),
                marker=dict(
                    symbol=marker_styles_dict[method]["marker_plotly"],
                    size=10,
                    # marker_color and face_color based on marker_styles_dict
                    color=marker_styles_dict[method]["face_color"],
                    line=dict(
                        color=marker_styles_dict[method]["marker_color"], width=1
                    ),
                ),
                # hovertext=f"Method: {method}<br>{'<br>'.join([f'{k}: {v}' for k, v in method_df.to_dict().items()])}",
                hovertext=method_df.apply(
                    lambda x: f"Method: {method}<br>{'<br>'.join([f'{k}: {v}' for k, v in x.to_dict().items()])}",
                    axis=1,
                ),
                hoverinfo="text",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    fig.update_xaxes(title_text=x_variable, row=1, col=1)
    fig.update_yaxes(title_text=y_variable, row=1, col=1)

    # make x-axis log scale
    fig.update_xaxes(type="log", row=1, col=1)
    # square plot fig
    fig.update_layout(height=600, width=600, showlegend=True)
    fig.show()
