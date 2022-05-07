# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Imports
# -

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# + pycharm={"name": "#%%\n"}
# %matplotlib inline
mpl.rcParams["figure.figsize"] = (16, 9)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## First look at the data

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Data Loading

# + pycharm={"name": "#%%\n"}
data_path = "./../../data/train.csv"

# + pycharm={"name": "#%%\n"}
data = pd.read_csv(data_path)

# + [markdown] pycharm={"name": "#%% md\n"}
# ### General Info

# + pycharm={"name": "#%%\n"}
data.head()

# + pycharm={"name": "#%%\n"}
data.describe()

# + pycharm={"name": "#%%\n"}
y = data.loc[:, "log_price"]

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Duplicates

# + pycharm={"name": "#%%\n"}
data.duplicated().sum()

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Missing data

# + pycharm={"name": "#%%\n"}
data.isnull().sum()

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Simple Visualization

# + pycharm={"name": "#%%\n"}
data.iloc[:, 1:].hist(bins=50)
plt.tight_layout()
plt.show()

# + pycharm={"name": "#%%\n"}
plt.figure(figsize=(12, 6))
plt.hist(y, bins=50);


# + [markdown] pycharm={"name": "#%% md\n"}
# Let's take a look at categorical data now.

# + pycharm={"name": "#%%\n"}
def plot_categorical_count(data, col_name, title, xlabel, ylabel="count", log=False):
    data[col_name].value_counts().plot.bar(log=log)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# + pycharm={"name": "#%%\n"}
plot_categorical_count(data, "property_type", "Log property types", "property type", "count", log=True)

# + pycharm={"name": "#%%\n"}
data["property_type"].value_counts()

# + pycharm={"name": "#%%\n"}
plot_categorical_count(data, "room_type", "Log room types", "room type", "count")

# + pycharm={"name": "#%%\n"}
data["room_type"].value_counts()

# + pycharm={"name": "#%%\n"}
plot_categorical_count(data, "city", "City", "city name", "count")

# + pycharm={"name": "#%%\n"}
data["city"].value_counts()

# + pycharm={"name": "#%%\n"}
plot_categorical_count(data, "cancellation_policy", "Cancellation policy", "policy", "count")

# + pycharm={"name": "#%%\n"}
data["cancellation_policy"].value_counts()

# + pycharm={"name": "#%%\n"}
data["neighbourhood"].value_counts().head()

# + pycharm={"name": "#%%\n"}


# + pycharm={"name": "#%%\n"}


# + [markdown] pycharm={"name": "#%% md\n"}
# ## Correlations

# + pycharm={"name": "#%%\n"}
corr_matrix = data.corr()

# + pycharm={"name": "#%%\n"}
plt.figure(figsize=(20, 12))
plt.imshow(corr_matrix, cmap="inferno")
plt.xticks(np.arange(corr_matrix.columns.values.shape[0]), labels=corr_matrix.columns.values)
plt.yticks(np.arange(corr_matrix.columns.values.shape[0]), labels=corr_matrix.columns.values)
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha="right",
         rotation_mode="anchor")
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = plt.gca().text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                              ha="center", va="center", color="w")


# + pycharm={"name": "#%%\n"}
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


# + pycharm={"name": "#%%\n"}
corr_matrix["log_price"].sort_values(ascending=False)

# + pycharm={"name": "#%%\n"}
#todo: map in the background

# + pycharm={"name": "#%%\n"}
data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# + pycharm={"name": "#%%\n"}
categories = data.loc[:, "room_type"].unique()
new_data = data.set_index("room_type")["log_price"]
seperated_data = []
for category in categories:
    seperated_data.append(new_data.loc[category])


# + pycharm={"name": "#%%\n"}
def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Room types')


# + pycharm={"name": "#%%\n"}
fig, axes = plt.subplots()
axes.violinplot(seperated_data)
set_axis_style(axes, categories)
axes.set_ylabel("Log price");

# + pycharm={"name": "#%%\n"}
# todo: the same as about but add neighbourhood
