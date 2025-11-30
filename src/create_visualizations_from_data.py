"""
This module stores various visualizations functions to plot results
from if desired.  Be sure to modify the INPUT paths to point to your
own CSV files with spaCy results.
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
sns.set_theme(style="darkgrid")

INPUT_PER_LABEL = Path("C:\\Users\\allis\\OneDrive\\code_repos\\AIT726_project\\data\\plot_data\\results_per_label.csv")
INPUT_OVERALL = Path("C:\\Users\\allis\\OneDrive\\code_repos\\AIT726_project\\data\\plot_data\\results_overall.csv")
INPUT_RUNS = Path("C:\\Users\\allis\\OneDrive\\code_repos\\AIT726_project\\data\\plot_data\\results_all_runs.csv")

def ingest_spacy_table_results(data_path: Path) -> pd.DataFrame:
    """
    Ingest spaCy table results from a CSV file.
    """
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    return df

def setup_fig_ax(width=8, height=6, style="whitegrid"):
    sns.set_theme(style=style)
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax

def style_ax(
        ax,
     #   ax_x_limit: tuple[float, float] = (0, 100),
     #   ax_y_limit: tuple[float, float] = (0, 100),
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title_loc: str = "left",
        title_size: int = 14,
        title_pad: int = 10,
        title_weight: str = "bold",
        label_size: int = 12,
        label_pad: int = 20,
        label_weight: str = "bold",
    ):  
    """Apply consistent styling to a matplotlib/seaborn Axes."""

 ##   ax.set_xlim(ax_x_limit)
 #   ax.set_ylim(ax_y_limit)
    if title is not None:
        ax.set_title(
            title,
            loc=title_loc,
            fontsize=title_size,
            pad=title_pad,
            fontweight=title_weight,
        )

    if xlabel is not None:
        ax.set_xlabel(
            xlabel, 
            fontsize=label_size, 
            labelpad=label_pad,
            fontweight=label_weight
        )

    if ylabel is not None:
        ax.set_ylabel(
            ylabel, 
            fontsize=label_size, 
            labelpad=label_pad, 
            fontweight=label_weight
        )

    return ax

def style_tickmarks(
        ax: plt.Axes,
        which: str = 'both',
        direction: str = 'in',
        color: str = 'black',
        length: int = 3,
        bottom: bool = True,
        left: bool = True,
        minorticks: str = "both",
        major_tick_interval: int = 5,
        minor_ticks_per_major: int = 5,
    ):

    """Apply consistent styling to tick marks on the current Axes."""
    ax = plt.gca()
    ax.tick_params(
        which = which, 
        direction=direction, 
        color = color,
        length = length,
        bottom = bottom, 
        left = left
    ) 
    ax.grid(which="major", color=color, linewidth=.5)

    match minorticks:

        case("both" | True):
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(AutoMinorLocator(minor_ticks_per_major))
            ax.yaxis.set_minor_locator(AutoMinorLocator(minor_ticks_per_major))

        case("none" | False):
            ax.minorticks_off()

        case("x"):
            ax.xaxis.set_minor_locator(AutoMinorLocator(minor_ticks_per_major))

        case("y"):
            ax.yaxis.set_minor_locator(AutoMinorLocator(minor_ticks_per_major))

        case __:
            raise ValueError(f"Invalid value for minorticks: {minorticks}")

    ax.xaxis.set_major_locator(MultipleLocator(major_tick_interval))

    return ax

def style_legend(
    ax,
    title: str | None = None,
    loc: str = "best",
    ncol: int = 1,
    fontsize: int = 10,
    title_fontsize: int | None = None,
    frameon: bool = False,
    bbox_to_anchor: tuple | None = None,
    sort_labels=True,
):
    """ Apply consistent styling to the legend of a matplotlib/seaborn Axes.
    """

    # Get existing handles/labels from whatever seaborn just drew
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None  # nothing to style

    # Sort labels alphabetically if requested
    if sort_labels:
        labels, handles = zip(*sorted(zip(labels, handles), 
                key=lambda x: x[0]))

    # Recreate the legend using the public API
    leg = ax.legend(
        handles,
        labels,
        title=title,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        frameon=frameon,
    )

    # Title styling
    if leg.get_title() is not None:
        leg.get_title().set_fontsize(title_fontsize or fontsize)
        leg.get_title().set_fontweight("bold")

    # Text styling
    for text in leg.get_texts():
        text.set_fontsize(fontsize)

    return leg

def plot_metric_trends(
        x_name: str, 
        y_name: str, 
        column_name:str):
    """
    Plot trends of a specific metric over multiple runs.
    """
    df = ingest_spacy_table_results(INPUT_RUNS)
    df[x_name] = pd.to_numeric(df[x_name])
    df[y_name] = pd.to_numeric(df[y_name])

    fig, ax = setup_fig_ax()

    sns.lineplot(
            x=x_name, 
            y=y_name,
            hue=column_name, 
            style=column_name,
            data=df,
            ax=ax,
        )
    
    style_ax(
            ax,
            title=f"TREND OF {y_name.upper()} OVER {x_name.upper()}",
            xlabel=x_name,
            ylabel=y_name,
        )

    style_tickmarks(
        ax,
        which='both',
        direction='in',
        color ='black',
        length=3,
        bottom=True,
        left=True,
        minorticks='both',
        major_tick_interval=5,
        minor_ticks_per_major=5,
    )

    plt.tight_layout()
    plt.show()

    return fig, ax

def plot_label_catplot(
        x_name: str, 
        y_name: str, 
        column_name: str
    ):
    """
    Plot per-label results as a categorical plot.
    NEEDS IMPROVEMENT - alphabetize columns, add better styling 
    if plots are to be used in presentations.
    """

    df = ingest_spacy_table_results(INPUT_PER_LABEL)

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df,
        x=x_name,
        y=y_name,
        col=column_name,
        kind="strip",
        jitter=False,
        height=2.5,
        aspect=0.8,
        dodge=False,
    )

    g.set(
        xlim=(19, 93), 
        xlabel=f"{x_name.upper()} (%)", 
        ylabel=y_name.upper()
    )

    g.set_titles(col_template="{col_name}")
    g.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()

    return

def plot_label_dotplot(
        x_name: str, 
        y_name: str, 
        dot_labels: str
    ):
    """
    Plot per-label results as a dotplot.
    """
    fig, ax = setup_fig_ax()

    df = ingest_spacy_table_results(INPUT_PER_LABEL)

    df[y_name] = df[y_name].astype("category")

    ax = sns.stripplot(
        data=df,
        x=x_name,          
        y=y_name,        
        hue=dot_labels,     
        dodge=True,
        jitter=False,
        size=8,
    )

    n = len(ax.get_yticklabels())
    for y in np.arange(-0.5, n - 0.5, 1):
        ax.axhline(
            y=y,
            color="lightgray",
            linewidth=0.8,
            zorder=0,
        )

    style_ax(
            ax,
            title=f"PER LABEL {y_name.upper()} BY {x_name.upper()}",
            xlabel=f"{x_name.upper()} (%)",
            ylabel=y_name.upper(),
        )

    style_legend(
        ax,
        title=dot_labels.upper(), 
        loc="upper left",
        ncol=1,
        fontsize=9,
        frameon=True,
    )

    style_tickmarks(
        ax,
        which='both',
        direction='in',
        color ='black',
        length=3,
        bottom=True,
        left=True,
        minorticks='x',
        major_tick_interval=10,
        minor_ticks_per_major=2,
    
    )

    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
"""
    Various plotting function calls available for reporting.
    Based these off three different metrics I have from Prod/Spacy runs.
"""
  #  plot_label_catplot("F1", "RUN","TERM")
   # plot_label_dotplot("F1", "RUN","TERM")
    fig, ax = plot_metric_trends("EPOCH", "SCORE", "RUN")
