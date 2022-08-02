def legend_without_duplicate_labels(figure, leg_loc='lower center', n_col=8):
    # The default key values work for the example plots, but may require editing for other data.
    handles, labels = figure.gca().get_legend_handles_labels()
    figure.legend(handles, labels, loc=leg_loc, ncol=n_col)

    return figure
