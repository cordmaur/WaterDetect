import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == "__main__":

    minimo = 0
    maximo = 1000

    # fonte1: https://matplotlib.org/examples/api/colorbar_only.html
    # fonte2: https://stackoverflow.com/questions/54986104/color-bar-only-with-custom-colors-values-and-gradient

    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=(4, 1))
    ax1 = fig.add_axes([0.05, 0.50, 0.90, 0.15])

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.

    norm = mpl.colors.Normalize(vmin=minimo, vmax=maximo)

    cdict = {'red': ((0.0, 0.0, 0.0),
                     (0.5, 0.0, 0.0),
                     (1.0, 1.0, 1.0)),

             'green': ((0.0, 0.0, 0.0),
                       (0.5, 0.8, 0.8),
                       (1.0, 0.0, 0.0)),

             'blue': ((0.0, 0.6, 1.0),
                      (0.5, 0.0, 0.0),
                      (1.0, 0.0, 0.0))}

    cmap = mpl.colors.LinearSegmentedColormap('custom', cdict)
    #=========================

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('Legenda')

    plt.show()
