import matplotlib.pyplot as plt

def scatPlotTwoLists(X1, Y1, X2, Y2, xlab, ylab, titlab, path, dpiinput, transparency):
    plt.ioff()
    plt.scatter(X1, Y1, edgecolors='r')
    plt.scatter(X2, Y2, edgecolors='b')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(titlab)
    # plt.show()
    plt.savefig(path, dpi=dpiinput, transparent=transparency)
    plt.close()