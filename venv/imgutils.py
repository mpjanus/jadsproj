import numpy as np
import pandas as pd
import skimage.io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
import pathlib as path

import warnings
warnings.filterwarnings(action='once')


def loadtiff(filename):
    """loads a tiff file"""
    img = skimage.io.imread(filename, plugin='tifffile')

    # if RGB, convert to 8 bit gray
    if (len(img.shape) == 3):  # 3rd dimension is rgb value
        img = np.trunc(rgb2gray(img) * 256)  # 8 bit values
    return img


def scanimgdir(folder, ext):
    """
    scans the given folder for image files as specified by ext
    and returns a dataframe with the file names
    """
    df = pd.DataFrame([['dummy']], columns=['filename'])
    df = df.drop(df.index[[0]])
    pathlist = path.Path(folder)
    for item in pathlist.iterdir():
        fullpath = str(item)
        if (fullpath[-len(ext):] == ext):
            df.loc[len(df)] = [fullpath]
    return df



def showimg(img):
    """shows the image.  Note that in pycharm, run environment needs to have Display=True """
    plt.interactive(False)      # required in pycharm
    plt.imshow(img, cmap='gray')
    plt.show()


def showimgs(imgs):
    """shows a 2d array of images. """
    plt.interactive(False)      # required in pycharm

    fig = plt.figure()
    ny, nx = imgs.shape
    i = 1
    for iy in range(ny):
        for ix in range(nx):
            subfig = fig.add_subplot(ny, nx, i)
            subfig.axes.get_xaxis().set_ticks([])
            subfig.axes.get_yaxis().set_ticks([])
            plt.imshow(imgs[iy, ix], cmap='gray')
            i = i + 1
    plt.show()


def showheatmap(imgs, heats, cmapname='summer', opacity=0.9):
    """
    shows a 2d array of images with a heatmap overlay.
    imgs - 2d array of images
    heats - 2d array of heats [0-1]
    cmapname - see https://matplotlib.org/examples/color/colormaps_reference.html
    opacity - opacity of the heat overlay
    """
    plt.interactive(False)      # required in pycharm

    ny, nx = imgs.shape
    fig = plt.figure()

    i = 1
    for iy in range(ny):
        for ix in range(nx):
            subfig = fig.add_subplot(ny, nx,i)
            subfig.axes.get_xaxis().set_ticks([])
            subfig.axes.get_yaxis().set_ticks([])

            img = imgs[iy, ix]
            plt.imshow(img, cmap='gray')
            overlay = np.full(img.shape, heats[iy,ix])

            # color map that is more transparent for low values
            org_cmap = plt.get_cmap(cmapname)
            alpha_cmap = org_cmap(np.arange(org_cmap.N))
            alpha_cmap[:, -1] = np.linspace(0, 1, org_cmap.N)
            alpha_cmap = ListedColormap(alpha_cmap)

            plt.imshow(overlay, cmap=alpha_cmap, alpha=opacity, vmin=0, vmax=1)

            i = i + 1
    plt.show()



def sliceimg(img, ny, nx):
    """
    slices the image up in subimages;
    nx, ny -> number of subimages in x resp. y direction
    return an array with the subimages
    """
    h, w = img.shape
    sh = h // ny
    sw = w // nx
    imgs = np.empty((ny, nx), dtype=object)
    for iy in range(ny):
        for ix in range(nx):
            slice = img[iy*sh:iy*sh + sh, ix*sw: ix*sw + sw]
            imgs[iy, ix] = slice
    return imgs

def sliceimg_df(imgnames, ny, nx):
    """ creates datframe with image slices; actually alias for slicestats without stat functions """
    return slicestats(imgnames, ny, nx)

def slicestats(imgnames, ny, nx, stats=None):
    """
    Loads each image in the list, slices it up and then apply
    statistics on each slice
    imgnames -> list of the image file names
    ny, nx -> number of slices in y and x direction
    stats -> list of statistical functions that will be called
    return a dataframe with image name and slice plus the statistics
    """
    df = pd.DataFrame([['dummy_file', 0, 0, 0, 0, 'dummy']], columns=['filename', 's_y','s_x', 'n_y', 'n_x', 'alias'])
    df = df.drop(df.index[[0]])

    # add colums for the provided statistical functions
    if (stats != None):
        for stat in stats:
            if (not hasattr(stat, '__call__')):
                raise TypeError('stats item is not a function')
            df[stat.__name__] = np.nan

    # make a copy of a single row as a template for later re-ordering of columns
    df_def = df.append({'n_y': 0}, ignore_index=True)

    c = 0  # counter for alias
    for s in imgnames:

        img = loadtiff(s)
        slices = sliceimg(img, ny, nx)
        for sy in range(ny):
            for sx in range(nx):
                # create a new dataframe row, leaving statistical values blank
                alias = 'img' + str(c) + "_" + str(sy) + "-" + str(sx)
                newdata = pd.DataFrame([[s, sy, sx, ny, nx, alias]], columns=df_def.columns[0:6])

                # call the statistical functions and fill in the result
                if (stats != None):
                    for stat in stats:
                        newdata[stat.__name__] = stat(slices[sy,sx])

                # add the new dataframe row
                df = pd.concat([df, newdata], ignore_index = True)

        c += 1

    # pd.concat changes column order (bug) so change it back:
    df = df.reindex(df_def.columns, axis='columns')
    return df

def img_mean(img):
    return np.mean(img)

def img_max(img):
    return np.max(img)

def img_min(img):
    return np.min(img)

def img_median(img):
    return np.median(img)

def img_var(img):
    return np.var(img)

def img_std(img):
    return np.std(img)



def plotwithimg(dfimgs, x_field, y_field, imgloadfunc, interactive=True):
    """
    Show an interactive scatter plot, showing corresponding image as
     provided by the imgloadfunc when data is selected.
    dfimgs -> A dataframe with image information and associated data
    x_field -> name of column with x data
    y_field -> name of column with y data
    imgloadfunc -> a function which will get the df row and should return
                   the corresponding image
    Remark: 
    for the interactive graph to work in jupyter, include %matplotlib notebook. You
    may need to restart the kernel before it works
    """

    if not interactive:
        plt.interactive(False)

    tolerance = 10 # points
    fig = plt.figure(1, figsize=(8,6))
    gs = gridspec.GridSpec(4, 5)
    graph = fig.add_subplot(gs[0:4,0:3])   
    graph.plot(dfimgs[x_field],dfimgs[y_field], linestyle='none', marker='o', picker=tolerance)
    imginset = fig.add_subplot(gs[0:2,3:5])
    imginset.axes.get_xaxis().set_ticks([])
    imginset.axes.get_yaxis().set_ticks([])
 
    text = graph.text(1, 1, '[ .. ]', ha='right', va='top', transform=graph.transAxes)
    cursor = graph.scatter([dfimgs[x_field][0]], [dfimgs[y_field][0]],s=130, color='green', alpha=0.7)  

    # event handler for interactivity
    def on_pick(event):
        artist = event.artist   # note: an 'artist' is an object in a pyplot    
        x, y = artist.get_xdata(), artist.get_ydata()
        ind = event.ind
        ix = ind[0]  
        
        # textual indication of datapoint:
        tx = '[{0:5d}: ({1:8.2f}, {2:8.2f})]'.format(ix, x[ix], y[ix])
        text.set_text(tx)       

        # highlighting the datapoint:
        cursor.set_offsets((x[ix], y[ix]))

        # showing the corresponding image in the inset
        im = imgloadfunc(dfimgs.iloc[[ix]])
        imginset.imshow(im, cmap='gray')

    if interactive:
        fig.canvas.callbacks.connect('pick_event', on_pick)

    plt.show()



def getslicedimg(img, sy, sx, ny, nx):
    """ get the slice at index sy, sx when divided into ny, nx slices """
    h, w = img.shape
    sh = h // ny
    sw = w // nx
    return img[sy * sh:sy * sh + sh, sx * sw: sx * sw + sw]



def getimgslice(dfrow):
    """
    returns an image slice as described in a dataframe row
    that has the format of the slicestats function
    """
    if (dfrow.empty):
        raise ValueError('dataframe is empty.')
    img_filename = dfrow.iloc[0]['filename']
    fullimg = loadtiff(img_filename)
    sx = dfrow.iloc[0]['s_x']
    sy = dfrow.iloc[0]['s_y']
    nx = dfrow.iloc[0]['n_x']
    ny = dfrow.iloc[0]['n_y']
    return getslicedimg(fullimg, sy, sx, ny, nx)


def highlightimgslice(dfrow, unhighlightfactor=0.6):
    """
    returns the image in the dfrow with the slice shown highlighted
    dfrow -> dataframe with the format of the slicestats function
    unhighlightfactor -> the scaler for the unhighlighted area
    linewidth -> width of border around highlighted slice
    """
    if (dfrow.empty):
        raise ValueError('dataframe is empty.')

    # get img and slice specs
    img_filename = dfrow.iloc[0]['filename']
    img = loadtiff(img_filename)
    h, w = img.shape
    sx = dfrow.iloc[0]['s_x']
    sy = dfrow.iloc[0]['s_y']
    nx = dfrow.iloc[0]['n_x']
    ny = dfrow.iloc[0]['n_y']
    sh = h // ny
    sw = w // nx
    sx_start = sx * sw
    sy_start = sy * sh
    sx_end = sx_start + sw
    sy_end = sy_start + sh

    # create a copy, but with lower intensities; then set the slice to original values
    himg = img * unhighlightfactor
    himg[sy_start:sy_end, sx_start:sx_end] = img[sy_start:sy_end, sx_start:sx_end]

    # draw marking line around slice
    dark = 0

    lw = max(1, (int)(0.003*w), (int)(0.003*h))
    himg[sy_start:sy_start+lw, sx_start:sx_end] = dark
    himg[sy_end-lw:sy_end, sx_start:sx_end] = dark
    himg[sy_start:sy_end, sx_start:sx_start+lw] = dark
    himg[sy_start:sy_end, sx_end-lw:sx_end] = dark

    return himg