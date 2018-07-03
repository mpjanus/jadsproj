import numpy as np
import pandas as pd
import scipy.stats as spstats
import skimage.io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
import pathlib as path

import warnings
warnings.filterwarnings(action='once')


# ----------------------------------------------------------------------------------
# Image IO :
# ----------------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------------
# Image Display :
# ----------------------------------------------------------------------------------

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


def showheatmap(imgs, heats, cmapname='summer', opacity=0.7, heatdepend_opacity = True):
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

            # color map that - if enabled - is more transparent for low values
            org_cmap = plt.get_cmap(cmapname)
            alpha_cmap = org_cmap(np.arange(org_cmap.N))                    
            if (heatdepend_opacity):                
                alpha_cmap[:, -1] = np.linspace(0, 1, org_cmap.N)
            alpha_cmap = ListedColormap(alpha_cmap)

            plt.imshow(overlay, cmap=alpha_cmap, alpha=opacity, vmin=0, vmax=1)

            i = i + 1
    plt.show()


# ----------------------------------------------------------------------------------
# Image Slicing And Statistics:
# ----------------------------------------------------------------------------------

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


def getslicedimg(img, sy, sx, ny, nx):
    """ get the slice at index sy, sx when divided into ny, nx slices """
    h, w = img.shape
    sh = h // ny
    sw = w // nx
    return img[sy * sh:sy * sh + sh, sx * sw: sx * sw + sw]

def sliceimg_df(imgnames, ny, nx):
    """ creates dataframe with image slices; actually alias for slicestats without stat functions """
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




def stat_names(statfunclist):
    """ Helper to return a list of names of the statistics functions in statfunclist."""
    return [f.__name__ for f in statfunclist]

def normalized_names(listofnames, pre = '|', post = '|' ):
    """ Helper to return a list of names formatted as the normalized version."""
    return [(pre + n + post) for n in listofnames]


def norm_standardize(df, columnname):
    """ Returns a dataframe column using standard normalization, i.e. |x| = (x - mean) / std_dev  """
    return (df[columnname] - df[columnname].mean()) / df[columnname].std()

def norm_minmax(df, columnname):
    """ Returns a dataframe column using standard normalization, i.e. |x| = (x - mean) / std_dev  """
    return (df[columnname] - df[columnname].min() ) / (df[columnname].max() - df[columnname].min())


def normalize(df, list_of_columnnames, normfunc = norm_standardize, columnname_pre = '|', columname_post = '|' ):
    """n Adds a normalized version of the df columns"""
    for columnname in list_of_columnnames:
        norm_columnname = columnname_pre + columnname + columname_post
        df[norm_columnname] = normfunc(df, columnname)


def getimgslices_fromdf(df, imgfilename, stat_field_name = None):
    """ Get all slices for specific image from the dataframe and return as 2d array of images."""
    df_img = df.loc[df['filename'] == imgfilename]
    fullimg = loadtiff(imgfilename)
    ny = df_img.iloc[0]['n_y']
    nx = df_img.iloc[0]['n_x']
    imgs = np.empty((ny, nx), dtype=object)
    stats = np.empty((ny,nx), dtype=float)
    for index, row in df_img.iterrows():
        sy = row['s_y']
        sx = row['s_x']
        imgs[sy,sx] =  getslicedimg(fullimg, sy, sx, ny, nx)
        if (stat_field_name != None):
            stats[sy,sx] = row[stat_field_name]
    return imgs, stats

# ----------------------------------------------------------------------------------
# Interactive Plots:
# ----------------------------------------------------------------------------------

def plotwithimg(df, x_field, y_field, imgloadfunc, interactive=True):
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
    graph.plot(df[x_field],df[y_field], linestyle='none', marker='o', picker=tolerance)
    graph.set_xlabel(x_field)
    graph.set_ylabel(y_field)
    graph.set_title(y_field + ' vs ' + x_field)
    imginset = fig.add_subplot(gs[0:2,3:5])
    imginset.axes.get_xaxis().set_ticks([])
    imginset.axes.get_yaxis().set_ticks([])
 
    text = graph.text(1, 1, '[ .. ]', ha='right', va='top', transform=graph.transAxes)
    cursor = graph.scatter([df[x_field].iloc[0]], [df[y_field].iloc[0]],s=130, color='red', alpha=0.7)

    # state to keep
    ix = 0  # selected index
    selline = graph.lines[0]   # selected plot
    npoints = len(selline.get_xdata())

    # event handler for interactivity
    def show_selected_point():
        x, y = selline.get_xdata(), selline.get_ydata()

        # textual indication of datapoint:
        tx = '[{0:5d}: ({1:8.2f}, {2:8.2f})]'.format(ix, x[ix], y[ix])
        text.set_text(tx)

        # highlighting the datapoint:
        cursor.set_offsets((x[ix], y[ix]))

        # showing the corresponding image in the inset
        im = imgloadfunc(df, ix)
        imginset.imshow(im, cmap='gray')
        fig.canvas.draw()

    def on_pick(event):
        artist = event.artist   # note: an 'artist' is an object in a pyplot
        if (isinstance(artist, plt.Line2D)):
            nonlocal selline, ix    # closures!

            selline = artist
            ind = event.ind
            ix = ind[0]
            show_selected_point()

    def on_key(event):
        nonlocal ix
        # most keys are already caught in interactive mode, so used the ones that are not
        if (event.key == 'down'):
            ix = max(ix - 1,0)
            show_selected_point()
        if (event.key == 'up'):
            ix = min(ix + 1, npoints-1)
            show_selected_point()

    def on_click(event):
        # not used yet, but maybe hook up image click func
        # if (event.dblclick and (event.inaxes == imginset)):
        #    fig.canvas.draw()
        return

    if interactive:
        fig.canvas.callbacks.connect('pick_event', on_pick)
        fig.canvas.callbacks.connect('key_press_event', on_key)
        fig.canvas.callbacks.connect('button_press_event', on_click)


    plt.show()


def getimgslice(df, rowindex):
    """
    returns an image slice (as an image) as described in a dataframe row
    that has the format of the slicestats function
    """
    if (df.empty):
        raise ValueError('dataframe is empty.')
    dfrow = df.iloc[[rowindex]]

    img_filename = dfrow.iloc[0]['filename']
    fullimg = loadtiff(img_filename)
    sx = dfrow.iloc[0]['s_x']
    sy = dfrow.iloc[0]['s_y']
    nx = dfrow.iloc[0]['n_x']
    ny = dfrow.iloc[0]['n_y']
    return getslicedimg(fullimg, sy, sx, ny, nx)

def highlightimgslice(df, rowindex, unhighlightfactor=0.6):
    """
    returns the image in the df at specified row with the slice shown highlighted
    df-> dataframe with the format of the slicestats function
    rowindex -> row in the dataframe for which image should be retrieved
    unhighlightfactor -> the scaler for the unhighlighted area
    linewidth -> width of border around highlighted slice
    """
    if (df.empty):
        raise ValueError('dataframe is empty.')
    dfrow = df.iloc[[rowindex]]

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
    light = np.max(img)
    lw = max(1, (int)(0.003*w), (int)(0.003*h))    
    himg[sy_start:sy_start+lw, sx_start:sx_end] = light
    himg[sy_end-lw:sy_end, sx_start:sx_end] = light
    himg[sy_start:sy_end, sx_start:sx_start+lw] = light
    himg[sy_start:sy_end, sx_end-lw:sx_end] = light
    himg[sy_start+lw:sy_start+lw+1, sx_start+lw:sx_end-lw] = dark  
    himg[sy_end-lw-1:sy_end-lw, sx_start+lw:sx_end-lw] = dark
    himg[sy_start+lw:sy_end-lw, sx_start+lw:sx_start+lw+1] = dark
    himg[sy_start+lw:sy_end-lw, sx_end-lw-1:sx_end-lw] = dark

    return himg



# ----------------------------------------------------------------------------------
# Image Statistical functions:
# ----------------------------------------------------------------------------------

# Histogram operations
# -------------------------
def img_histogram(img, bins=1024, normalize=False, ref_interval_only=False, ref_range=0.95, bincenters=False):
    """Returns the histogram of the image, which is two arrays, one with the counts and one
     with the bin centers or edges"""
    if ref_interval_only:
        tail = (1 - ref_range) / 2
        low = img_blacktail(img, tail)
        high = img_whitetail(img, tail)
        hist = np.histogram(img, bins, range=(low, high), density=normalize)
    else:
        hist= np.histogram(img, bins, density=False)
    histcnts = hist[0]
    histbins = hist[1]
    if (normalize):
        histcnts = norm_histcounts(hist[0])
    if (bincenters):
        prev = 0
        newbins = histbins.copy()
        for i in range(0,len(histbins)):
            newbins[i] = 0.5 * (prev + histbins[i])
            prev = histbins[i]
        histbins = newbins
    return (histcnts, histbins)


    return hist


def norm_histcounts(histarray):
    sum = max(1, np.sum(histarray)) # avoid div by 0
    return histarray/sum

def smooth_histogram(hist, window=5):
    sigma = window
    gaussian_func = lambda x, sigma: 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x**2)/(2*sigma**2))
    gau_x = np.linspace(-2.7*sigma, 2.7*sigma, 6*sigma)
    gau_mask = gaussian_func(gau_x, sigma)
    return np.convolve(hist, gau_mask, 'same')


# Common Statistics
# -------------------------


def img_mean(img):
    """Return the mean pixel intensity of the image."""
    return np.mean(img)

def img_max(img):
    """Return the maximum pixel intensity of the image."""
    return np.max(img)

def img_min(img):
    """Return the minimum pixel intensity of the image."""
    return np.min(img)

def img_range(img):
    """Return the range of pixel intensities of the image.], i.e. max -  min"""
    return np.max(img) - np.min(img)

def img_refinterval(img, range=0.95):
    """Returns the inteval of pixel intensities in the image that fall within
    the specified range (i.e. dropping the outliers outside."""
    if (range==1):
        return np.max(img) - np.min(img)
    tail = (1-range)/2
    return np.percentile(img, 100*(range+tail)) - np.percentile(img, 100*tail)

def img_refinterval_low(img, range=0.95):
    """Returns the 'distance' between the blacktail cut-point and mean of pixel intensities in the image that fall within
    in the image. This is an indication of assymetry when compared to high part."""
    tail = (1-range)/2
    return np.mean(img) - np.percentile(img, 100*tail)

def img_refinterval_high(img, range=0.95):
    """Returns the 'distance' between the whitetail cut-point and mean of pixel intensities in the image that fall within
    in the image. This is an indication of assymetry when compared to high part."""
    tail = (1-range)/2
    return np.percentile(img, 100*(range+tail)) - np.mean(img)


def img_blacktail(img, tail=0.025):
    """Returns the 'black tail' of the pixel intensities, which is the lower part
     of the histogram that is typically considered as outliers or noise."""
    return np.percentile(img, 100*tail)

def img_whitetail(img, tail=0.025):
    """Returns the 'white tail' of the pixel intensities, which is the upper part
     of the histogram that is typically considered as outliers or noise."""
    return np.percentile(img, 100*(1-tail))


def img_median(img):
    """Return the median of the pixel intensities of the image."""
    return np.median(img)

def img_var(img):
    """Return the variance (=std^2) of the pixel intensities of the image."""
    return np.var(img)

def img_relstd(img):
    """Return the variance (=std^2) of the pixel intensities of the image."""
    return np.std(img) / (np.mean(img) + 0.001)  # +0.001 to avoid div by 0


def img_std(img):
    """Return the standard deviation (=sqrt(variance)) of the pixel intensities of the image."""
    return np.std(img)


# Quantile Statistics:
# -------------------------
# see http://influentialpoints.com/Training/quantiles_as_summary_statistics-principles-properties-assumptions.htm

def img_quartile1(img):
    """Return the first quartile of the pixel intensities of the image. """
    return np.percentile(img, 25)

def img_quartile2(img):
    """Return the second quartile of the pixel intensities of the image; this is the median. """
    return np.percentile(img, 50)

def img_quartile3(img):
    """Return the third quartile of the pixel intensities of the image. """
    return np.percentile(img, 75)

def img_interquartilerange(img):
    """Returns the 'distance' between the third and first quantile of the pixel intensities of the image."""
    return np.percentile(img, 75) - np.percentile(img,25)

def img_interquartilerange_low(img):
    """Returns the 'distance' between the first quantile and median of the pixel intensities of the image."""
    return np.median(img) - np.percentile(img,25)

def img_interquartilerange_high(img):
    """Returns the 'distance' between the third quantile and median of the pixel intensities of the image."""
    return np.percentile(img, 75) - np.median(img)

def img_percentile(img, percentage):
    """Returns the percentile of the pixel intensities of the image. You can feed this
    to the stats fucntions """
    return np.percentile(img, percentage)

def img_quintile1(img):
    """Return the first  quintile of the pixel intensities of the image. """
    return np.percentile(img, 20)
def img_quintile2(img):
    """Return the second  quintile of the pixel intensities of the image. """
    return np.percentile(img, 40)
def img_quintile3(img):
    """Return the third  quintile of the pixel intensities of the image. """
    return np.percentile(img, 60)
def img_quintile4(img):
    """Return the fourth quintile of the pixel intensities of the image. """
    return np.percentile(img, 80)

def img_mode(img, use_fast=True):
    if (use_fast):
        return img_histmode(img,1024)[0]
    else:
        return spstats.mode(img, axis=None)[0][0]

def img_mode_cnt(img, use_fast=True):
    if (use_fast):
        return img_histmode(img,1024)[1]
    else:
        return spstats.mode(img, axis=None)[1][0]


def img_histmode(img, bins=1024):
    """Return the mode of the grayscales in the image plus the counts, based
    on the histogram, an approximation for the most common value in the image.
    (the full version mode via scipy stats is slow)"""
    hist = img_histogram(img, bins=bins, bincenters=True)
    imax = np.argmax(hist[0])   # biggest count
    cntmax = hist[0][imax]
    valmax = hist[1][imax]
    return (valmax, cntmax)

def img_kurtosis(img):
    """Return the kurtosis (Fisher definition) of the image, a measure of tail
     heaviness of the histogram in comparison to a normal distribution."""
    return spstats.kurtosis(img, axis=None)

def img_skewness(img):
    """Returns the skeweness (Fisher defintion) of the distribution, a measure
    for the symmetry of the histogram in comparison to a normal distribution."""
    return spstats.skew(img, axis=None)


def img_shapevalue(img, bins = 100, smoothing_window = 5, dynamic_range_only=True):
    """Return the mvalue which is an indication for number of modes in the histogram
    (not very well known, see http://www.brendangregg.com/FrequencyTrails/modes.html)
    Note tat this only works well on smoothed histograms as it is derivative based
    """
    hist_org = img_histogram(img, bins, ref_interval_only=dynamic_range_only, ref_range=0.99)
    hist = smooth_histogram(hist_org[0], smoothing_window)
    
    sum = 0
    prevcount = hist[0]
    for index, count in np.ndenumerate(hist):
        if (index == 0): continue
        sum = sum + np.abs(count - prevcount)
        prevcount = count
    mvalue = sum / max(np.max(hist),1)  # avoid division by 0
    return mvalue



# Common combinations of statistical functions:
# ---------------------------------------------

def statfuncs_common():
    """Returns the most common image statistical functions, which is a 4 number summary
     of min, max, mean and standard deviation."""
    return [img_min, img_max, img_mean, img_std, img_median]

def statfuncs_common_ext():
    """Returns an extended version of the most common image statistical functions, which
     also includes the min and max of after outliers has been taken out ('white and black tail')"""
    return [img_min, img_max, img_mean, img_std, img_median, img_range, img_var, img_relstd, img_blacktail, img_whitetail, img_refinterval]

def statfuncs_selection1():
    """A selection of statistics that seem to describe well"""
    return [img_blacktail, img_mean, img_whitetail, img_std, img_quartile1, img_quartile3, img_median, img_interquartilerange]

def statfuncs_5numsummary():
    """Returns the 5 statistical functions known as the '5 number summary', i.e.
    the minimum, first quartile, median, third quartile and mzximum."""
    return [img_min, img_quartile1, img_median, img_quartile3, img_max]

def statfuncs_7numsummary():
    """Returns the 7 numbers that describes the histogram, i.e. the min, max, median
     and the quintiles."""
    return [img_min, img_quintile1, img_quintile2, img_median, img_quintile3, img_quintile4, img_max]

def statfuncs_boxandwhisker():
    """Returns the 'box-and-whisker' statistical functions often used for box plots, which
    is a 3 number summary of  the median, inter-quartile range and the reference interval."""
    return [img_median, img_interquartilerange, img_refinterval]

def statfuncs_boxandwhisker_ext():
    """Returns extended version of the 'box-and-whisker' statistical functions, including the
    ranges upper and lower parts."""
    return [img_median, img_interquartilerange, img_refinterval, img_interquartilerange_low, img_interquartilerange_high, img_refinterval_low, img_refinterval_high]

