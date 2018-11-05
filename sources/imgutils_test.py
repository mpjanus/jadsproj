import numpy as np
import pandas as pd
import imgutils
import os

# TEST FUNCTIONS:

TEST_DIR = '' # ''../../data/Crystals_Apr_12/Tileset7'

def test_scanimgdir():
    df = imgutils.scanimgdir(TEST_DIR, '.tif')
    print(df.head)

def test_loadandshowimg():
    img = imgutils.loadtiff('testimage1.tif')
    imgutils.showimg(img)

def test_loadandsaveimg():
    img = imgutils.loadtiff('testimage1.tif')
    imgutils.savetiff('temp.tif', img)
    img2 = imgutils.loadtiff('temp.tif')
    imgutils.showimg(img2)
    if os.path.exists('temp.tif'): os.remove('temp.tif')

def test_loadandshowimgs():
    img1 = imgutils.loadtiff('testimage1.tif')
    img2 = imgutils.loadtiff('testimage2.tif')
    imgs = np.empty((2, 3), dtype=object)
    imgs[0,0] = imgs[1,1] = imgs[0,2] = img1
    imgs[0,1] = imgs[1,0] = imgs[1,2] = img2
    imgutils.showimgs(imgs)

def test_sliceimage(ny, nx):
    img = imgutils.loadtiff('testimage1.tif')
    slices = imgutils.sliceimg(img, ny, nx)
    imgutils.showimgs(slices)


def test_heatmap():
    """Load image, slice it up and create a heatmap"""
    a = imgutils.loadtiff('testimage1.tif')
    b = imgutils.sliceimg(a, 4, 4)
    heats = np.array([[0.1, 0.5, 0.1, 0],
                     [0.2, 0.6, 0.2, 0],
                     [0.2, 0.8, 0.4, 0.1],
                     [0, 0.3, 0.1, 0]])
    imgutils.showheatmap(b, heats)

def test_heatmap2():
    """Load image, slice it up and create a heatmap"""
    a = imgutils.loadtiff('testimage1.tif')
    b = imgutils.sliceimg(a, 4, 4)
    heats = np.zeros(shape=(4,4))
    for sy in range(4):
        for sx in range(4):
            heats[sy,sx] = 1 - (np.median(b[sy,sx]) / np.max(b[sy,sx]))
    imgutils.showheatmap(b, heats, tile_labels=True)

def test_largeheatmap():
    df_files = imgutils.scanimgdir(TEST_DIR, '.tif')
    n_files = len(df_files)
    df_imgstats = imgutils.slicestats(list(df_files['filename'][:]), 4, 4, [imgutils.img_std])
    # using mean as heat
    imgutils.show_large_heatmap(df_imgstats,'img_std', df_files['filename'], 1, n_files, annotate_tiles=True, show_extra_info=True)

def test_statfuncs(img):
    print('min: ', imgutils.img_min(img))
    print('max: ', imgutils.img_max(img))
    print('range: ', imgutils.img_range(img))
    print('median: ', imgutils.img_median(img))
    print('mean: ', imgutils.img_mean(img))
    print('std_dev', imgutils.img_std(img))


def test_slicestats_df():
    df_files = imgutils.scanimgdir(TEST_DIR, '.tif')
    df_slices = imgutils.sliceimg_df(list(df_files['filename'][:]), 3, 3)
    print(df_slices.head())
    return df_slices

def test_slicestats():
    statfuncs = [imgutils.img_min, imgutils.img_max, imgutils.img_mean, imgutils.img_std]
    df_files = imgutils.scanimgdir(TEST_DIR, '.tif')
    df_slices = imgutils.slicestats(list(df_files['filename'][:]), 4, 4, statfuncs)
    print(df_slices.head(8))
    return df_slices

def test_plotwithimg(thumbs = False, interactive=True):
    statfuncs = [imgutils.img_min, imgutils.img_max, imgutils.img_mean, imgutils.img_std]
    df_files = imgutils.scanimgdir(TEST_DIR, '.tif')
    df = imgutils.slicestats(list(df_files['filename'][:]), 4, 4, statfuncs)
    imgutils.plotwithimg(df, 'img_min', 'img_max', imgutils.getimgslice, interactive=interactive, thumbnails=thumbs)

def test_highlightimgslice(i):
    df_files = imgutils.scanimgdir(TEST_DIR, '.tif')
    df_slices = imgutils.sliceimg_df(list(df_files['filename'][:]), 4, 4)
    print(df_slices.head())
    img = imgutils.highlightimgslice(df_slices.iloc[[i]])
    imgutils.showimg(img)

# EXECUTE TESTS:
#test_scanimgdir()
#test_heatmap2()
#test_slicestats()
#test_loadandsaveimg()
#test_loadandshowimgs()
#test_plotwithimg(thumbs=True, interactive=False)
#test_plotwithimg(thumbs=False, interactive=True)
#test_plotwithimg(thumbs=True, interactive=True)

#test_highlightimgslice(8)

#test_largeheatmap()

# RESCALING IMAGES:
#sourcedir = '../data/Crystals_Apr_12/Tileset6_subset'
#targetdir = '../data/Crystals_Apr_12/Tileset6_subset_1K'
#imgutils.downsample_tiffs(sourcedir, targetdir, 2)


