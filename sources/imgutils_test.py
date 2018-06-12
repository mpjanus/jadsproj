import numpy as np
import pandas as pd
import imgutils

# TEST FUNCTIONS:

TEST_DIR = '' # ''../../data/Crystals_Apr_12/Tileset7'

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
    imgutils.showheatmap(b, heats)

def test_scanimgdir():
    df = imgutils.scanimgdir(TEST_DIR, '.tif')
    print(df.head)

def test_slicestats():
    statfuncs = [imgutils.img_min, imgutils.img_median, imgutils.img_max, imgutils.img_std]
    df_files = imgutils.scanimgdir(TEST_DIR, '.tif')
    df_slices = imgutils.slicestats(list(df_files['filename'][:]), 2, 2, statfuncs)
    print(df_slices.head(10))
    return df_slices

def test_plotwithimg():
    statfuncs = [imgutils.img_min, imgutils.img_median, imgutils.img_max, imgutils.img_std]
    df_files = imgutils.scanimgdir(TEST_DIR, '.tif')
    df = imgutils.slicestats(list(df_files['filename'][:]), 2, 2, statfuncs)
    imgutils.plotwithimg(df, 'img_min', 'img_max', imgutils.getimgslice, True)

def test_getimgslice_from_slicestatdf(i):
    statfuncs = [imgutils.img_median]
    df_files = imgutils.scanimgdir(TEST_DIR, '.tif')
    df_slices = imgutils.slicestats(list(df_files['filename'][:]), 2, 2, statfuncs)
    img = imgutils.getimgslice(df_slices.iloc[[i]])
    imgutils.showimg(img)

def test_highlightimgslice(i):
    df_files = imgutils.scanimgdir(TEST_DIR, '.tif')
    df_slices = imgutils.sliceimg_df(list(df_files['filename'][:]), 6, 6)
    print(df_slices.head())
    img = imgutils.highlightimgslice(df_slices.iloc[[i]])
    imgutils.showimg(img)

# EXECUTE TESTS:
#test_scanimgdir()
#test_heatmap2()
#test_slicestats()
#test_plotwithimg()
#test_getimgslice_from_slicestatdf(1)
test_highlightimgslice(8)
print("test")





