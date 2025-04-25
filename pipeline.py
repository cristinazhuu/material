from collections import deque
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
import tifffile
from matplotlib import pyplot as plt
from matplotlib import patches
from cellpose import models
from pathlib import Path
from matplotlib.patches import Patch
import skimage
from scipy.signal import wiener
import os
import imageio
from matplotlib.widgets import RectangleSelector
def wavelength_to_rgb(wavelength, gamma=0.8):
    """
    Converts a wavelength in the range of 380 to 750 nm to an approximate RGB color.

    Parameters:
        wavelength (float): Wavelength in nanometers.
        gamma (float): Gamma correction factor.

    Returns:
        (int, int, int): Tuple representing RGB color values.
    """
    # Validate wavelength range
    if not (380 <= wavelength <= 750):
        return (0, 0, 0)

    # Initial RGB components and attenuation
    R = G = B = 0.0
    attenuation = 1.0

    # Define color transition ranges for each wavelength band
    if 380 <= wavelength < 440:
        R = (440 - wavelength) / (440 - 380)
        B = 1.0
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
    elif 440 <= wavelength < 490:
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength < 510:
        G = 1.0
        B = (510 - wavelength) / (510 - 490)
    elif 510 <= wavelength < 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
    elif 580 <= wavelength < 645:
        R = 1.0
        G = (645 - wavelength) / (645 - 580)
    elif 645 <= wavelength <= 750:
        R = 1.0
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)

    # Apply gamma correction and adjust by attenuation for realistic fading at spectrum edges
    R = int((R * attenuation) ** gamma * 255)
    G = int((G * attenuation) ** gamma * 255)
    B = int((B * attenuation) ** gamma * 255)

    return (R, G, B)


def wavelength_to_rgb_vect(vect, gamma=1):
    if isinstance(vect, int) | isinstance(vect, float):
        RGB_vect = wavelength_to_rgb(vect, gamma=gamma)
    else:
        RGB_vect = np.zeros((len(vect), 3))
        for i in range(len(vect)):
            RGB = wavelength_to_rgb(vect[i], gamma=gamma)
            RGB_vect[i, 0] = RGB[0]
            RGB_vect[i, 1] = RGB[1]
            RGB_vect[i, 2] = RGB[2]
    return RGB_vect


def SpectralStack2RGB(Stack, Ch_wavelength):
    RGB_vect = wavelength_to_rgb_vect(Ch_wavelength)
    Size = np.shape(Stack)
    RGB_img = np.zeros((Size[1], Size[2], 3))
    for ch in range(Size[0]):
        for col in range(3):
            RGB_img[:, :, col] = RGB_img[:, :, col] + Stack[ch, :, :] * RGB_vect[ch, col]
    return RGB_img


def phasor_transform(x, n_harm=1, axis=0, debug=False):
    """Computes the phasor transform of 'x'.
    n_harm: which harmonic of the fourier transform to take.
    axis: which dimension of 'x' the phasor transform is computed along."""
    print(x.shape)
    fft = np.fft.fft(x, axis=axis)
    # make sure to take nth harmonic from the correctly collapsed axis
    # this is fancy stuff to make this function work with any dimension of data
    n_harm_index = [slice(None) for _ in range(x.ndim)]
    n_harm_index[axis] = n_harm
    harm_0_index = [slice(None) for _ in range(x.ndim)]
    harm_0_index[axis] = 0

    # actual phasor transform expression
    gs = fft[tuple(n_harm_index)] / fft[tuple(harm_0_index)]
    g, s = np.real(gs), np.imag(gs)

    if debug:
        g_isnan = np.isnan(g)
        s_isnan = np.isnan(s)
        print(f'g nans: {np.sum(g_isnan)}')
        print(f'g good: {np.sum(~g_isnan)}')
        print(f's nans: {np.sum(s_isnan)}')
        print(f's good: {np.sum(~s_isnan)}')
    return (g, s)



def get_phasor_unmix_matrix_less_coords(mat, use_extra_coord=True, s4=False):
    """Generalized unmixing matrix for any number of mixed spectra."""
    # mat.shape: (n_channels, n_spectra (e.g. components))
    n_spectra = mat.shape[1]

    if use_extra_coord:
        n_full_harms = (n_spectra - 1) // 2
    else:
        n_full_harms = n_spectra // 2

    gs = np.zeros((n_spectra, n_spectra))
    for i in range(n_full_harms):
        n_harm = i + 1
        gs[:, 2 * i], gs[:, 2 * i + 1] = phasor_transform(mat, n_harm=n_harm, axis=0)

    if use_extra_coord:
        # an even number of spectra will have two cols blank after full harmonics
        if n_spectra % 2 == 0:
            # first add the s coordinate of the next harmonic
            g_last, s_last = phasor_transform(mat, n_harm=n_full_harms + 1, axis=0)
            gs[:, -2] = s_last

        # add a column of ones to make the matrix a square. This happens regardless of spectra parity
        gs[:, -1] = 1
    else:
        # an even number of spectra will be done after using full harmonics
        if n_spectra % 2 == 1:
            gs[:, -1] = 1
            print('this is being used')

    if s4:
        g_last, s_last = phasor_transform(mat, n_harm=n_full_harms + 1, axis=0)
        gs[:, -1] = s_last

    return gs


def image_to_components_multi_harm_less_coords(img, pure_phasors, use_extra_coord, s4=False):
    """Converts an (c, m, n) image into a (c, m, n) stack of unmixed components.

    Works for number of spectra that require multiple fft harmonics to unmix.
    Does not use the extra g or s coordinate if not necessary."""

    n_components = pure_phasors.shape[0]
    print('n_comps', n_components)

    if use_extra_coord:
        n_full_harms = (n_components - 1) // 2
    else:
        n_full_harms = n_components // 2
    stack_input = []

    for i in range(n_full_harms):
        harm_n = i + 1
        g_harm, s_harm = phasor_transform(img, n_harm=harm_n, axis=0)
        stack_input.extend([np.ravel(g_harm), np.ravel(s_harm)])

    if not use_extra_coord:
        if n_components % 2 == 1:
            if s4:
                g_last, s_last = phasor_transform(img, n_harm=n_full_harms + 1, axis=0)
                stack_input.append(np.ravel(s_last))
                print('4th harmonic is being used')

            else:
                stack_input.append(np.ones_like(stack_input[0]))


    else:
        if n_components % 2 == 0:
            g_last, s_last = phasor_transform(img, n_harm=n_full_harms + 1, axis=0)
            stack_input.append(np.ravel(s_last))

        stack_input.append(np.ones_like(stack_input[0]))

    int_img = np.sum(img, axis=0)  # total intensities for each pixel

    gs_mat = np.stack(stack_input, axis=-1)  # (n_pix, n_harm * 2)
    print('gs_mat.shape', gs_mat.shape)
    phasor_to_frac = np.linalg.inv(pure_phasors)
    print('phasor_to_frac.shape', phasor_to_frac.shape)
    spectra_fracs = gs_mat @ phasor_to_frac

    spectra_fracs[np.isnan(spectra_fracs) | (spectra_fracs < 0)] = 0
    spectra_fracs /= np.sum(spectra_fracs, axis=1)[:, None]
    spectra_fracs[np.isnan(spectra_fracs)] = 0

    spectra_fracs_img = np.zeros((n_components, *int_img.shape))
    # TODO: try transpose method, should be faster
    for i in range(n_components):
        spectra_fracs_img[i] = np.reshape(spectra_fracs[:, i], int_img.shape)

    return spectra_fracs_img * int_img  # (c, 1024, 1024)


def get_phasor_unmix_matrix_manual(mat, use_extra_coord):
    # mat: (channels, n_spectra)
    n_spectra = 4
    gs = np.zeros((n_spectra, n_spectra))

    g1, s1 = phasor_transform(mat, n_harm=1, axis=0)
    g2, s2 = phasor_transform(mat, n_harm=2, axis=0)
    gs = np.stack([g1, s1, g2, s2], axis=1)
    return gs


def load_pure_phasors(pure_spectra_path=None, dye_list=None, ch_vec=None, use_extra_coord=True, verbose=False,
                      manual=False, s4=False):
    """Loads the pure phasors from a spectra file.
    pure_spectra_path: Path object for desired set of spectra.
    dye_list: list of dyes present in the sample. Necessary since spectra file may have extra spectra.
    ch_vec: array of what channels to use. (i.e. 0-32 or 10-32, etc)
    use_extra_coord: whether or not to use the redundant last harmonic coordinate.
    :type s4: object"""

    with np.load(pure_spectra_path) as npz:
        dye_labels = npz['labels']
        pure_spectra = npz['spectra']

    dye_idxs = [np.nonzero(dye_labels == dye_name)[0][0] for dye_name in dye_list]
    pure_spectra = pure_spectra[:, dye_idxs]  # (32, n_spectra)
    if manual == True:
        pure_phasors = get_phasor_unmix_matrix_manual(pure_spectra, use_extra_coord=use_extra_coord)
    else:
        pure_phasors = get_phasor_unmix_matrix_less_coords(pure_spectra, use_extra_coord=use_extra_coord, s4=s4)
    if ch_vec is not None:
        pure_spectra = pure_spectra[ch_vec, :]

    print(pure_spectra.shape)

    # pure_phasors = get_phasor_unmix_matrix(pure_spectra)
    if verbose:
        print(pure_phasors.shape)
        print(pure_phasors)

    return pure_phasors
    # return np.load('/content/drive/Shareddrives/Laboratory for Fluorescence Dynamics/Data/2022.07.22 - Contractility 2DG/pure_phasors.npy')


def get_comps_by_name(mat, names):
    '''return only those components which were named.
    mat: (n_comps, n, m)
    names: list (len <= n_comps)
    '''
    try:
        idxs = [DYE_LIST.index(name) for name in names]
    except NameError as e:
        raise e("You have not defined a DYE_LIST. This is a list which determines the component appearance order.")

    return mat[idxs, ...]



def tile_stitching(img, m, n, bidirectional=False, percentage_overlap = 0):
    X = img.shape[-2]
    Y = img.shape[-1]
    dX = int(percentage_overlap/100*X/2)
    dY = int(percentage_overlap/100*Y/2)

    img_rec = np.zeros(np.array([img.shape[1], X-2*dX, Y-2*dY]) * np.array([1, m, n]))

    cnt_slice = 0
    for i in range(m):
        if bidirectional & ((i % 2) != 0):
            j_range = np.flip(np.arange(n))
        else:
            j_range = np.arange(n)
        for j in j_range:
            img_rec[:,i*(X-2*dX):(i+1)*(X-2*dX),j*(Y-2*dY):(j+1)*(Y-2*dY)] = img[cnt_slice,:,dX:X-dX,dY:Y-dY]
            cnt_slice += 1
    return img_rec

def rep_img(g_wiener, s_wiener, pure_phasor_g, pure_phasor_s,all_cellpose_masks, RGB_img, brighfield,comp_imgs_wiener ):
    n_plots_x = max(len(row_1_axis_titles), len(row_2_axis_titles))

    comp_fig = plt.figure(constrained_layout=True, figsize=(n_plots_x * 5, 8))
    gs = comp_fig.add_gridspec(2, 4 * n_comps)
    axs = [[comp_fig.add_subplot(gs[0, n_comps * i:n_comps * i + n_comps]) for i in range(4)],
           [comp_fig.add_subplot(gs[1, 4 * i:4 * i + 4]) for i in range(n_comps)]]

    comp_fig.suptitle(f'{cell_type} {treatment} {tile_id}', fontsize='30')

    axs[0][0].set_box_aspect(1)
    axs[0][0].hist2d(np.ravel(g_wiener), np.ravel(s_wiener), bins=HIST_BINS, range=[[-1, 1], [-1, 1]],
                     cmap='nipy_spectral')
    axs[0][0].scatter(pure_phasor_g, pure_phasor_s, c='white')
    circle = plt.Circle((0, 0), radius=1, color='w', fill=False)
    axs[0][0].add_patch(circle)
    axs[0][0].set_title(f'{cell_type} {treatment} {tile_id}\nHarmonic: {NTH_HARMONIC}')
    #
    # # plot all other images
    axs[0][1].imshow(all_cellpose_masks['cyto2'], cmap=MASK_CMAP, interpolation='none')
    axs[0][2].imshow(RGB_img / np.percentile(RGB_img, 99.5))
    axs[0][3].imshow(brightfield, cmap='gray', vmax=np.percentile(brightfield, 99))

    # plot individual components
    for i in range(n_comps):
        im = axs[1][i].imshow(comp_imgs_wiener[i], vmax=np.percentile(comp_imgs_wiener[i], 99), vmin=0, cmap='gray')
        # vmax=np.percentile(comp_imgs[i], 99)
        cbar = plt.colorbar(im, ax=axs[1][i], shrink=0.4)
    #
    #
    # # formatting
    for i, title in enumerate(row_1_axis_titles):
        axs[0][i].axis('off')
        title = row_1_axis_titles[i]
        axs[0][i].set_title(title, fontsize='15')

    for i, title in enumerate(row_2_axis_titles):
        axs[1][i].axis('off')
        axs[1][i].set_title(title, fontsize='15')


    plt.show()



def onselect(eclick, erelease):
    """Handles the selection of a region in the image."""
    global selected_region, ax2, ax3, fig

    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)

    # Ensure valid coordinates
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    # Extract selected region
    selected_region = channels[:32, y1:y2, x1:x2]

    # Extract intensity mask only for the selected region
    threshold_mask = (int_img[y1:y2, x1:x2] > np.mean(int_img[y1:y2, x1:x2]))

    # Compute phasor transform of selected region
    g_selected, s_selected = phasor_transform(selected_region, n_harm=NTH_HARMONIC, axis=0)
    s_selected = -s_selected
    # Apply threshold mask only if shapes match
    if threshold_mask.shape == g_selected.shape[1:]:
        g_selected = g_selected[:, threshold_mask]
        s_selected = -s_selected[:, threshold_mask]

    # Clear and update the phasor plot
    ax2.clear()
    ax2.hist2d(g_selected.ravel(), s_selected.ravel(), bins=HIST_BINS, range=[[-1, 1], [-1, 1]], cmap='nipy_spectral')
    ax2.scatter(pure_phasor_g, pure_phasor_s, c='white')
    circle = plt.Circle((0, 0), radius=1, color='w', fill=False)
    ax2.add_patch(circle)
    ax2.set_title("Phasor Plot of Selected Region")

    # Compute and plot the sum of selected region
    summed_values = np.sum(np.sum(selected_region, axis=1), axis=1)
    ax3.clear()
    ax3.plot(summed_values, marker='o')
    ax3.set_title("Sum of Selected Region Across Channels")
    ax3.set_xlabel("Channel Index")
    ax3.set_ylabel("Summed Intensity")

    fig.canvas.draw()


def process_file(file_path: Path) -> np.ndarray:
    """
    Read a single LSM/TIFF, run phasor‐unmixing, and return a (n_comps, H, W) array.
    """
    # 1) Read the raw stack
    raw = tifffile.imread(str(file_path))

    # 2) Select your 32 spectral channels + brightfield
    if raw.shape[0] == 33:
        channels = raw[CHANNEL_VEC]
        brightfield = raw[-1]
    else:
        channels = raw[CHANNEL_VEC]
        brightfield = np.zeros(raw.shape[1:])

    # 3) (Optional) Wiener filter
    filt = wiener(channels, (1,5,5)) if WIENER else channels

    # 4) Load or reuse your pure_phasors
    #    (if you already have a global `pure_phasors`, skip re‐loading,
    #     otherwise call load_pure_phasors here.)
    # pure_phasors = load_pure_phasors(DYE_SPECTRA_FP, DYE_LIST, CHANNEL_VEC, USE_EXTRA_COORD, s4=s4)

    # 5) Run the unmixing
    comps = image_to_components_multi_harm_less_coords(
        filt,
        pure_phasors,
        use_extra_coord=USE_EXTRA_COORD,
        s4=s4
    )

    return comps

## Setting all emprty arrays

filterted_channels = []
g_wiener = []
s_wiener = []
comp_imgs_wiener = []


DATA_PATH = Path('/Users/cristinazhu/PycharmProjects/PhasorFinal/pythonProject1')
DYE_SPECTRA_FP = DATA_PATH / 'spectra/Lipi Modified spectra.npz'


## CONFIGURATIONS
SAVE_DATA = True  # Whether or not unmixed components get saved.
SAVE_CELLPOSE_INPUTS = False  # whether or not CellPose inputs are prepared and saved from the data
USE_EXTRA_COORD = False  # whether or not to use redundant phasor harmonic component in unmixing.
COMPUTE_HISTOGRAMS = False  # don't set this to true, this was just to debug the intensity values
NTH_HARMONIC = 1  # what harmonic of phasor transform is computed and plotted
HIST_BINS = 100
INTENSITY_CMAP = 'gnuplot2'  # name of colormap used for intensity graph
MASK_CMAP = 'nipy_spectral'  # name of colormap used for CellPose masks
CHANNEL_VEC = np.arange(0, 32)  # which channels to use
# CHANNEL_VEC = np.arange(6, 32) # use less channels to better encapsule proper spectra range
CELL_DIAMETER = 200  # estimated cell diameter in pixels. Adjust this if the masks aren't great.
SHOW_MASK_FIG = False
s4 = True
WIENER = True

## Loading All Cellpose Model

MODEL_TYPE = 'cyto2'
cellpose_model = models.CellposeModel(gpu=True, model_type=MODEL_TYPE)

model_path_from_name = {
    'custom_contractility': DATA_PATH / 'models/Contractility_Model',
    'custom_contr+starv': DATA_PATH / 'models/Combined_OA+Contractility_Model'
}

cellpose_models = {
    'cyto2': models.CellposeModel(gpu=True, model_type='cyto2')
}
cellpose_models.update({
    name: models.CellposeModel(gpu=True, pretrained_model=str(model_path))
    for name, model_path in model_path_from_name.items()
})

print(cellpose_models)

## Excuation
# Section A - All prerequisites
with np.load(DYE_SPECTRA_FP) as npz:
    print(npz['labels'])

DYE_LIST = ['LipiBlue', '342', 'BODIPY', 'pHrodo', 'TMRM', 'Lyso', 'Tubulin']  # for 7 color
n_comps = len(DYE_LIST)
pure_phasors = load_pure_phasors(pure_spectra_path=DYE_SPECTRA_FP,
                                 dye_list=DYE_LIST,
                                 ch_vec=CHANNEL_VEC,
                                 use_extra_coord=USE_EXTRA_COORD)

row_1_axis_titles = (f'Phasors; Harm={NTH_HARMONIC}', 'Cellpose Masks', 'Spectral RGB', 'Brightfield')
row_2_axis_titles = (*DYE_LIST,)  # add in 'Unmixed RGB' here

any_LSM880_path = '/Users/cristinazhu/PycharmProjects/PhasorUnmix_01162024edit/Data/Anchor/Control_01.lsm'
with tifffile.TiffFile(any_LSM880_path) as tif:
    ms_880 = tif.lsm_metadata
    channel_lambdas = np.asarray(ms_880['ChannelColors']['ColorNames'][:32], dtype=np.float32)

pure_phasor_g = pure_phasors[:, (NTH_HARMONIC - 1) * 2]
pure_phasor_s = -pure_phasors[:, (NTH_HARMONIC - 1) * 2 + 1]
