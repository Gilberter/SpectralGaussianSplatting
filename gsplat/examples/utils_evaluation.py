import os
import numpy as np
import cv2
import argparse

# XYZ from chromaticity helper
def xyz_from_xy(x,y):
    return np.array((x,y,1-x-y))

ILUMINANT = {
    'D65': xyz_from_xy(0.3127, 0.3291),
    'E':  xyz_from_xy(1/3, 1/3),
}

COLOR_SPACE = {
    'sRGB': (xyz_from_xy(0.64, 0.33),
             xyz_from_xy(0.30, 0.60),
             xyz_from_xy(0.15, 0.06),
             ILUMINANT['D65']),

    'AdobeRGB': (xyz_from_xy(0.64, 0.33),
                 xyz_from_xy(0.21, 0.71),
                 xyz_from_xy(0.15, 0.06),
                 ILUMINANT['D65']),

    'AppleRGB': (xyz_from_xy(0.625, 0.34),
                 xyz_from_xy(0.28, 0.595),
                 xyz_from_xy(0.155, 0.07),
                 ILUMINANT['D65']),

    'UHDTV': (xyz_from_xy(0.708, 0.292),
              xyz_from_xy(0.170, 0.797),
              xyz_from_xy(0.131, 0.046),
              ILUMINANT['D65']),

    'CIERGB': (xyz_from_xy(0.7347, 0.2653),
               xyz_from_xy(0.2738, 0.7174),
               xyz_from_xy(0.1666, 0.0089),
               ILUMINANT['E']),
}


# CIE Analytical CMF (Color Matching Function)

def piecewise_gaussian(x,mu,tau1,tau2):
    result = np.zeros_like(x) 
    left_mask = x < mu
    right_mask = x >= mu
    result[left_mask] = np.exp((-0.5*tau1**2)*((x[left_mask]-mu)**2))
    result[right_mask] = np.exp((-0.5*tau2**2)*((x[right_mask]-mu)**2))
    return result


def x_bar_1931(wavelength):
    return (1.056 * piecewise_gaussian(wavelength, 599.8, 0.0264, 0.0323) +
            0.362 * piecewise_gaussian(wavelength, 442.0, 0.0624, 0.0374) -
            0.065 * piecewise_gaussian(wavelength, 501.1, 0.0490, 0.0382))

def y_bar_1931(wavelength):
    return (0.821 * piecewise_gaussian(wavelength, 568.8, 0.0213, 0.0247) +
            0.286 * piecewise_gaussian(wavelength, 530.9, 0.0613, 0.0322))

def z_bar_1931(wavelength):
    return (1.217 * piecewise_gaussian(wavelength, 437.0, 0.0845, 0.0278) +
            0.681 * piecewise_gaussian(wavelength, 459.0, 0.0385, 0.0725)) 

def compute_cmf_1931(wavelength):
    x_bar = x_bar_1931(wavelength) # shape (N,)
    y_bar = y_bar_1931(wavelength)
    z_bar = z_bar_1931(wavelength)
    return np.stack([x_bar, y_bar, z_bar], axis=1) # shape (N,3) where N is the number of wavelengths (bands)



class ColourSystem:

    def __init__(self,start=450,end=650,bands=21, cspace='sRGB'):

        wavelength = np.linspace(start,end,bands)

        self.cmf = compute_cmf_1931(wavelength)

        self.red, self.green, self.blue, self.white = COLOR_SPACE[cspace]

        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T
        self.MI = np.linalg.inv(self.M)

        # White scaling array
        self.wscale = self.MI.dot(self.white)

        # xyz -> rgb transformation matrix
        self.A = self.MI / self.wscale[:, np.newaxis]

        # Cache the transform matrix
        self._transform = self._compute_transform()

    def _compute_transform(self):
            """spectrum (N, bands) -> rgb (N, 3), no cross-band normalization."""
            XYZ = self.cmf           # (bands, 3)
            RGB = XYZ @ self.A.T     # (bands, 3)
            # Normalize so white spectrum maps to (1,1,1)
            white_rgb = np.ones((1, XYZ.shape[0])) @ RGB  # sum over bands
            RGB = RGB / white_rgb    # scale each channel
            return RGB               # (bands, 3)

    def spec_to_rgb(self, spec):
        """Convert a spectrum to an rgb value."""
        return spec @ self._transform  # (N, 3)
    


# Module-level cache
_colour_system_cache = {}

def spectrum_to_rgb(spectrum, start=450, end=650, bands=21, apply_gamma=True):
    import torch

    assert spectrum.shape[-1] == bands, \
        f"Expected {bands} bands, got {spectrum.shape[-1]}"

    is_torch = isinstance(spectrum, torch.Tensor)
    device = spectrum.device if is_torch else None
    dtype = spectrum.dtype if is_torch else None

    spec_np = spectrum.detach().cpu().numpy() if is_torch else np.asarray(spectrum, dtype=np.float64)

    original_shape = spec_np.shape
    spec_2d = spec_np.reshape(-1, bands)

    # Use cached ColourSystem
    key = (start, end, bands)
    if key not in _colour_system_cache:
        _colour_system_cache[key] = ColourSystem(start=start, end=end, bands=bands)
    cs = _colour_system_cache[key]

    rgb_2d = cs.spec_to_rgb(spec_2d)
    rgb_2d = np.clip(rgb_2d, 0, 1)

    if apply_gamma:
        rgb_2d = np.where(
            rgb_2d < 0.0031308,
            12.92 * rgb_2d,
            1.055 * (rgb_2d ** (1.0 / 2.4)) - 0.055
        )

    rgb = rgb_2d.reshape(original_shape[:-1] + (3,))

    if is_torch:
        rgb = torch.from_numpy(rgb).to(device=device, dtype=dtype)

    return rgb