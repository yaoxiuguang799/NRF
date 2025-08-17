import os
import numpy as np
from tqdm import tqdm
import h5py
def readH5py(h5_file,readTime=True):
    with h5py.File(h5_file, "r") as hf:  # read data from h5 file
        data = hf["X"][:]
        lat = hf["latitude"][:]
        lon = hf["longitude"][:]
        if readTime:
            time = hf["time"][:]
        else:
            time = []
    return data,lat,lon,time

def writeH5py(h5_file,data,lat,lon,time=[]):
    with h5py.File(h5_file, "w") as hf:
        hf.create_dataset("X", data=np.array(data,dtype=np.float32))
        hf.create_dataset("latitude", data=np.array(lat,dtype=np.float32))
        hf.create_dataset("longitude", data=np.array(lon,dtype=np.float32))
        hf.create_dataset("time", data=time)

def search_window(irow, icol, nrow, ncol, w=3):
    """
    Return the row/column index range of a square window centered
    at (irow, icol), with clipping at the image borders.
    """
    h = w // 2

    drow = max(irow - h, 0)
    urow = min(irow + h + 1, nrow)

    lcol = max(icol - h, 0)
    rcol = min(icol + h + 1, ncol)

    # ensure window has size w (if clipped at boundary)
    if urow - drow < w:
        if drow == 0:
            urow = min(w, nrow)
        else:
            drow = max(urow - w, 0)

    if rcol - lcol < w:
        if lcol == 0:
            rcol = min(w, ncol)
        else:
            lcol = max(rcol - w, 0)

    return drow, urow, lcol, rcol


def guided_filter(p, I, eps):
    """Closed‐Form Guided Filter."""
    mean_I = np.mean(I)
    mean_p = np.mean(p)
    corr_I = np.mean(I * I)
    corr_Ip = np.mean(I * p)
    var_I = corr_I - mean_I**2
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return a, b


class GuidedFilterRefiner:
    """
    Refinement of avcm-pwv results using local guided filtering.

    Parameters
    ----------
    window : int      size of local window (in grid points)
    eps    : float    regularization parameter
    max_iter : int    maximum number of iterations per day
    """

    def __init__(self, window=3, eps=0.1, max_iter=10):
        self.window = window
        self.eps = eps
        self.max_iter = max_iter

    @staticmethod
    def _collect_valid_pairs(refine_w, avc_w, mask_w):
        idx = np.where((mask_w == 1) | (mask_w == 2) | (mask_w == 4))
        fus = refine_w[idx]
        avc = avc_w[idx]
        return fus, avc

    def _guided_iteration(self, avc_day, refine_day, mask_day):
        """
        Execute a single guided filtering iteration for all invalid pixels of one day.
        """
        nlat, nlon = avc_day.shape
        refine_new = refine_day.copy()
        mask_new = mask_day.copy()

        rows, cols = np.where((mask_day == 0) | (mask_day == 3))
        for r, c in zip(rows, cols):
            drow, urow, lcol, rcol = search_window(r, c, nlat, nlon, self.window)
            refine_w = refine_day[drow:urow, lcol:rcol]
            avc_w = avc_day[drow:urow, lcol:rcol]
            mask_w = mask_day[drow:urow, lcol:rcol]

            fus, avc = self._collect_valid_pairs(refine_w, avc_w, mask_w)
            if len(fus) < 2:
                # not enough samples -> use mean difference
                delta = np.mean(fus - avc) if len(fus) > 0 else 0
                refine_new[r, c] = avc_day[r, c] + delta
            else:
                dpwv = fus - avc
                a, b = guided_filter(dpwv, avc, self.eps)
                refine_new[r, c] = avc_day[r, c] + (a * avc_day[r, c] + b)

            mask_new[r, c] = 5  # mark as refined

        mask_new[mask_new == 5] = 4
        return refine_new, mask_new

    def _refine_single_image(self, avc_day, refine_day, mask_day):
        """
        Multi-iteration refinement on a single image.
        """
        for itr in range(self.max_iter):
            rows, _ = np.where((mask_day == 0) | (mask_day == 3))
            if len(rows) == 0:
                break

            refine_day, mask_day = self._guided_iteration(avc_day, refine_day, mask_day)

        # fallback  (still unfilled → directly copy avc value)
        rows, cols = np.where((mask_day == 0) | (mask_day == 3))
        if len(rows) > 0:
            refine_day[rows, cols] = avc_day[rows, cols]
            mask_day[rows, cols] = 4

        return refine_day, mask_day

    def refine(self, avc_pwvs, refine_pwvs, mask):
        """
        Perform the guided refinement over all days.

        Returns
        -------
        refine_pwvs   (filled PWV cube)
        mask          (updated mask)
        """
        nday = avc_pwvs.shape[0]
        for d in tqdm(range(nday), desc="Guided refinement"):
            refine_d, mask_d = self._refine_single_image(
                avc_pwvs[d], refine_pwvs[d], mask[d]
            )
            refine_pwvs[d] = refine_d
            mask[d] = mask_d
        return refine_pwvs, mask

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # ---------- INPUT PATHS ----------
    path_NBCM_PWV = './NBCM/nbcm_corrected.h5'
    path_AVCM_PWV = './AVCM/avcm_fused.h5'

    # ---------- OUTPUT PATHS ----------
    
    path_NRF_PWV = './NRF/NRF_PWV.h5'

    pwvs_nbcm,lat,lon,time = readH5py(path_NBCM_PWV)
    pwvs_avc,lat,lon,time = readH5py(path_AVCM_PWV)

    mask = (~np.isnan(pwvs_nbcm)).astype(int)

    refiner = GuidedFilterRefiner(window=3, eps=0.1, max_iter=10)
    refine_pwvs, mask = refiner.refine(pwvs_avc, pwvs_nbcm, mask)


    writeH5py(path_NRF_PWV, refine_pwvs, lat, lon, time)
