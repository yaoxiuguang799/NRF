import os 
import numpy as np
from tqdm import tqdm
from pykrige.ok import OrdinaryKriging
import numpy as np
import xarray as xr
import h5py
from numpy import matrix as mat
from matplotlib import pyplot as plt
from Levenberg_Marquardt import avcModel, avceModel, Levenberg_Marquardt

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


class ERA5Interpolator:
    """
    Interpolate ERA5 data to a high-resolution grid.
    Support: cubic / linear / nearest / kriging / auto (automatic selection).
    """

    def __init__(self, target_lat_range, target_lon_range, resolution=0.01):
        self.lat_min, self.lat_max = target_lat_range
        self.lon_min, self.lon_max = target_lon_range
        self.resolution = resolution

        # Build grid
        self.target_lats = np.arange(self.lat_min, self.lat_max + resolution, resolution)
        self.target_lons = np.arange(self.lon_min, self.lon_max + resolution, resolution)

        # Precompute target grid size
        self.grid_size = len(self.target_lats) * len(self.target_lons)

    def _interp_xarray(self, da, method):
        """Xarray-based interpolation."""
        target_coords = {
            "latitude": xr.DataArray(self.target_lats, dims="latitude"),
            "longitude": xr.DataArray(self.target_lons, dims="longitude"),
        }
        return da.interp(
            latitude=target_coords["latitude"],
            longitude=target_coords["longitude"],
            method=method
        )

    def _interp_kriging(self, da):
        """Kriging interpolation (PyKrige)."""
        lats = da.latitude.values
        lons = da.longitude.values
        values = da.values

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        x = lon_grid.ravel()
        y = lat_grid.ravel()
        z = values.ravel()

        OK = OrdinaryKriging(
            x, y, z, variogram_model="gaussian", verbose=False
        )

        # krige to high-resolution grid
        z_interp, _ = OK.execute("grid", self.target_lons, self.target_lats)

        return xr.DataArray(
            z_interp,
            coords={"latitude": self.target_lats,
                    "longitude": self.target_lons},
            dims=("latitude", "longitude")
        )

    def interpolate(self, ds, var_name, method="auto", kriging_threshold=10000):
        """
        Parameters
        ----------
        ds : xr.Dataset
        var_name : str
        method : 'cubic' | 'linear' | 'nearest' | 'kriging' | 'auto'
        kriging_threshold : int
            Number of target grid points below which kriging is automatically used.
        """
        da = ds[var_name]

        # 1) auto mode: choose method based on region size
        if method == "auto":
            # Use kriging for small regions
            method = "kriging" if self.grid_size <= kriging_threshold else "cubic"

        # 2) call corresponding method
        if method == "kriging":
            return self._interp_kriging(da)
        elif method in ("cubic", "linear", "nearest"):
            return self._interp_xarray(da, method)
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")



class AVCGapFiller:
    """
    Fill MOD PWV gaps using ERA5-derived avc/avce model.

    Parameters
    ----------
    filter_file : str   -> Filtered MOD PWV file (.h5)
    filter_mask_file : str -> Filter mask file (.h5)
    era5_interp_file : str -> ERA5 interpolated PWV file (.h5)
    avc_pwv_out : str -> Output file for avc PWV (.h5)
    avc_model_out : str -> Output file for avc/avce model parameters (.h5)
    plot_mod : bool -> Whether to export sample plots
    """

    def __init__(self, nbcm_file, era5_interp_file,
                 avcm_out_file, avcm_model_file, plot_mod=True):

        self.filter_file = nbcm_file
        self.era5_interp_file = era5_interp_file
        self.avcm_out_file = avcm_out_file
        self.avcm_model_file = avcm_model_file
        self.plot_mod = plot_mod

        # output plot folder
        self.mod_plot_path = '.\\AVCM\\Plot'
        os.makedirs(self.mod_plot_path, exist_ok=True)

        # read files
        self.pwvs_nbcm, self.lat, self.lon, self.time = readH5py(self.filter_file)
        self.pwvs_era5_interp, _, _, _ = readH5py(self.era5_interp_file)

        # day-of-year vector
        nday = self.pwvs_nbcm.shape[0]
        self.doys = np.arange(1, nday + 1).reshape(-1, 1)

    @staticmethod
    def _conditional_judgment(t, n_in_sub=3, n_subs=10, n_all=20):
        """Check if there are enough samples in different seasonal periods."""
        f_n = (t <= 90).sum()
        m_n = ((t > 90) & (t <= 235)).sum()
        b_n = (t > 235).sum()
        return (f_n >= n_in_sub) + (m_n >= n_in_sub) + (b_n >= n_in_sub) >= n_subs and len(t) >= n_all

    def _plot_result(self, ilat, ilon, t_all, ipwv_mod, pwv_prd_era5, pwv_avc, pwv_avce):
        fig_file = os.path.join(self.mod_plot_path, f"row{ilat}_col{ilon}.png")
        plt.figure(figsize=(10, 6))
        plt.title('Optimization Results')
        plt.scatter(t_all, ipwv_mod, s=30, label='mod-pwv')
        plt.plot(t_all, pwv_prd_era5, label='avc-ERA5')
        plt.plot(t_all, pwv_avc, label='avc-mod')
        plt.plot(t_all, pwv_avce, label='avce-mod')
        plt.legend()
        plt.savefig(fig_file)
        plt.close()

    def _fit_pixel(self, ipwv_era5, ipwv_mod, ilat, ilon, avc_params):
        """
        Fit avc/avce model for a single pixel.
        """
        t_all = self.doys

        # --- ERA5 avc ---
        params0 = mat([[8.0], [2.0], [12.0]])
        result = Levenberg_Marquardt(t_all, ipwv_era5, params0, avcModel)
        params_era5 = np.array(result[-1]).flatten()
        avc_params[:3, ilat, ilon] = params_era5

        # --- MOD avce ---
        idx_valid = np.where(~np.isnan(ipwv_mod))[0]
        if idx_valid.size == 0:
            return avc_params[:, ilat, ilon], np.full_like(ipwv_mod, np.nan), None, None

        t_valid = t_all[idx_valid]
        dpwv = ipwv_era5[idx_valid] - avcModel(params_era5, t_valid)
        x_valid = np.column_stack((t_valid, dpwv))
        y_valid = ipwv_mod[idx_valid].reshape(-1, 1)

        init_params = mat([[params_era5[0]], [params_era5[1]], [params_era5[2]], [1.0]])
        if self._conditional_judgment(t_valid):
            result = Levenberg_Marquardt(x_valid, y_valid, init_params, avceModel)
            params_mod = np.array(result[-1]).flatten()
        else:
            params_mod = np.array(init_params).flatten()

        avc_params[3:7, ilat, ilon] = params_mod

        # predictions
        pwv_prd_era5 = avcModel(params_era5, t_all).flatten()
        dpwv_all = ipwv_era5.flatten() - pwv_prd_era5
        x_all = np.column_stack((t_all, dpwv_all))
        pwv_avce = avceModel(params_mod, x_all).flatten()
        pwv_avc = avcModel(params_mod[:3], t_all).flatten()

        return avc_params[:, ilat, ilon], pwv_avce, pwv_prd_era5, pwv_avc

    def run(self):
        """
        Run avc/avce modelling and export the results.
        """
        n_day, n_lat, n_lon = self.pwvs_nbcm.shape
        avc_pwvs = np.full((n_day, n_lat, n_lon), np.nan)
        avc_params = np.full((7, n_lat, n_lon), np.nan)

        for ilat in tqdm(range(n_lat), desc='lat'):
            for ilon in range(n_lon):
                pwv_era5 = self.pwvs_era5_interp[:, ilat, ilon].reshape(-1, 1)
                pwv_mod = self.pwvs_nbcm[:, ilat, ilon]

                params, pwv_avce, pwv_era5_pred, pwv_avc = self._fit_pixel(
                    pwv_era5, pwv_mod, ilat, ilon, avc_params
                )
                avc_params[:, ilat, ilon] = params
                avc_pwvs[:, ilat, ilon] = pwv_avce

                if (self.plot_mod and ilat % 50 == 0 and ilon % 50 == 0
                    and pwv_era5_pred is not None):
                    self._plot_result(ilat, ilon, self.doys, pwv_mod,
                                      pwv_era5_pred, pwv_avc, pwv_avce)

        writeH5py(self.avcm_out_file, avc_pwvs, self.lat, self.lon, self.time)
        writeH5py(self.avcm_model_file, avc_params, self.lat, self.lon)

        print('all is over!')


# =============================================================================
# Example use  (you can directly run the file)
# =============================================================================
if __name__ == "__main__":
    # ---------- INPUT PATHS ----------
    path_NBCM_PWV = './NBCM/nbcm_corrected.h5'
    path_ERA5 = './data/ERA5/'

    # ---------- OUTPUT PATHS ----------
    path_ERA5_interp = './data/ERA5/'
    path_AVCM_PWV = './AVCM/avcm_fused.h5'
    path_AVCM_modle = './AVCM/avcm_model.h5'

    
    ds = xr.open_dataset(path_ERA5)
    # Load NBCM data
    with h5py.File(path_NBCM_PWV, "r") as f:
        lat = f["latitude"][:]     # MOD latitudes
        lon = f["longitude"][:]    # MOD longitudes

    maxLat = np.max(lat); minLat = np.min(lat)
    maxLon = np.max(lon); minLon = np.min(lon)

    # ----------------- step 1 : interpolate ERA5 to 0.01°×0.01° ------------- #
    # 创建插值器
    interpolator = ERA5Interpolator(target_lat_range=(minLat, maxLat),
                                    target_lon_range=(minLon, maxLon),
                                    resolution=0.01)
    # 自动选择方法（小区域 → kriging，大区域 → cubic）
    ERA5_interp = interpolator.interpolate(ds, "tcwv", method="auto")
    writeH5py(path_ERA5_interp, ERA5_interp, lat, lon)

    # ---------------- step 2 ： Build ACVM to gapfill MODIS PWV --------------- #
    fusion = AVCGapFiller(
    filter_file=path_NBCM_PWV,
    era5_interp_file=path_ERA5_interp,
    avc_pwv_out=path_AVCM_PWV,
    avc_model_out=path_AVCM_modle,
    plot_mod=True
    )
    fusion.run()