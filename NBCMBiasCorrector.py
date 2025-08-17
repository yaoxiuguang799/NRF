import os
import numpy as np
import xarray as xr
import h5py
from osgeo import gdal
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def outlier_with_sigma(data,thod=3):
    # 
    mean = np.mean(data)
    std_dev = np.std(data)
    # 
    threshold_upper = mean + thod * std_dev
    threshold_lower = mean - thod * std_dev
    # 
    ind = np.where((data <= threshold_lower) | (data >= threshold_upper) )[0]
    return ind

def accuracyEvaluate(y_true,y_pred,removeOutlier=False):
    R,BIAS,MAE,MRE,RMSE,KGE = 0.0,0.0,0.0,0.0,0.0,0.0
    y_true_collector = np.copy(y_true)
    y_pred_collector = np.copy(y_pred)

    ### 1. remove nan
    idx = np.where(np.isnan(y_true_collector))
    y_true_collector = np.delete(y_true_collector,idx,axis=0)
    y_pred_collector = np.delete(y_pred_collector,idx,axis=0)
    idx = np.where(np.isnan(y_pred_collector))
    y_true_collector = np.delete(y_true_collector,idx,axis=0)
    y_pred_collector = np.delete(y_pred_collector,idx,axis=0)
    if len(y_pred_collector) <= 1:
        return R,BIAS,MAE,MRE,RMSE,KGE
    
    ### 2. remove outliers
    dy = y_pred_collector - y_true_collector
    if removeOutlier:
        idx = outlier_with_sigma(dy)
        # idx = out_range_method(dy)
        dy_smooth = np.delete(dy,idx,axis=0)
        y_true_collector = np.delete(y_true_collector,idx,axis=0)
        y_pred_collector = np.delete(y_pred_collector,idx,axis=0)
    else:
        dy_smooth = dy

    ### 3.1 Pearson Correlation Coefficient
    R = pearsonr(y_true_collector, y_pred_collector)[0]

    ### 3.2 bias
    BIAS = np.mean(dy_smooth)

    ### 3.3 mae
    MAE = np.sum(np.abs(dy_smooth))/len(dy_smooth)  

    ### 3.4 mre
    MRE = np.sum(np.abs(dy_smooth))/np.sum(np.abs(y_true_collector)) 

    ### 3.5 rmse
    RMSE = np.sqrt(mean_squared_error(y_true_collector, y_pred_collector))

    ### 3.6 KGE
    beta = np.mean(y_pred_collector)/np.mean(y_true_collector)
    gamma = np.std(y_pred_collector)/np.std(y_true_collector)
    KGE = 1 - np.sqrt((R-1)*(R-1) + (beta-1)*(beta-1) + (gamma-1)*(gamma-1))

    return R,BIAS,MAE,MRE,RMSE,KGE

class NBCMBiasCorrector:
    """
    Bias correction class for MOD-based PWV values using ERA5 data, DEM,
    NDVI and Random Forest modeling.

    Steps performed by this class:
      1) Construct a training dataset by matching MOD PWV to ERA5 PWV
         at nearest grid cells and combining DEM & NDVI features.
      2) Train a random-forest model with hyper-parameter grid search.
      3) Apply the trained model to the full MOD PWV time-series to obtain
         bias-corrected values.
    """

    def __init__(
        self,
        mod_file,
        era5_file,
        mask_file,
        ndvi_file,
        dem_file,
        output_dir,
        param_grid=None,
    ):
        """
        Parameters
        ----------
        mod_file : str
            Path to the MOD PWV HDF5 file.
        era5_file : str
            Path to ERA5 PWV NetCDF file.
        mask_file : str
            Path to land/sea mask (1 = valid, 0 = invalid).
        ndvi_file : str
            NDVI time-series (HDF5 format).
        dem_file : str
            DEM raster file (GeoTIFF).
        output_dir : str
            Directory where intermediate and output files will be written.
        param_grid : dict, optional
            Hyper-parameter grid for RandomForestRegressor. If not provided,
            a default grid is used.
        """

        self.mod_file = mod_file
        self.era5_file = era5_file
        self.mask_file = mask_file
        self.ndvi_file = ndvi_file
        self.dem_file = dem_file
        self.output_dir = output_dir

        # Make sure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # --- class member paths (so user doesn’t need to pass them explicitly) ---
        self.dataset_path = os.path.join(self.output_dir, "dataset.csv")
        self.model_path = os.path.join(self.output_dir, "nbcm_model.pkl")
        self.corrected_path = os.path.join(self.output_dir, "nbcm_corrected.h5")

        # --- parameter grid for RandomForest hyper-parameter tuning ---
        self.param_grid = param_grid or {
            "n_estimators": [50, 100, 200, 400, 600],
            "max_depth": [10, 20, 30, 50, 100],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

    # ---------------------------------------------------------------------
    # Utility functions
    # ---------------------------------------------------------------------
    @staticmethod
    def date_to_doy(date_str: str) -> int:
        """
        Convert a YYYY-MM-DD string to day-of-year (1…365).
        """
        return datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday

    @staticmethod
    def read_dem(dem_file):
        """
        Open a DEM file using GDAL. Returns an opened dataset.
        """
        ds = gdal.Open(dem_file)
        if ds is None:
            raise RuntimeError(f"Unable to open DEM file: {dem_file}")
        return ds

    @staticmethod
    def interpolate_dem(dem_ds, x, y):
        """
        Bilateral / nearest-pixel extraction of DEM height for a given lon/lat.
        """
        gt = dem_ds.GetGeoTransform()
        px = int((x - gt[0]) / gt[1])
        py = int((y - gt[3]) / gt[5])
        if px < 0 or py < 0:
            return -9999
        return float(dem_ds.ReadAsArray(px, py, 1, 1)[0, 0])

    @staticmethod
    def find_nearest_idx(arr, val):
        """
        Returns the index of `arr` that is closest to the value `val`.
        (Used to map ERA5 grid to MOD grid.)
        """
        return int(np.abs(arr - val).argmin())

    # ---------------------------------------------------------------------
    # Dataset building
    # ---------------------------------------------------------------------
    def prepare_dataset(self):
        """
        Build a training dataset by aligning ERA5 PWV with MOD PWV,
        NDVI and DEM information. The dataset is saved as CSV and
        contains records in the form:
            [lat, lon, elevation, DOY, ndvi, mod_pwv, era5_pwv].
        """
        # Load MOD data
        with h5py.File(self.mod_file, "r") as f:
            mod_pwvs = f["X"][:]            # PWV array [time, lat, lon] on MOD grid
            mod_lat = f["latitude"][:]     # MOD latitudes
            mod_lon = f["longitude"][:]    # MOD longitudes

        # Load NDVI and mask
        ndvis = h5py.File(self.ndvi_file, "r")["X"][:]
        mask  = h5py.File(self.mask_file, "r")["X"][:]

        # Load ERA5 PWV
        da = xr.open_dataarray(self.era5_file)
        era5_pwvs = da.values
        era_lat = da["latitude"].values
        era_lon = da["longitude"].values

        # Load DEM file and build index mapping between ERA5 and MOD grid
        dem_ds = self.read_dem(self.dem_file)
        lat_idx = [self.find_nearest_idx(mod_lat, b) for b in era_lat]
        lon_idx = [self.find_nearest_idx(mod_lon, l) for l in era_lon]

        dataset_list = []
        for ilat, lat_val in enumerate(tqdm(era_lat, desc="Preparing dataset")):
            indLat = lat_idx[ilat]
            for ilon, lon_val in enumerate(era_lon):
                # Mask = 0 indicates water or invalid area → skip
                if mask[ilat, ilon] == 0:
                    continue

                # Interpolate DEM height for the current location
                dem_h = self.interpolate_dem(dem_ds, lon_val, lat_val)
                if dem_h < 0:
                    continue

                indLon = lon_idx[ilon]

                e_series = era5_pwvs[:, ilat, ilon]
                m_series = mod_pwvs[:, indLat, indLon]
                ndvi_series = ndvis[:, indLat, indLon]

                for i, (e_val, m_val) in enumerate(zip(e_series, m_series)):
                    # Skip invalid or negative PWV values
                    if (
                        np.isnan(e_val) or
                        np.isnan(m_val) or
                        e_val < 0 or m_val < 0 or
                        np.isnan(ndvi_series[i])
                    ):
                        continue

                    dataset_list.append(
                        [lat_val, lon_val, dem_h, i + 1,
                         ndvi_series[i], m_val, e_val]
                    )

        dataset_arr = np.asarray(dataset_list, dtype=np.float32)
        np.savetxt(self.dataset_path, dataset_arr, delimiter=",", fmt="%.5f")

    # ---------------------------------------------------------------------
    @staticmethod
    def preprocess(dataset_path):
        """
        Load the CSV dataset, remove outliers using sigma-clipping,
        and split it into training and testing sets.
        """
        dataset = np.genfromtxt(dataset_path, delimiter=",")
        idx_delete = outlier_with_sigma(dataset[:, 6] - dataset[:, 5])
        dataset = np.delete(dataset, idx_delete, axis=0)
        X = dataset[:, :6]
        y = dataset[:, 6]
        return train_test_split(X, y, train_size=0.8, random_state=42)

    # ---------------------------------------------------------------------
    def train_model(self, X_train, X_test, y_train, y_test):
        """
        Train a RandomForest regressor with cross-validated grid search,
        save the best fitted model and print performance metrics.
        """
        rf = RandomForestRegressor(random_state=42)
        gs = GridSearchCV(
            rf,
            param_grid=self.param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=2,
        )
        gs.fit(X_train, y_train)
        joblib.dump(gs.best_estimator_, self.model_path)

        for split_name, X_split, y_split in [("train", X_train, y_train),
                                             ("test", X_test, y_test)]:
            preds = gs.predict(X_split)
            R, BIAS, MAE, MRE, RMSE, KGE = accuracyEvaluate(y_split, preds)
            print(f"{split_name}  R={R:.3f}  Bias={BIAS:.2f}  RMSE={RMSE:.2f}")

    # ---------------------------------------------------------------------
    def _predict_single_day(self, model, ipwv, ndvi, doy, dem_v, lat_v, lon_v):
        """
        Apply the trained model for one specific day and return the predicted PWV grid.
        """
        mask = ~np.isnan(ipwv)
        if not mask.any():
            return ipwv

        X = np.column_stack((
            lat_v[mask],
            lon_v[mask],
            dem_v[mask],
            np.full(np.sum(mask), doy + 1),
            ndvi[mask],
            ipwv[mask],
        ))
        ipwv[mask] = model.predict(X)
        return ipwv

    # ---------------------------------------------------------------------
    def bias_correction(self):
        """
        Load the trained model and apply it to the full MOD time-series,
        then save the bias-corrected PWV grid as an HDF5 file.
        """
        # --- Load MOD PWV and grids ---
        with h5py.File(self.mod_file, "r") as f:
            mod_pwvs = f["X"][:]
            lat = f["latitude"][:]
            lon = f["longitude"][:]
            time = f["time"][:]

        # NDVI series and DEM
        ndvis = h5py.File(self.ndvi_file, "r")["X"][:]
        dem = self.read_dem(self.dem_file).ReadAsArray()[:1300, :3100][::-1, :]

        # Create flat lat/lon coordinate vectors
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
        dem_v = dem.flatten()
        lat_v = lat_grid.flatten()
        lon_v = lon_grid.flatten()

        # Load trained model
        model = joblib.load(self.model_path)

        # Apply the model day-by-day
        for doy in tqdm(range(mod_pwvs.shape[0]), desc="Bias correction"):
            mod_pwvs[doy] = self._predict_single_day(
                model,
                mod_pwvs[doy],
                ndvis[doy],
                doy,
                dem_v,
                lat_v,
                lon_v,
            )

        # Save corrected dataset
        with h5py.File(self.corrected_path, "w") as hf:
            hf.create_dataset("X", data=mod_pwvs)
            hf.create_dataset("latitude", data=lat)
            hf.create_dataset("longitude", data=lon)
            hf.create_dataset("time", data=time)

    # ---------------------------------------------------------------------
    def run(self):
        """
        Full workflow entry: dataset preparation → training → bias correction.
        """
        print("Step 1: Preparing dataset...")
        self.prepare_dataset()

        print("Step 2: Splitting & preprocessing...")
        X_train, X_test, y_train, y_test = self.preprocess(self.dataset_path)

        print("Step 3: Training model...")
        self.train_model(X_train, X_test, y_train, y_test)

        print("Step 4: Applying bias correction...")
        self.bias_correction()

        print("All finished. Output file:", self.corrected_path)
        return self.corrected_path

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # ---------- INPUT PATHS ----------
    path_MODIS = './data/MODIS'
    path_ERA5 = './data/ERA5'
    path_land_mask = './data/land_mask'
    path_NDVI = './data/NDVI'
    path_DEM = './data/DEM'

    # ---------- OUTPUT PATHS ----------
    output_dir = './NBCM'

    nbcm = NBCMBiasCorrector(
        mod_file=path_MODIS,
        era5_file=path_ERA5,
        mask_file=path_land_mask,
        ndvi_file=path_NDVI,
        dem_file=path_DEM,
        output_dir="./output_nbcm",
        # Example: custom hyper-parameter grid
        # param_grid={"n_estimators":[300],"max_depth":[25]}
    )

    nbcm.run()


