from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()
print(f"\nImputeGAP datasets : {ts.datasets}")

# load and normalize the dataset from file or from the code
ts.load_series(utils.search_path("airq"))
# ts.normalize(normalizer="z_score")

# print and plot a subset of time series
ts.print(nbr_series=6, nbr_val=20)
# ts.plot(input_data=ts.data, nbr_series=6, nbr_val=100, save_path="./imputegap_assets")


# contaminate the time series with MCAR pattern
ts_m = ts.Contamination.mcar(
    ts.data, rate_dataset=0.2, rate_series=0.4, block_size=10, seed=True
)
# ts.plot(ts.data, ts_m, nbr_series=9, subplot=True, save_path="./imputegap_assets/contamination")


# imputer = Imputation.DeepLearning.GRIN(ts_m)
# imputer.impute()


imputer = Imputation.DeepLearning.RECTSI(ts_m, periodicity=24)
imputer.impute()
imputer.score(ts_m, imputer.recov_data)
