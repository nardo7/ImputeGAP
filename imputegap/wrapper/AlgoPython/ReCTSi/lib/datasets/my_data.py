# ===============================================================================================================
# SOURCE: https://github.com/Graph-Machine-Learning-Group/grin
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://openreview.net/pdf?id=kOu3-S3wJ7
# ===============================================================================================================


import numpy as np
import pandas as pd

from ..utils import sample_mask
from .pd_dataset import PandasDataset


class MyData(PandasDataset):
    def __init__(self, data, mask_tr, periodicity):
        self.periodicity = periodicity
        df, dist, mask = self.load(data=data, mask_tr=mask_tr, periodicity=periodicity)
        self.dist = dist
        self.len, self.ncol = df.shape
        super().__init__(
            dataframe=df, u=None, mask=mask, name="bay", freq=None, aggr="nearest"
        )

    def load(self, data=None, mask_tr=None, periodicity=None):
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)

        if periodicity is None:
            raise ValueError("Periodicity must be provided.")

        # Compute tod and dow and concatenate them to the dataframe
        tod = np.array(
            [(i % periodicity) / periodicity for i in range(df.shape[0])]
        ).reshape(-1, 1)  # normalize day's steps to [0, 1]

        dow = np.array(
            [(i % (periodicity * 7)) // periodicity for i in range(df.shape[0])]
        ).reshape(-1, 1)
        tod_tiled = np.tile(tod, (1, df.shape[1]))
        dow_tiled = np.tile(dow, (1, df.shape[1]))
        tod_pd = pd.DataFrame(tod_tiled, index=df.index)
        dow_pd = pd.DataFrame(dow_tiled, index=df.index)
        df = pd.concat([df, tod_pd, dow_pd], axis=1)

        mask = mask_tr
        dist = np.ones(shape=(df.shape[1], df.shape[1]))
        return df.astype("float32"), dist, mask.astype(bool)

    def get_similarity(
        self, type="dcrnn", thr=0.1, force_symmetric=False, sparse=False
    ):
        """
        Return similarity matrix among nodes. Implemented to match DCRNN.

        :param type: type of similarity matrix.
        :param thr: threshold to increase saprseness.
        :param trainlen: number of steps that can be used for computing the similarity.
        :param force_symmetric: force the result to be simmetric.
        :return: and NxN array representig similarity among nodes.
        """
        if type == "dcrnn":
            finite_dist = self.dist.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            sigma = finite_dist.std()

            if sigma == 0 or np.isnan(sigma):
                sigma = 1e-8  # Small constant to avoid division by zero

            adj = np.exp(-np.square(self.dist / sigma))
        elif type == "stcn":
            sigma = 10
            adj = np.exp(-np.square(self.dist) / sigma)
        else:
            raise NotImplementedError
        adj[adj < thr] = 0.0
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps

            adj = sps.coo_matrix(adj)
        return adj

    @property
    def mask(self):
        if self._mask is None:
            return self.df.values != 0.0
        return self._mask


class MissingValuesMyData(MyData):
    def __init__(
        self, data, tr_mask, periodicity, seed=42, p_fault=0.0015, p_noise=0.05
    ):
        super(MissingValuesMyData, self).__init__(data, tr_mask, periodicity)
        self.rng = np.random.default_rng(seed)
        self.p_fault = p_fault
        self.p_noise = p_noise
        eval_mask = sample_mask(
            self.numpy().shape,
            p=p_fault,
            p_noise=p_noise,
            min_seq=12,
            max_seq=12 * 4,
            rng=self.rng,
        )
        # self.eval_mask = (eval_mask & self.mask).astype("uint8")
        self.eval_mask = tr_mask.astype(bool)
        self._mask = tr_mask.astype(bool)
        self.tr_mask = tr_mask.astype(bool)

    @property
    def training_mask(self):
        return self.tr_mask
        # return (self.mask if self.eval_mask is None else (self.mask & (1 - self.eval_mask)))

    def splitter(self, dataset, train_len=0.9, window=24):
        idx = np.arange(len(dataset))
        train_size = int(len(dataset) * train_len)
        train_idxs = idx[: train_size - window]
        test_idxs = idx[train_size:]

        # presumably val_idxs is used as the whole input data for validation in imputeGAP
        return train_idxs, idx, test_idxs
