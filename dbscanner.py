import numpy as np
from scipy import sparse
from ..base import BaseEstimator, ClusterMixin
from ..utils import check_array, check_consistent_length
from ..neighbors import NearestNeighbors
from ._dbscan_inner import dbscan_inner
def dbscan(X, eps=0.5, min_samples=5, metric='minkowski', metric_params=None,
           algorithm='auto', leaf_size=30, p=2, sample_weight=None, n_jobs=1):
    if not eps > 0.0:
        raise ValueError("eps must be positive.")
    X = check_array(X, accept_sparse='csr')
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        check_consistent_length(X, sample_weight)
    # Calculate neighborhood for all samples. This leaves the original point
    # in, which needs to be considered later (i.e. point i is in the
    # neighborhood of point i. While True, its useless information)
    if metric == 'precomputed' and sparse.issparse(X):
        neighborhoods = np.empty(X.shape[0], dtype=object)
        X.sum_duplicates()  # XXX: modifies X's internals in-place
        X_mask = X.data <= eps
        masked_indices = X.indices.astype(np.intp, copy=False)[X_mask]
        masked_indptr = np.concatenate(([0], np.cumsum(X_mask)))[X.indptr[1:]]
        # insert the diagonal: a point is its own neighbor, but 0 distance
        # means absence from sparse matrix data
        masked_indices = np.insert(masked_indices, masked_indptr,
                                   np.arange(X.shape[0]))
        masked_indptr = masked_indptr[:-1] + np.arange(1, X.shape[0])
        # split into rows
        neighborhoods[:] = np.split(masked_indices, masked_indptr)
    else:
        neighbors_model = NearestNeighbors(radius=eps, algorithm=algorithm,
                                           leaf_size=leaf_size,
                                           metric=metric,
                                           metric_params=metric_params, p=p,
                                           n_jobs=n_jobs)
        neighbors_model.fit(X)
        # This has worst case O(n^2) memory complexity
        neighborhoods = neighbors_model.radius_neighbors(X, eps,
                                                         return_distance=False)
    if sample_weight is None:
        n_neighbors = np.array([len(neighbors)
                                for neighbors in neighborhoods])
    else:
        n_neighbors = np.array([np.sum(sample_weight[neighbors])
                                for neighbors in neighborhoods])
    # Initially, all samples are noise.
    labels = -np.ones(X.shape[0], dtype=np.intp)
    # A list of all core samples found.
    core_samples = np.asarray(n_neighbors >= min_samples, dtype=np.uint8)
    dbscan_inner(core_samples, neighborhoods, labels)
    return np.where(core_samples)[0], labels
    
class DBSCAN(BaseEstimator, ClusterMixin):
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean',
                 metric_params=None, algorithm='auto', leaf_size=30, p=None,
                 n_jobs=1):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs
    def fit(self, X, y=None, sample_weight=None):
        X = check_array(X, accept_sparse='csr')
        clust = dbscan(X, sample_weight=sample_weight,
                       **self.get_params())
        self.core_sample_indices_, self.labels_ = clust
        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))
        return self
    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X, sample_weight=sample_weight)
        return self.labels_