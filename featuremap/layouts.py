import numpy as np
import numba
import featuremap.distances as dist
from featuremap.utils import tau_rand_int
from tqdm.auto import tqdm


@numba.njit()
def clip_arr(val):
    """Standard clamping of values of array into a fixed range (in this case -4.0 to
    4.0) 

    Parameters
    ----------
    val: float
        The value to be clamped.

    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    for i in range(val.shape[0]):
        if val[i] > 4.0:
            val[i] = 4.0
        elif val[i] < -4.0:
            val[i] = -4.0
        else:
            continue
    return val

@numba.njit()
def clip(val):
    """Standard clamping of a value into a fixed range (in this case -4.0 to
    4.0)

    Parameters
    ----------
    val: float
        The value to be clamped.

    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val

@numba.njit()
def unit_vector(vector):
    """
    Returns
    -------
    The unit vector of the vector

    """
    return vector / np.linalg.norm(vector)

# @numba.njit()
def angle_between(v1, v2):
    """
    Returns
    -------
    The angle in radians between vectors v1 and v2

    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    cache=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.intp,
    },
)
def rdist(x, y):
    """Reduced Euclidean distance.

    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)

    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result

def vdiff(x, y):
    """    
    Vector difference.
    
    Parameters
    ----------
    x : array of shape (embedding_dim,)
    y : array of shape (embedding_dim,)
    
    Returns
    -------
    The vector difference between x and y 

    """
    result = y - x
    return result



def _optimize_layout_euclidean_single_epoch_grad(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    featuremap_flag,
    feat_phi_sum,
    feat_re_sum,
    feat_re_cov,
    feat_re_std,
    feat_re_mean,
    feat_lambda,
    feat_R,
    feat_VH_embedding,
    feat_mu,
    feat_mu_tot,
    
):  
    # print('epochs_per_sample.shape[0]' + str(epochs_per_sample.shape[0]))
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]
            
            # vec_diff = vdiff(current, other)
            vec_diff = other - current

            inner_product = np.dot(vec_diff, vec_diff)
            outer_product = np.outer(vec_diff, vec_diff)
            
            grad_d = np.zeros(dim, dtype=np.float32)
            
            #dim = 1

            if featuremap_flag:
                current_VH = feat_VH_embedding[j] # rotation matrix embedding;  
                other_VH = feat_VH_embedding[k]
                
                grad_cor_coeff = np.zeros(dim, dtype=np.float32)
                # TODO: focus on each d of dim
                for d in numba.prange(dim):
                    phi = 1.0 / (1.0 + a * pow(inner_product, b))
                    dphi_term = (
                        2 * a * b * pow(inner_product, b - 1) * vec_diff / 
                        (1.0 + a * pow(inner_product, b))
                    )

                    v_j = current_VH[d]
                    v_k = other_VH[d]
                    project_vec_j = np.dot(v_j, vec_diff)
                    project_vec_k = np.dot(v_k, vec_diff)

                    #TODO: check feat_phi_sum, feat_re_sum
                    # Have changed the order of j, k
                    q_jk = phi / feat_phi_sum[j]
                    q_kj = phi / feat_phi_sum[k]
                    
                   
                    # drj = q_jk * (
                    #     project_vec_j * (2 * v_j -  project_vec_j * dphi_term ) / np.exp(feat_re_sum[j,d]) + dphi_term
                    # )
                    # drk = q_kj * (
                    #     project_vec_k * (2 * v_k -  project_vec_k * dphi_term ) / np.exp(feat_re_sum[k,d]) + dphi_term
                    # )
                    drj = np.zeros(dim)
                    drk = np.zeros(dim)
                    for s in numba.prange(dim):
                        drj[s] = q_jk * (
                            project_vec_j * (2 * v_j[s] -  project_vec_j * dphi_term[s] ) / np.exp(feat_re_sum[j,d]) + dphi_term[s]
                        )
                        drk[s] = q_kj * (
                            project_vec_k * (2 * v_k[s] -  project_vec_k * dphi_term[s] ) / np.exp(feat_re_sum[k,d]) + dphi_term[s]
                        )
                    
                    # check feat_re_std: array shape (dim,)
                    re_std_sq = feat_re_std[d] * feat_re_std[d]
                    
         
                    weight_j = (
                        feat_R[j,d]
                        - feat_re_cov[d] * (feat_re_sum[j,d] - feat_re_mean[d]) / re_std_sq
                    )
                    weight_k = (
                        feat_R[k,d]
                        - feat_re_cov[d] * (feat_re_sum[k,d] - feat_re_mean[d]) / re_std_sq
                    )
                    for s in numba.prange(dim):
                        grad_cor_coeff[s] += (weight_j * drj[s] + weight_k * drk[s]) / feat_re_std[d]
    
                for s in numba.prange(dim):
                    grad_cor_coeff[s] = (
                        grad_cor_coeff[s]
                        * feat_lambda
                        * feat_mu_tot
                        / feat_mu[i]
                        / n_vertices
                    )

            # grad_coeff = np.zeros(dim, dtype=np.float32)
            if inner_product > 0.0:
                # gradient of log Q_jk 
                grad_coeff_term = (-2.0) * a * b * pow(inner_product, b - 1.0) 
                grad_coeff_term = grad_coeff_term / (a * pow(inner_product, b) + 1.0)
            else:
                grad_coeff_term = 0
                  
            # gradient w.r.t y_j; sampling edge (j,k), Z_jk = y_k - y_j 
            # grad_d = clip_arr(* vec_diff[d]) * vec_diff[d] * (-1.0)
            for d in numba.prange(dim):
                grad_d = clip(grad_coeff_term * vec_diff[d] * (-1.0))
                # grad_d = grad_coeff[d] * (-1.0) * vec_diff[d]

                if featuremap_flag:
                    # FIXME: grad_cor_coeff might be referenced before assignment
                    grad_d += clip(grad_cor_coeff[d] * (-1.0))
                
                # 
                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in numba.prange(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices

                other = tail_embedding[k]

                # vec_diff = vdiff(current, other)
                vec_diff = other - current
                inner_product = np.dot(vec_diff, vec_diff)
                
                if inner_product > 0.0:
                    grad_coeff_term = 2.0 * gamma * b
                    #divisor = np.repeat((0.001 + inner_product) * (a * pow(inner_product, b) + 1), dim)
                    grad_coeff_term = grad_coeff_term / ((0.001 + inner_product) * (a * pow(inner_product, b) + 1))
                elif j == k:
                    continue
                else:
                    grad_coeff_term = 0.0
                

                for d in numba.prange(dim):
                    if grad_coeff_term > 0.0:
                        grad_d = clip(grad_coeff_term * vec_diff[d] * (-1.0))
                    else:
                        grad_d = 4.0 
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )
            


# Compute the variance in each direction of embedding under rotation VH 
def _optimize_layout_euclidean_featuremap_epoch_init_grad(
    head_embedding,
    tail_embedding,
    head,
    tail,
    # random_state,
    # gauge_vh,
    VH_embedding,
    # rotation_angle,
    a,
    b,
    re_sum,
    phi_sum,
):
    """
    Compute the principal radius in the dim-dimensional embedding space.
    
    Parameter
    ---------
    re_sum: array of shape (n_vertices, dim) 
        The principal radius in the embedding space
    phi_sum: array of shape (n_vertices,)
        For node i, the sum of edge existing probability incident to this node
    """
    
    # VH_embedding.fill(0)
    # random_initial = random_state.randint(0, head_embedding.shape[0], 1).astype(np.int64)
    # vh_initial = gauge_vh[random_initial[0]][0]
    # for i in range(head_embedding.shape[0]):
    #     vh_vector = gauge_vh[i][0]
    #     # angle = angle_between(vh_initial, vh_vector)
    #     angle = random_state.random() * 3.14
    #     # angle = random_state.random() * 3.14
    #     # angle = random_state.normal(0,1)

    #     # rotation_angle[i] = angle
    #     vh_embedding = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    #     # vh_embedding = np.identity(2)
    #     VH_embedding[i] = vh_embedding
 
    re_sum.fill(0)
    phi_sum.fill(0)
    
    dim = head_embedding.shape[1]

    for i in numba.prange(head.size):
        j = head[i]
        k = tail[i]

        current = head_embedding[j]
        other = tail_embedding[k]
        
        # current_VH = VH[j][:,:dim] # projection matrix; keep first dim dimensions 
        # other_VH = VH[k][:,:dim]
        
        # Restrict dim to 2
        # theta_j = rotation_angle[j]
        # theta_k = rotation_angle[k]
        
        current_VH = VH_embedding[j] # array shape of (dim, dim)
        other_VH = VH_embedding[k]
        # current_VH = np.identity(2)
        # other_VH = np.identity(2)
        
        # vec_diff = vdiff(current, other)
        vec_diff = other - current
        # inner_product = np.dot(vec_diff, vec_diff)
        dist_squared = rdist(current, other)
        
        phi = 1.0 / (1.0 + a * pow(dist_squared, b))
        phi_sum[j] += phi
        phi_sum[k] += phi
        
        for d in numba.prange(dim):
            vec_proj_vh_j = np.dot(vec_diff, current_VH[d]) # project to d-th rotation direction
            vec_proj_vh_k = np.dot(vec_diff, other_VH[d])        
     
            re_sum[j,d] += phi * vec_proj_vh_j * vec_proj_vh_j
            re_sum[k,d] += phi * vec_proj_vh_k * vec_proj_vh_k
        
        # vec_proj_vh_j = np.dot(current_VH, vec_diff) # project to rotation direction
        # vec_proj_vh_k = np.dot(other_VH, vec_diff)     
        
        # re_sum[j] += phi * vec_proj_vh_j * vec_proj_vh_j
        # re_sum[k] += phi * vec_proj_vh_k * vec_proj_vh_k
         
    epsilon = 1e-8
    for i in numba.prange(re_sum.shape[0]):
        re_sum[i] = np.log(epsilon + (re_sum[i] / phi_sum[i]))
    
    # for i in range(re_sum.shape[0]):
    #     re_sum[i] = np.log(epsilon + np.sqrt((re_sum[i] / phi_sum[i])))
   
        



def optimize_layout_euclidean_anisotropic_projection(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    parallel=False,
    verbose=False,
    # featuremap=False,
    featuremap_kwds=None,
    tqdm_kwds=None,
    move_other=False,
):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).
    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.
    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.
    head: array of shape (n_1_simplices)
        The indices of the heads of 1-simplices with non-zero membership.
    tail: array of shape (n_1_simplices)
        The indices of the tails of 1-simplices with non-zero membership.
    n_epochs: int
        The number of training epochs to use in optimization.
    n_vertices: int
        The number of vertices (0-simplices) in the dataset.
    epochs_per_sample: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.
    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.
    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.
    parallel: bool (optional, default False)
        Whether to run the computation using numba parallel.
        Running in parallel is non-deterministic, and is not used
        if a random seed has been set, to ensure reproducibility.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    featuremap: bool (optional, default False)
        Whether to use the feature-augmented featuremap objective
    featuremap_kwds: dict (optional, default None)
        Auxiliary data for featuremap
    tqdm_kwds: dict (optional, default None)
        Keyword arguments for tqdm progress bar.
    move_other: bool (optional, default False)
        Whether to adjust tail_embedding alongside head_embedding
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = head_embedding.shape[1]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    optimize_fn = numba.njit(
           _optimize_layout_euclidean_single_epoch_grad, 
           fastmath=True, 
           parallel=parallel,
           nopython=True,
           cache=True
    )
    # optimize_fn = _optimize_layout_euclidean_single_epoch_grad
    
    if featuremap_kwds is None:
        featuremap_kwds = {}
    if tqdm_kwds is None:
        tqdm_kwds = {}
    
    # if featuremap:
    feat_init_fn = numba.njit(
         _optimize_layout_euclidean_featuremap_epoch_init_grad,
         fastmath=True,
         parallel=parallel,
         nopython=True,
         cache=True
    )
    
    
    # feat_init_fn = _optimize_layout_euclidean_featuremap_epoch_init_grad

    feat_mu_tot = np.sum(featuremap_kwds["mu_sum"]) / 2  # sum of all edges' existing probability, float, shape of (1,)
    # should we modify lambda? Yes, we should
    #TODO: modify lambda
    feat_lambda = featuremap_kwds["lambda"] 
    feat_R = featuremap_kwds["R"] # array shape of (n_vertices d)
    feat_VH = featuremap_kwds["VH"]
    feat_VH_embedding = featuremap_kwds["VH_embedding"] # array of shape (n_vertices, dim)
    # feat_rotation_angle = featuremap_kwds["rotation_angle"]
    feat_mu = featuremap_kwds["mu"] # edge probability
    feat_phi_sum = np.zeros(n_vertices, dtype=np.float32) # For each node i in embedding space, sum of edge existing probality incident to this node
    feat_re_sum = np.zeros([n_vertices, dim], dtype=np.float32) # Embedding radius in principal directions
    feat_var_shift = featuremap_kwds["var_shift"]
    # else: 
    #     feat_mu_tot = 0
    #     feat_lambda = 0
    #     feat_R = np.zeros(1, dtype=np.float32)
    #     feat_VH = np.zeros(1, dtype=np.float32)
    #     feat_VH_embedding = np.zeros(1, dtype=np.float32)
    #     # feat_rotation_angle = np.zeros(1, dtype=np.float32)
    #     feat_mu = np.zeros(1, dtype=np.float32)
    #     feat_phi_sum = np.zeros(1, dtype=np.float32)
    #     feat_re_sum = np.zeros(1, dtype=np.float32)
        
    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose
    
    # print('Test1')
    import time
    for n in tqdm(range(n_epochs), **tqdm_kwds):
        # print('Test2')
        featuremap_flag = (
            # featuremap and 
            (featuremap_kwds["lambda"] > 0)
            and (((n + 1) / float(n_epochs)) > (1 - featuremap_kwds["frac"]))
        )
        # print(f'featuremap_flag_{featuremap_flag}')
        if featuremap_flag:
            # FIXME: feat_init_fn might be referenced before assignment
            
            # Compute the initial embedding under rotation VH
            # print('feat_re_sum, ' + str(feat_re_sum))
            # T1 = time.time()
            feat_init_fn(
                head_embedding,
                tail_embedding,
                head,
                tail,
                # random_state,
                # feat_VH,
                feat_VH_embedding,
                a,
                b,
                feat_re_sum,
                feat_phi_sum,
            )
            # T2 = time.time()
            # print(f'featuremap initialization time is {T2-T1}')
              
            # feat_init_fn.inspect_types()
            
            # FIXME: feat_var_shift might be referenced before assignment
            feat_re_std = np.sqrt(np.var(feat_re_sum, axis=0) + feat_var_shift)
            feat_re_mean = np.mean(feat_re_sum, axis=0)
            feat_re_sum_centered = np.subtract(feat_re_sum, feat_re_mean)

            product = np.diag(np.dot(feat_re_sum_centered.T, feat_R[:,:dim]))
            feat_re_cov = product / (n_vertices - 1)
        else:
            feat_re_std = np.zeros(dim, dtype=np.float32)
            feat_re_mean = np.zeros(dim, dtype=np.float32)
            feat_re_cov = np.zeros(dim, dtype=np.float32)
        
        # T1 = time.time()
        optimize_fn(
            head_embedding,
            tail_embedding,
            head,
            tail,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            dim,
            move_other,
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n,
            featuremap_flag,
            feat_phi_sum,
            feat_re_sum,
            feat_re_cov,
            feat_re_std,
            feat_re_mean,
            feat_lambda,
            feat_R,
            feat_VH_embedding,
            feat_mu,
            feat_mu_tot,
        )
        # T2 = time.time()
        # print(f'Optimize_fn time is {T2-T1}')
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))
        
    return head_embedding


