"""
Implementation
6 and 26 neighbors
"""
import numpy as np
import itertools
from tqdm import tqdm_notebook as tqdm


def _get_nbhd_26(pt, checked, dims):
    nbhd = []
    # check pt values whether in range or not
    # search range
    r = [-1,0,1]
    moves = list(itertools.product(r,r,r))
    moves.remove((0,0,0))
    
    for mv in moves:
        if min((pt[0]+mv[0], pt[1]+mv[1], pt[2]+mv[2])) < 0:
            continue
        if (pt[0]+mv[0] > dims[0]-1) or (pt[1]+mv[1] > dims[1]-1) or (pt[2]+mv[2] > dims[2]-1):
            continue
        if not checked[pt[0]+mv[0], pt[1]+mv[1], pt[2]+mv[2]]:
            nbhd.append((pt[0]+mv[0], pt[1]+mv[1], pt[2]+mv[2]))

    return nbhd

def regionGrowing(img, seeds, t=1, ramda_thresh=0.90):
    """
    img  : ndarray, ndim=3
    seed : tuple, len=3
    t    : int, range of search
     The image neighborhood radius for the inclusion criteria
    """
    # segmented and checked volume
    seg = np.zeros(img.shape, dtype=np.bool)
    checked = np.zeros_like(seg)
    
    for seed in tqdm(seeds):
        value_thresh = img[seed[0],:,:].max()/2

        seg[seed] = True
        checked[seed] = True
        needs_check = _get_nbhd_26(seed, checked, img.shape) # get neighborhood

        while len(needs_check) > 0:
            # processing points one by one
            pt = needs_check.pop()

            # A point can be put in needs_check even if the point was already marked checked.
            if checked[pt]:
                continue
            checked[pt] = True

            imin = max(pt[0]-t, 0)
            imax = min(pt[0]+t, img.shape[0]-1)
            jmin = max(pt[1]-t, 0)
            jmax = min(pt[1]+t, img.shape[1]-1)
            kmin = max(pt[2]-t, 0)
            kmax = min(pt[2]+t, img.shape[2]-1)

            # Count nums of over thresh
            ramda = np.sum(img[imin:imax+1, jmin:jmax+1, kmin:kmax+1] > value_thresh) / 26

            # adapt the conditions below for paper
            if ramda >= ramda_thresh:
                # Include the voxel in the segmentation and
                # add its neighbors to be checked.
                seg[pt] = True
                needs_check += _get_nbhd_26(pt, checked, img.shape)

    return seg
            