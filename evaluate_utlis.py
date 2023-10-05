import random
import numpy as np
import torch
from torch.nn import functional as F
import seaborn as sns
from numpy import linalg as LA
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon, MultiPolygon

from matplotlib import pyplot as pl
import ot
import ot.plot

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def gen_colors(num_colors):
    """
    Generate uniformly distributed `num_colors` colors
    :param num_colors:
    :return:
    """
    if num_colors <= 12:
        palette = sns.color_palette("Paired", num_colors)
    else:
        palette1 = sns.color_palette("Paired", 12)
        palette2 = sns.color_palette("Set2", num_colors - 12)
        palette = palette1 + palette2
        
    rgb_triples = [[int(x[0]*255), int(x[1]*255), int(x[2]*255)] for x in palette]
    return rgb_triples



@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.module.get_block_size() if hasattr(model, "module") else model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x



@torch.no_grad()
def vae_sample(model, x, mu, logvar, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.module.get_block_size() if hasattr(model, "module") else model.get_block_size()   # max_length of the sequence
    model.eval()
    z = model.module.reparameterization(mu, logvar) if hasattr(model, "module") else model.reparameterization(mu, logvar)
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits = model.module.inference(x_cond, z) if hasattr(model, "module") else model.inference(x_cond, z)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)   # pick up one of the top-5, based on their relative predicted weights.
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x




def trim_tokens(tokens, bos, eos, pad=None):
    bos_idx = np.where(tokens == bos)[0]
    tokens = tokens[bos_idx[0]+1:] if len(bos_idx) > 0 else tokens
    eos_idx = np.where(tokens == eos)[0]
    tokens = tokens[:eos_idx[0]] if len(eos_idx) > 0 else tokens
    # tokens = tokens[tokens != bos]
    # tokens = tokens[tokens != eos]
    if pad is not None:
        tokens = tokens[tokens != pad]
    return tokens


def get_max_match(cost_matrix, maximize = True):
    num_layout = np.amax(cost_matrix.shape)
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=maximize)
    max_match = cost_matrix[row_ind, col_ind]

    return max_match.sum() / np.float16(num_layout)



def get_similar_value(bbx1, bbx2):
    C = 0.5
    C_S = 2

    center1 = np.array( [(bbx1[0] + bbx1[2]) / 2.0, (bbx1[1] + bbx1[3]) / 2.0] )
    center2 = np.array( [(bbx2[0] + bbx2[2]) / 2.0, (bbx2[1] + bbx2[3]) / 2.0] )

    size1 = np.array( [(bbx1[2] - bbx1[0]) , (bbx1[3] - bbx1[1])] )
    size2 = np.array( [(bbx2[2] - bbx2[0]) , (bbx2[3] - bbx2[1])] )


    delta_C = LA.norm(center1 - center2, 2)
    delta_S = LA.norm(size1 - size2, 1)
    alpha_a = min( [size1[0] * size1[1], size2[0] * size2[1]] ) ** C

    exp = -delta_C - C_S * delta_S
    value = alpha_a * pow(2, exp)

    return value


def get_DocSim(cat1, layout1, cat2, layout2):
    cat1 = np.array((cat1 - 255), np.int16).reshape(-1)
    cat2 = np.array((cat2 - 255), np.int16).reshape(-1)

    layout1 = layout1 / 256.0
    layout2 = layout2 / 256.0

    len1 = len(layout1)
    len2 = len(layout2)
    max_len = max(len1, len2)
    min_len = min(len1, len2)
    
    cost_matrix = np.zeros((max_len, max_len))

    for i in range(len1):
        for j in range(len2):
            if cat1[i] == cat2[j]:
                cost_matrix[i, j] = get_similar_value(layout1[i], layout2[j])

    
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    max_match = cost_matrix[row_ind, col_ind]
    value = max_match.sum() / np.double(max_len)

    return value



def calculate_images(cat, layout):

    images = np.zeros((13,64,64))
    for ind, thiscat in enumerate(cat):
        bbox = layout[ind]
        temp_image = np.zeros((64,64))
        temp_image[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1
        
        images[thiscat,:,:] += temp_image
    return images

    
def get_Docustein(cat1, layout1, cat2, layout2, visualize=False):
    cat1 = np.array((cat1 - 256), np.int16).reshape(-1)
    cat2 = np.array((cat2 - 256), np.int16).reshape(-1)

    layout1 = np.array( [[np.floor(lay[0]/4.0), np.floor(lay[1]/4.0), np.ceil(lay[2]/4.0), np.ceil(lay[3]/4.0)] for lay in  layout1], dtype=np.uint8)
    layout2 = np.array( [[np.floor(lay[0]/4.0), np.floor(lay[1]/4.0), np.ceil(lay[2]/4.0), np.ceil(lay[3]/4.0)] for lay in  layout2], dtype=np.uint8)
    images1 = calculate_images(cat1, layout1)
    images2 = calculate_images(cat2, layout2)

    category_distances = np.zeros((13,1))
    for category in range(13):
        img1 = np.squeeze( images1[category,:,:] )
        img2 = np.squeeze( images2[category,:,:] )
        xy1 = np.argwhere(img1 > 0)
        xy2 = np.argwhere(img2 > 0)
        
        xy1 = xy1/256
        xy2 = xy2/256

        if xy1.size == 0 and xy2.size == 0:
            category_distances[category] = 0
            continue
        elif xy1.size == 0:
            category_distances[category] = 1 # xy2.shape[0] 
        elif xy2.size == 0:
            category_distances[category] = 1 # xy1.shape[0] 
        else:
            # calculate the EMD
            #M = ot.dist(xy1, xy2)
            m = xy1.shape[0]
            n = xy2.shape[0]
            weights1, weights2 = np.ones((m,)) / m, np.ones((n,)) / n
            #G0 = ot.sinkhorn(weights1, weights2, M)
            cost = ot.max_sliced_wasserstein_distance( xy1, xy2, weights1, weights2)#np.sum((M*G0)[:])
            category_distances[category] = cost

            ###### visualization 
            if visualize:
                M = ot.dist(xy1, xy2)
                G0 = ot.sinkhorn(weights1, weights2, M)
                ot.plot.plot2D_samples_mat(xy1, xy2, G0, color=[.5, .5, 1])
                pl.plot(xy1[:, 0], xy1[:, 1], '+b', label='Source samples')
                pl.plot(xy2[:, 0], xy2[:, 1], 'xr', label='Target samples')
                pl.legend(loc=0)
                pl.title('OT matrix Sinkhorn with samples')
                pl.show()

    return sum(category_distances)



def get_bbx(bbx):
    x1, y1, x2, y2 = bbx
    bl = ( x1, y1 )
    br = ( x2, y1 )
    ur = ( x2, y2 )
    ul = ( x1, y2 )
    return bl, br, ur, ul



def get_overlap_loss(bbx):
    iou_loss = 0.0
    area_overlap = 0.0
    numm = bbx.shape[0]

    for i in range(numm):
        for j in range(i+1, numm):
            p1 = Polygon(get_bbx(bbx[i]))
            p2 = Polygon(get_bbx(bbx[j]))

            p_intersect = p1.intersection(p2).area
            if p_intersect > 0.0:
                if p1.area < 0.01 or p2.area < 0.01:
                    print('line detected: ', p1.area , p2.area)
                    continue

                p_overlap = p_intersect / (p1.area + p2.area - p_intersect + 1e-5)
                # print(i, j, p_intersect, (p1.area + p2.area - p_intersect + 1e-5), p_overlap)
                iou_loss += p_overlap
                area_overlap += p_intersect

    layout = []
    for i in range(numm):
        p1 = Polygon(get_bbx(bbx[i]))
        layout.append(p1)

    multi_poly = MultiPolygon(layout)
    area = multi_poly.area
    coverage = area / np.double(256**2)

    print('final: ',iou_loss / numm, area_overlap / numm, coverage)
    return iou_loss / numm, area_overlap / numm, coverage