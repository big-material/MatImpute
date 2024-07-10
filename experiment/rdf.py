import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set_theme(style="whitegrid",font="Times New Roman",font_scale=1.5)

from Utils import simulate_nan


def gen_data(df: pd.DataFrame, ratio: float, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(np.random.randint(0, np.iinfo(np.int32).max))
    vals = df.values
    while True:
        try:
            MAR_result = simulate_nan(vals, ratio, mecha="MAR", p_obs=0.0)
        except:
            continue
        if MAR_result is not None:
            break
    MCAR_result = simulate_nan(vals, ratio, mecha="MCAR")
    while True:
        try:
            MNAR_result = simulate_nan(vals, ratio, mecha="MNAR")
        except:
            continue
        if MNAR_result is not None:
            break
    MAR_mask = MAR_result["mask"].astype(bool)
    MCAR_mask = MCAR_result["mask"].astype(bool)
    MNAR_mask = MNAR_result["mask"].astype(bool)
    # fint the col in MAR not exists nan
    for i in range(MAR_mask.shape[1]):
        if not np.any(MAR_mask[:,i]):
            idx = i
    # drop the col in MAR not exists nan
    MAR_mask = np.delete(MAR_mask, idx, axis=1)
    MCAR_mask = np.delete(MCAR_mask, idx, axis=1)
    MNAR_mask = np.delete(MNAR_mask, idx, axis=1)
    print("Generate succcess.")
    return MAR_mask, MCAR_mask, MNAR_mask


def get_points(mask):
    points = []
    n, m = mask.shape
    box_size = max(n, m)
    v_repeat = math.ceil(box_size / n)
    h_repeat = math.ceil(box_size / m)
    box  = np.tile(mask, (v_repeat, h_repeat))
    box = box[:box_size, :box_size]

    for i in range(box_size):
        for j in range(box_size):
            if box[i, j]:
                points.append([i, j])
    points = np.array(points)
    return points, box_size

def cal_dis(points, box_size):
    dis = []
    n = points.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            posi = points[i]
            posj = points[j]
            dr = posj - posi
            dr = dr - box_size * np.floor(dr / box_size + 0.5)

            dist = np.sqrt(np.sum(dr**2))
            dis.append(dist)
    return np.array(dis)

def cal_dis_vec(points, box_size):
    points = np.asarray(points, dtype=np.float16)
    box_size = np.float16(box_size)
    
    n = points.shape[0]
    
    # Calculate all pairwise differences
    diff = points[np.newaxis, :, :] - points[:, np.newaxis, :]
    
    # Apply minimum image convention
    diff -= box_size * np.floor(diff / box_size + 0.5)
    
    # Calculate distances
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Get upper triangular indices excluding diagonal
    i_upper, j_upper = np.triu_indices(n, k=1)
    upper_triangular = dist[i_upper, j_upper]
    
    # Get lower triangular indices excluding diagonal
    i_lower, j_lower = np.tril_indices(n, k=-1)
    lower_triangular = dist[i_lower, j_lower]

    all_distances = np.concatenate((upper_triangular, lower_triangular))

    return all_distances


def hist_dis(dists, max_dist, bin_size):
    bins = np.arange(0, max_dist + bin_size, bin_size)
    hist, bin_edge = np.histogram(dists,bins=bins)
    return hist, bin_edge

def plot_hist(hist, bin_edge):
    bin_center = (bin_edge[:-1] + bin_edge[1:]) / 2.0
    plt.plot(bin_center, hist, marker="o")
    plt.xlabel("$r$")
    plt.ylabel("N(r)")
    plt.show()

def plot_rdf(gofr, bin_center, save=False, save_path=None):
    plt.plot(bin_center, gofr, marker="o")
    plt.xlabel("$r$")
    plt.ylabel("g(r)")
    if save and save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()

def get_gofr(hist,bin_edges,num_particles, box_size):
    rho = num_particles/ box_size / box_size
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2.0
    dr = bin_edges[1]-bin_edges[0]
    denominator = 2.*np.pi*bin_centers*dr*rho*num_particles
    gofr = hist/denominator
    
    return gofr, bin_centers

if __name__ == "__main__":
    from tqdm import tqdm
    import math 

    df = pd.read_csv("glass.csv")
    # select the numeric columns
    df = df.select_dtypes(include=[np.number])  

    gofr_all = {
        "MAR" : [],
        "MCAR" : [],
        "MNAR" : []
    }   

    num_tests = 100
    for i in tqdm(range(num_tests)):
        MAR_mask, MCAR_mask, MNAR_mask = gen_data(df.copy(), 0.1)
        MAR_points,box_size = get_points(MAR_mask)
        MCAR_points,_ = get_points(MCAR_mask)
        MNAR_points,_ = get_points(MNAR_mask)
        all_points = {
            "MAR" : MAR_points,
            "MCAR" : MCAR_points,
            "MNAR" : MNAR_points,
        }
        for k, points in all_points.items():
            num_particles = len(points)
            dis = cal_dis_vec(points, box_size=box_size)
            bin_size = 4
            max_dist = box_size / 2.0
            hist, bin_edges = hist_dis(dis, max_dist, bin_size)
            gofr, bin_centers = get_gofr(hist, bin_edges, num_particles, box_size)
            gofr_all[k].append(gofr)
    avg_gofr_MAR = np.mean(gofr_all["MAR"], axis=0)
    avg_gofr_MCAR = np.mean(gofr_all["MCAR"], axis=0)
    avg_gofr_MNAR = np.mean(gofr_all["MNAR"], axis=0)
    np.save("./3types-rdf-glass.npy", np.array([avg_gofr_MAR, avg_gofr_MCAR, avg_gofr_MNAR, bin_centers]))

    gofr_all = {
        "0.1" : [],
        "0.3" : [],
        "0.5" : []
    }
    num_tests = 100
    for i in tqdm(range(num_tests)):
        for ratio in gofr_all.keys():
            MAR_mask, _, _ = gen_data(df.copy(), float(ratio))
            MAR_points,box_size = get_points(MAR_mask)
            num_particles = len(MAR_points)
            dis = cal_dis_vec(MAR_points, box_size=box_size)
            bin_size = 4
            max_dist = box_size / 2.0
            hist, bin_edges = hist_dis(dis, max_dist, bin_size)
            gofr, bin_centers = get_gofr(hist, bin_edges, num_particles, box_size)
            gofr_all[ratio].append(gofr)
    avg_gofr_01 = np.mean(gofr_all["0.1"], axis=0)
    avg_gofr_03 = np.mean(gofr_all["0.3"], axis=0)
    avg_gofr_05 = np.mean(gofr_all["0.5"], axis=0)
    np.save("./3ratios-rdf-glass.npy", np.array([avg_gofr_01, avg_gofr_03, avg_gofr_05, bin_centers]))