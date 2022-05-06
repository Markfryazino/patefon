import numpy as np
import torch
import ripserplusplus as rpp_py

from tqdm.auto import trange, tqdm
from sklearn.model_selection import train_test_split


def generate_orbit(point_0, r, n=1000):
    X = np.zeros([n, 2])
    xcur, ycur = point_0[0], point_0[1]
    
    for idx in range(n):
        xcur = (xcur + r * ycur * (1. - ycur)) % 1
        ycur = (ycur + r * xcur * (1. - xcur)) % 1
        X[idx, :] = [xcur, ycur]
    
    return X


def generate_orbits(m, rs=[2.5, 3.5, 4.0, 4.1, 4.3], n=1000, random_state=None):
    orbits = np.zeros((m * len(rs), n, 2))
    
    for j, r in enumerate(rs):
        points_0 = random_state.uniform(size=(m,2))

        for i, point_0 in enumerate(points_0):
            orbits[j*m + i] = generate_orbit(points_0[i], rs[j])
            
    return orbits, np.repeat(np.arange(len(rs), dtype=int), m)


def compute_diagrams(orbits):
    result = []
    for i in trange(len(orbits)):
        diagram = rpp_py.run("--format point-cloud --dim 1", orbits[i])
        H0 = np.array([homology[1] for homology in diagram[0]])
        H1 = np.array([np.array(list(homology)) for homology in diagram[1]])
        result.append([H0, H1])

    return result


def create_dataset():
    X, y = generate_orbits(1000, random_state=np.random.RandomState(42))
    diagrams = compute_diagrams(X)
    D_train, D_test, y_train, y_test = train_test_split(diagrams, y, test_size=0.3, shuffle=True)
    return D_train, D_test, y_train, y_test


def filter_topk_homology(H0, H1, top0, top1):
    H0_idxs = np.argsort(H0)[::-1][:top0]
    H1_idxs = np.argsort(H1[:,1] - H1[:,0])[::-1][:top1]
    return H0[H0_idxs], H1[H1_idxs]


class OrbitDataset(torch.utils.data.Dataset):
    def __init__(self, diagrams, y, top0, top1):
        self.y = y
        self.H = []
        for H0, H1 in tqdm(diagrams):
            self.H.append(filter_topk_homology(H0, H1, top0, top1))
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.H[idx], self.y[idx]


def collate_orbits(data):
    H0s = [obj[0] for obj, y in data]
    H1s = [obj[1] for obj, y in data]
    y = torch.LongTensor([y for obj, y in data])

    H0_len = max([len(H0) for H0 in H0s])
    H1_len = max([len(H1) for H1 in H1s])

    H0_matrix = torch.zeros((len(data), H0_len))
    H1_matrix = torch.zeros((len(data), H1_len, 2))
    H0_mask = torch.zeros_like(H0_matrix, dtype=int)
    H1_mask = torch.zeros((len(data), H1_len), dtype=int)

    for i, H0 in enumerate(H0s):
        H0_matrix[i, :H0.shape[0]] = torch.Tensor(H0)
        H0_mask[i, :H0.shape[0]] = 1

    for i, H1 in enumerate(H1s):
        H1_matrix[i, :H1.shape[0], :] = torch.Tensor(H1)
        H1_mask[i, :H1.shape[0]] = 1

    return {
        "H0": H0_matrix,
        "H1": H1_matrix,
        "H0_mask": H0_mask,
        "H1_mask": H1_mask,
        "target": y
    }
