import numpy as np
import sys

def gen_sparse_mat(thresh, mat):
    (l, w) = mat.shape
    for i in xrange(l):
        for j in xrange(w):
            r = np.random.uniform()
            if r <= thresh:
                mat[i][j] = 1
    non_sparse = np.sum(mat)
    density  = non_sparse*1.0/(l * w)
    print "Density is "+str(density)
    return mat.astype(np.int32)

def gen_zero(l, w):
    return np.zeros((l, w)).astype(np.int32)


if __name__ == '__main__':
    l, w, thresh = sys.argv[1], sys.argv[2], sys.argv[3]
    starter = gen_zero(int(l), int(w))
    sparse_mat = gen_sparse_mat(np.float32(thresh)/100.0, starter)
    np.savetxt(open('data.in', 'wb'), sparse_mat.astype(np.int32), fmt="%i",delimiter=' ')

