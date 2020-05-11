from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

PARAMS = {'none': 0, 'sep_conv_3x3': 504, 'avg_pool_5x5': 0, 'max_pool_3x3': 0, 'max_pool_5x5': 0, 'avg_pool_3x3': 0,
          'skip_connect': 0, 'dil_conv_3x3': 252, 'conv_3x3': 1296, 'sep_conv_5x5': 888, 'conv_3x1_1x3': 864,
          'dil_conv_5x5': 444}

# independent search space
PRIMITIVES_NORMAL = [
    "conv_3x3",
    "conv_3x1_1x3",
    "sep_conv_3x3",
    "sep_conv_5x5",
    'dil_conv_3x3',
    'dil_conv_5x5',
]

PRIMITIVES_REDUCE = [
    "none",
    "skip_connect",
    "avg_pool_3x3",
    "max_pool_3x3",
    "avg_pool_5x5",
    "max_pool_5x5"
]


# DARTS
# gpu hours 10.4, param 3.3MB, valid_acc 95.29
DARTS_old = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

# DARTS + reg
# gpu hours 10.3, params 2.2MB , valid_acc 95.13, eta=0.001
DARTS_reg = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 0),
            ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2),
            ('skip_connect', 3), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

# DARTS+ind
# gpu hours 8.9, params 9.5MB, valid_acc 95.69
DARTS_ind = Genotype(
    normal=[('conv_3x1_1x3', 0), ('conv_3x1_1x3', 1), ('conv_3x1_1x3', 0), ('sep_conv_3x3', 2), ('conv_3x1_1x3', 3),
            ('conv_3x1_1x3', 0), ('conv_3x3', 4), ('conv_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 3), ('skip_connect', 2), ('none', 4)], reduce_concat=range(2, 6))

# DARTS+ind+reg
# eta=0.001
# gpu hours 8.8, params 8.9MB, valid_acc 95.23
DARTS_0001 = Genotype(
    normal=[('conv_3x1_1x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('conv 3x3', 2), ('conv 3x3', 3),
            ('conv_3x1_1x3', 2), ('conv_3x1_1x3', 2), ('conv_3x1_1x3', 0)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('none', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3),
            ('skip_connect', 2), ('none', 4), ('skip_connect', 2)], reduce_concat=range(2, 6))

# eta=0.1
# gpu hours 8.75, params 5.9MB, valid_acc 95.20
DARTS_01 = Genotype(
    normal=[('sep_conv_3x3', 1), ('conv_3x1_1x3', 0), ('conv_3x1_1x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('conv_3x1_1x3', 2)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3),
            ('skip_connect', 2), ('none', 4), ('skip_connect', 2)], reduce_concat=range(2, 6))

# eta=0.5
# gpu hours 8.5, params 6.2MB, valid_acc 95.38
DARTS_05 = Genotype(
    normal=[('conv_3x1_1x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3),
            ('sep_conv_3x3', 0), ('conv_3x1_1x3', 0), ('conv_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3),
            ('skip_connect', 2), ('none', 4), ('skip_connect', 2)], reduce_concat=range(2, 6))

# eta=0.8
# gpu hours 8.95, params 5.3MB, valid_acc 95.14
DARTS_08 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('conv_3x1_1x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3),
            ('conv_3x1_1x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 3), ('none', 4), ('skip_connect', 2)], reduce_concat=range(2, 6))

# eta=1
# gpu hours 8.75, params 4.1MB, valid_acc 95.04
DARTS_1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('conv_3x1_1x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2),
            ('dil_conv_5x5', 3), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('none', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 3), ('skip_connect', 2), ('none', 4)], reduce_concat=range(2, 6))

# eta=1.5
# gpu hours 9, params 3.8MB, valid_acc 94.56
DARTS_15 = Genotype(
    normal=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 3), ('skip_connect', 2), ('none', 4)], reduce_concat=range(2, 6))

# eta=2
# params 5.3  the balance is broken
# DARTS_2 = Genotype(
#     normal=[('conv_3x1_1x3', 0), ('conv_3x1_1x3', 1), ('sep_conv_3x3', 0), ('conv_3x1_1x3', 2), ('dil_conv_3x3', 2),
#             ('dil_conv_3x3', 3), ('conv_3x1_1x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
#     reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2),
#             ('skip_connect', 3), ('skip_connect', 2), ('none', 4)], reduce_concat=range(2, 6))

DARTS = DARTS_15
