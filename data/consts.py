INSTR_TO_INDEX = {
    -1: 0,
    0: 1,
    25: 2,
    32: 3,
    40: 4,
    80: 5
}

INDEX_TO_INSTR = dict([(v, k) for k, v in INSTR_TO_INDEX.items()])
EMPTY_INDEX = 1726