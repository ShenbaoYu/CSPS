import random
import numpy as np
import itertools
import copy


def merge_stu_exe(stu_exe, q_matrix):
    """
    FUNCTION: Integrating the problems that have the idendity knowledge concepts set.
    METHOD: we do two pre-process respectively:
        1. In Q-matrix, delete the redundant problems and remain only one.
        2. In Exercise-Student data matrix, for each student, average the scores of these problems referring the same knowledge concepts set.

    Inputs:
    ----------------- 
    :param stu_exe --> numpy.ndarray
        the Student-Exercise Data Matrix
        row: exercise
        col: student
    
    :param q_matrix --> numpy.ndarray
        the Q-matrix
        row: exercise
        col: knowledge concept
    
    Outputs:
    ----------------- 
    :return stu_exe_merge
        the pre-processing Student-Exercise Data Matrix 
        row: exercise
        col: student

    :return q_merge
        the pre-processing Q-Matrix
        row: exercise
        col: knowledge concept
    """
    
    """ find the exercises that have the same knowledge concepts. """
    exe_same_pair = list()
    exe_same_list = [[]]
    for pair in itertools.combinations(range(len(q_matrix)), 2):  # comparing everay pair of exercise
        if np.array_equal(q_matrix[pair[0]], q_matrix[pair[1]]):
            exe_same_pair.append(list(pair))
            exe_same_list.append(list(set(exe_same_list[0]).union(set(pair))))
            exe_same_list.pop(0)
    # merging every pair to form the sets that are mutually exclusive
    while True:
        flag = False
        exe_same_pair_new = copy.copy(exe_same_pair)  # make a copy
        for pair in itertools.combinations(range(len(exe_same_pair)), 2):
            if len(set(exe_same_pair[pair[0]]) & set(exe_same_pair[pair[1]])) != 0:
                flag = True
                _union_a = exe_same_pair[pair[0]]
                _union_b = exe_same_pair[pair[1]]
                exe_same_pair_new.append(list(set(_union_a).union(set(_union_b))))
                try:
                    exe_same_pair_new.pop(exe_same_pair_new.index(_union_a))
                    exe_same_pair_new.pop(exe_same_pair_new.index(_union_b))
                except:
                    continue
                break   
        if flag == False:
            break
        exe_same_pair = copy.copy(exe_same_pair_new)

    """ form the new Student-Exercise Data Matrix and Q-Matrix respectively"""
    new_problem_2_stu_exe = dict()
    new_problem_2_q_matrx = dict()
    _count = 0
    for _ in exe_same_pair:
        new_problem_2_q_matrx[_count] = q_matrix[_[0]]
        new_problem_2_stu_exe[_count] = np.mean(stu_exe[_], axis=0)
        _count += 1
    
    stu_exe_new = np.delete(stu_exe, exe_same_list, axis=0)
    q_matrix_new = np.delete(q_matrix, exe_same_list, axis=0)

    for key, value in new_problem_2_q_matrx.items():
        q_matrix_new = np.r_[q_matrix_new, np.array([value])]
        stu_exe_new = np.r_[stu_exe_new, np.array([new_problem_2_stu_exe[key]])] 
    
    return stu_exe_new, q_matrix_new


def missing_stu_exe(stu_exe, miss_rate):
    """
    FUNCTION: Missing some entries in the Student-Exercise Data Matrix.
    METHOD: The data deficiency is totally random.
    
    Inputs:
    -----------------
    :param stu_exe --> numpy.ndarray
        the Student-Exercise Data Matrix
        row: exercise
        col: student
    
    :param miss_rate --> float
        missing ratio

    Outputs:
    -----------------
    :return stu_exe_miss --> numpy.ndarray
        the new Student-Exercise Data Matrix 
        after running data-missing preprocessing.
        row: exercise
        col: student
    
    :return miss_coo --> list(tuple(),...)
        the indexes of the missing entries
    """

    shape = stu_exe.shape  # the row and column of stu_exe --> (row number, col number)
    miss_coo = list()  # record the index of missing entry
    stu_exe_miss = stu_exe.copy()  # deep copy
    miss_count = int(miss_rate * (shape[0] * shape[1]))
    while miss_count != 0:
        row = random.randint(0, shape[0]-1)
        col = random.randint(0, shape[1]-1)
        if not np.isnan(stu_exe_miss[row][col]):
            stu_exe_miss[row][col] = np.NaN
            miss_count -= 1
            miss_coo.append((row, col))

    return stu_exe_miss, miss_coo


def sample_stu_exe(stu_set, stu_exe):
    """
    FUNCTION: Reconstructing the small dataset(stu_exe_sample) 
    from the full dataset(stu_exe) based on the student'IDs in stu_set.
    
    Inputs:
    -----------------
    :param stu_set --> dict()
        The selected student'IDs.

    :param stu_exe --> numpy.ndarray
        The Student-Exercise Data Matrix.
        row: exercise
        col: student

    Outputs:
    -----------------
    :return stu_exe_sample --> numpy.ndarray
        The reconstructed dataset.
    """

    stu_exe_sample = np.empty(shape=[stu_exe.shape[0], len(stu_set)])

    _ = 0
    for _stu_id in stu_set:
        stu_exe_sample[:, _] = stu_exe[:, _stu_id]
        _ += 1
    
    return stu_exe_sample


def cal_stu_skill_state(stu_exe, q_m):
    """
    Function: Calculate each student's skills' mastery based on counting

    Inputs:
    -----------------
    :params stu_exe --> numpy.ndarray()
        the students' response data

    :param q_m --> numpy.ndarray()
        the Q matrix
    
    Outputs:
    -----------------
    :return stu_skill_state --> dict(stu id : {skill id : state, ...})
    """

    stu_skill_state = dict()  # initialization
    exe_num, stu_num = stu_exe.shape
    # skill_num = q_m.shape[1]

    for stu in range(stu_num):
        if not stu in stu_skill_state.keys():
            stu_skill_state[stu] = dict()
        for exe in range(exe_num):
            ans = stu_exe[exe][stu]
            skills = np.argwhere(q_m[exe] == 1).tolist()  # get the skill set related to exe
            for skill in skills:
                try:
                    stu_skill_state[stu][skill[0]] += ans
                except:
                    stu_skill_state[stu][skill[0]] = 0
                    stu_skill_state[stu][skill[0]] += ans
                
    return stu_skill_state


def translate_to_graph(nm):
    """
    FUNCTION: translate the adjacency matrix to the graph

    Inputs:
    -----------------
    :param nm --> numpy.ndarray()
        The adjacency matrix
    
    Outputs:
    -----------------
    :return lg --> dict(id : [parent id])
        the latents' graph
    """

    lg =dict()

    row = nm.shape[0]
    for _ in range(row):
        lg[_] = list()
        parents = np.argwhere(nm[_] == 1).tolist()  # find all parents
        for pa in parents:
            lg[_].append(pa[0])

    return lg


def translate_to_nm(graph):
    """
    FUNCTION: translate the graph to the adjacency matrix
    
    Inputs:
    -----------------
    :param lg --> dict(id : [parent id])
        the latents' graph
    
    Outputs:
    -----------------
    :return nm --> numpy.ndarray()
        The adjacency matrix
    """
    skill_num = len(graph)
    nm = np.zeros(shape=(skill_num, skill_num), dtype=int)

    for sk, parents in graph.items():
        for pa in parents:
            nm[sk][pa] = 1
            nm[pa][sk] = -1

    return nm