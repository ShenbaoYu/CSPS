# -*- coding: utf-8 -*-
"""
Data preprocessing for KDD Cup data format.
"""

from distutils.command.build import build
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import statistics as stas
import itertools
import networkx as nx


def read(train_file, test_file):
    """
    FUNCTION: data pre-processing

    Inputs:
    -------
    :param train_file --> str
        the training dataset
    
    :param test_file --> str
        the testing dataset
    
    Outputs:
    -------
    :return stu_exe_ans --> dict
        the student answer record,
        each problem contains the total correct times and incorrect times, respectively.
    
    :return exe_stu --> ditc
        the exercise with the students who answered it
    
    :return exe_skill --> dict
        the exercise with its related skills.
    
    :return skill_exe --> dict
        the skill with its related exercises
    """

    data = dict()

    with open(train_file, "r") as file:  # read the training data
        for line in file.readlines()[1:]:  # skip the first line
            line = line.split('\t')
            data[line[0]] = line
    with open(test_file, "r") as file:  # read the testing data
        for line in file.readlines()[1:]:  # skip the first line
            line = line.split('\t')
            data[line[0]] = line
    sorted(data.keys())  # sort the original data based on the row number

    illegal = r'.*SkillRule.*'  # skill's illegal pattern, it is ilegal if containing "SkillRule"

    # stdudent's answer record for problems 
    # --> {stu_name: {exe_name: [correct_num incorrect_num], ...}, ...}
    stu_exe_ans = dict()
    # each exercise with the students who answered it
    exe_stu = dict()
    # each exercise with the covering skills
    exe_skill = dict()
    # each skill with the related exercises
    skill_exe = dict()
    # each exercise with its correct skill steps descriptions
    exe_skill_seq = dict()

    lines = list()
    current_exe = data['1'][2] + '-' + data['1'][3] + '-' + data['1'][4]  # the first exercise
    for datum in data.values():
        # get exercise name ('Problem Hierarchy' + 'Problem Name' + 'Problem View')
        exe_name = datum[2] + '-' + datum[3] + '-' + datum[4]
        if exe_name != current_exe:  # the current step is the another new exercise
            current_exe = exe_name
            
            # process the log of the student answering the previous exercise
            if len(lines):  # it contains the integral steps for the exercise
                is_legal = check_skill_legal(lines, illegal)  # check whether the skill name is legal
                if is_legal: # restore the previous logs
                    
                    # restore the skill sequence referring correct steps
                    _ = lines[0]
                    exe_name = _[2]+'-'+_[3]+'-'+_[4]
                    if not exe_name in exe_skill_seq.keys():  # new exercise
                        exe_skill_seq[exe_name] = dict()
                        # if all steps are correct and the exercise has never been recorded
                        if all([line[16] != 0 for line in lines]) and len(exe_skill_seq[exe_name]) == 0:
                            exe_skill_seq[exe_name] = dict()
                            s_set = [line[17] for line in lines]
                            step = 1
                            for step_skills in s_set:
                                ss = [s.split(';')[0].split(':')[1] for s in step_skills.split('~~')]  # get all the skills of the current step
                                exe_skill_seq[exe_name][step] = ss
                                step += 1
   
                    while len(lines):
                        _ = lines.pop(0)
                        stu_name = _[1]
                        exe_name = _[2]+'-'+_[3]+'-'+_[4]
                        step_correct = int(_[13]) # correct times at the current step (correct first attempt)
                        step_incorrect = int(_[14])  # incorrect times at the current step

                        if not stu_name in stu_exe_ans.keys():
                            stu_exe_ans[stu_name] = dict()
                        try:
                            stu_exe_ans[stu_name][exe_name][0] += step_correct
                            stu_exe_ans[stu_name][exe_name][1] += step_incorrect
                        except:
                            stu_exe_ans[stu_name][exe_name] = [0]*2
                            stu_exe_ans[stu_name][exe_name][0] += step_correct  # cumulate the correct times
                            stu_exe_ans[stu_name][exe_name][1] += step_incorrect  # cumulate the incorrect times
                        
                        if not exe_name in exe_stu.keys():
                            exe_stu[exe_name] = list()
                        if not stu_name in exe_stu[exe_name]:
                            exe_stu[exe_name].append(stu_name)
                        
                        if not exe_name in exe_skill.keys():
                            exe_skill[exe_name] = list()
                        
                        if int(_[16]) != 0:  # step anwer is correct 
                            s = _[17].split('~~')
                            # if not len(exe_skill_seq[exe_names]):
                            #     _step += 1
                            #     exe_skill_seq[exe_name][_step] = []
                            for r in s:
                                sn = r.split(';')[0].split(':')[1]  # skill name
                                # exe_skill_seq[exe_name][_step].append(sn)
                            
                                if not sn in exe_skill[exe_name]:
                                    exe_skill[exe_name].append(sn)   
                                if not sn in skill_exe.keys():
                                    skill_exe[sn] = list()
                                if not exe_name in skill_exe[sn]:
                                    skill_exe[sn].append(exe_name)                    
                else:
                    lines.clear()  # clear all steps     
        
        # store all answering step of the current exercise
        lines.append(datum)
        
    print("There are %d students, %d exercises, and %d skills" %(len(stu_exe_ans), len(exe_skill), len(skill_exe)))

    # plot the histogram of the exercises with the number of students answering it
    # plt.bar(range(len(exe_skill)), [len(x) for x in exe_stu.values()], color='b')
    # _ = sorted(exe_stu.items(), key=lambda x: len(x[1]), reverse=True)
    # plt.bar(range(len(exe_skill)), [len(x[1]) for x in _], color='b')
    # plt.xlabel('Exercise ID')
    # plt.ylabel('The number of students who did the exercise')
    # plt.show()
    # plt.savefig(BASE_DIR + '/Data/' + DATASET + '/exercise count.pdf')

    print('The sparsity of the student exercise response matrix is %.4f'
           % (sum([len(x) for x in stu_exe_ans.values()]) / (len(stu_exe_ans) * len(exe_skill))))
    print('The number of skills per exercise is %.2f' 
           % stas.mean([len(x) for x in exe_skill.values()]))
    print('The number of exercises per skill is %.2f' 
           % stas.mean([len(x) for x in skill_exe.values()]))

    return stu_exe_ans, exe_stu, exe_skill, skill_exe, exe_skill_seq


def check_skill_legal(lines, pattern):
    is_legal = True
    for line in lines:
        if re.match(pattern, line[17]):
            continue
        else:
            is_legal = False
            break
    return is_legal


def pre_exe(stu_exe_ans, exe_stu, exe_skill, skill_exe, D):
    """
    FUNCTION: delete exercises that 
    the number of students answering is less than D
    """

    exe_del = list()
    for exe, stus in exe_stu.items():
        if len(stus) <= D:
            exe_del.append(exe)

    for exe in exe_del:
        stus = exe_stu[exe]
        for stu in stus:
            stu_exe_ans[stu].pop(exe)
        skills = exe_skill[exe]
        exe_skill.pop(exe)
        for skill in skills:
            skill_exe[skill].remove(exe)
            if len(skill_exe[skill]) == 0:
                skill_exe.pop(skill)

    [exe_stu.pop(_) for _ in exe_del]
    
    stu_num, exe_num, skill_num = len(stu_exe_ans), len(exe_skill), len(skill_exe)

    print("After exercise deletion, there are %d students, %d exercises, and %d skills" %(stu_num, exe_num, skill_num))
    print('The sparsity of the student exercise response matrix is %.4f'
           % (sum([len(x) for x in stu_exe_ans.values()]) / (len(stu_exe_ans) * len(exe_skill))))
    print('The number of skills per exercise is %.2f' 
           % stas.mean([len(x) for x in exe_skill.values()]))
    print('The number of exercises per skill is %.2f' 
           % stas.mean([len(x) for x in skill_exe.values()]))

    return stu_exe_ans, exe_skill, skill_exe


def pre_skill(exe_skill, skill_exe, exe_skill_seq, del_skill, merge_skill):
    """
    FUNCTION: preprocess the skills
    """

    skill_edges_con = dict()

    # merge skills
    for pa, ch in merge_skill.items():
        for s_mer in ch:
            [skill_exe[pa].append(e) for e in skill_exe[s_mer] if not e in skill_exe[pa]]
            skill_exe.pop(s_mer)
            for e, s_set in exe_skill.items():
                if s_mer in s_set:
                    exe_skill[e].remove(s_mer)
                    if not pa in s_set:
                        exe_skill[e].append(pa)
            for e, s_seq in exe_skill_seq.items():
                for step, s_set in s_seq.items():
                    if s_mer in s_set:
                        _index = s_set.index(s_mer)
                        exe_skill_seq[e][step][_index] = pa
                        exe_skill_seq[e][step] = list(set(s_set))  # delete the duplicated skills

    # create skill edges
    for s_seq in exe_skill_seq.values():
        for k in range(2, len(s_seq)+1):
            pres = s_seq[k-1]
            poss = s_seq[k]
            for edge in itertools.product(pres, poss):
                if edge[0] == edge[1]:
                    continue
                try:
                    skill_edges_con[edge] += 1
                except:
                    skill_edges_con[edge] = 0
                    skill_edges_con[edge] += 1
    # Vote
    _temp = skill_edges_con.copy()
    for e_i, n_i in skill_edges_con.items():
        for e_j, n_j in skill_edges_con.items():
            if e_i != e_j and set(e_i) == set(e_j) and e_i in _temp.keys() and e_j in _temp.keys():
                _temp.pop(e_i) if n_i < n_j and e_i else _temp.pop(e_j)
    skill_edges = list(sorted(_temp.keys()))
    _temp = skill_edges.copy()
    [skill_edges.remove(edge) for edge in _temp if edge[0] in del_skill or edge[1] in del_skill]
    

                    
    # check
    flag = True
    for e, s_set in exe_skill.items():
        for s in s_set:
            if not e in skill_exe[s]:
                flag = False
    print('skill pre-processing is successful?:', flag)

    print("After skill pre-processing, there are %d valid skills" %(len(skill_exe)))
                         
    return exe_skill, skill_exe, skill_edges


def write(stu_exe_ans, exe_skill, skill_exe, g):
    """
    FUNCTION: Output:
        (1): the student-exercise response matrix
        (2): the Q-Matrix
        (3): students' descriptions
        (4): exercises' descriptions
        (5): skills' descriptions
        (6): skills' triggers
    """

    stu_num = len(stu_exe_ans)
    exe_num = len(exe_skill)
    skill_num = len(skill_exe)

    stu_exe = np.full((stu_num, exe_num), np.NaN)  # row:student, col:exercise
    q_matrix = np.zeros(shape=(exe_num, skill_num), dtype=int)  # row:exercise, col:skill
    stu_names = pd.DataFrame(columns=['No.', 'Student Names'])
    exe_names = pd.DataFrame(columns=['No.', 'Exercise Names'])
    skill_names = pd.DataFrame(columns=['No.', 'Skill Names'])
    trigger_names = pd.DataFrame(columns=['No.', 'Skill Names', 'Trigger Names'])

    stu_name_id = dict()
    for stu in stu_exe_ans.keys():
        stu_name_id[stu] = len(stu_name_id)
    exe_name_id = dict()
    for exe in exe_skill.keys():
        exe_name_id[exe] = len(exe_name_id)
    skill_name_id = dict()
    for skill in skill_exe.keys():
        skill_name_id[skill] = len(skill_name_id)
              
    #  --- get the student-exercise response matrix ---
    for stu_name, stu_id in stu_name_id.items():
        for exe_name, ans in stu_exe_ans[stu_name].items():
            exe_id = exe_name_id[exe_name]
            correct = ans[0]
            incorrect = ans[1]
            try:
                stu_exe[stu_id][exe_id] = correct / (correct + incorrect)
            except ZeroDivisionError:
                stu_exe[stu_id][exe_id] = 0
    
    # --- get the Q-matrix ---
    for exe_name, skills in exe_skill.items():
        exe_id = exe_name_id[exe_name]
        for skill_name in skills:
            skill_id = skill_name_id[skill_name]
            q_matrix[exe_id][skill_id] = 1  
    
    # --- get the ID with the corresponding name ---
    # --- get the triggers
    i, j, k = 1, 1, 1
    for stu_name, stu_id in stu_name_id.items():
        stu_names.loc[i] = {'No.': stu_id, 'Student Names': stu_name}
        i += 1 
    for exe_name, exe_id in exe_name_id.items():
        exe_names.loc[j] = {'No.': exe_id, 'Exercise Names': exe_name}
        j += 1
    for skill_name, skill_id in skill_name_id.items():
        skill_names.loc[k] = {'No.': skill_id, 'Skill Names': skill_name}
        trigger_names.loc[k] = {'No.': skill_id, 'Skill Names': skill_name, 'Trigger Names': g[skill_name]}
        k += 1

    return stu_exe, q_matrix, stu_names, exe_names, skill_names, trigger_names


def translate_to_graph(edges):
    _g = dict()  # {id:[parent id, ...], ...}
    for edge in edges:
        pre, pos = edge[0], edge[1]
        if not pre in _g.keys():
            _g[pre] = list()
        if not pos in _g.keys():
            _g[pos] = list()
            _g[pos].append(pre)
        else:
            _g[pos].append(pre)

    return _g


def build_graph(graph_info):
    g = nx.DiGraph()  # initialize an empty directed graph
    for node, parents in graph_info.items():
        g.add_node(node)
        for pa in parents:
            g.add_edge(pa, node)
    return g


def vis_graph(save_path, g):
    graph = build_graph(g)

    # graph attributes
    options = {
        'with_labels': True,
        'node_size': 500,
        # 'node_color': 'black',
        'font_color': 'black',
        'font_size': 14,
        'edge_color': 'Grey',
        'width': 0.7,
        'cmap': plt.cm.get_cmap('summer_r')
    }
    nx.draw(graph, **options)
    plt.savefig(save_path + '/graph' + ".pdf")




if __name__ == "__main__":
    DATASET = 'Alg0506'

    # because the original was divided into training data and testing data,
    # while the records in testing are randomly sampled from the original data,
    # which damages the integrality of the students' problem-solving process.
    stu_exe_ans, exe_stu, exe_skill, skill_exe, exe_skill_seq = \
    read(train_file = BASE_DIR+"/Data/"+DATASET+'/Ori/'+'algebra_2005_2006_train.txt',
         test_file = BASE_DIR+"/Data/"+DATASET+'/Ori/'+'algebra_2005_2006_master.txt')
    
    stu_exe_ans, exe_skill, skill_exe = pre_exe(stu_exe_ans, exe_stu, exe_skill, skill_exe, D=35)

    DEL_SKILL = [' Done?', ' done no solutions', ' done infinite solutions']
    MERGE_SKILL = {' Remove coefficient'         : [' Remove negative coefficient', ' Remove positive coefficient'],
                   ' Eliminate Parens'           : [' Select Eliminate Parens', ' Calculate Eliminate Parens'],
                   ' Multiply/Divide'            : [' Select Multiply', ' Select Multiply/Divide, nested'],
                   ' Remove constant'            : [' Isolate positive', ' Isolate negative', ' ax+b=c, negative'],
                   ' Consolidate vars, no coeff' : [' Select Combine Terms']}
    exe_skill, skill_exe, skill_edges = pre_skill(exe_skill, skill_exe, exe_skill_seq, DEL_SKILL, MERGE_SKILL)
    g = translate_to_graph(skill_edges)
    # vis_graph(BASE_DIR + '/Data/' + DATASET, g)

    # save file
    stu_exe, q_matrix, stu_names, exe_names, skill_names, trigger_names = write(stu_exe_ans, exe_skill, skill_exe, g)
    np.savetxt(BASE_DIR + '/Data/' + DATASET + '/data.txt', stu_exe, delimiter='\t', fmt='%.4f')
    np.savetxt(BASE_DIR + '/Data/' + DATASET + '/q.txt', q_matrix, fmt='%d')
    stu_names.to_csv(BASE_DIR + '/Data/' + DATASET + '/stunames.txt', sep='\t', index=False)
    exe_names.to_csv(BASE_DIR + '/Data/' + DATASET + '/exenames.txt', sep='\t', index=False)
    skill_names.to_csv(BASE_DIR + '/Data/' + DATASET + '/qnames.txt', sep='\t', index=False)
    trigger_names.to_csv(BASE_DIR + '/Data/' + DATASET + '/triggernames.txt', sep='\t', index=False)