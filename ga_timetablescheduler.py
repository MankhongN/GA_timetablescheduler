import pandas as pd
import numpy as np
from tabulate import tabulate
import math
import itertools
import time

df = pd.read_excel('Project_data 1.xlsx', sheet_name=['subject', 'professor', 'student', 'room', 'class'])

df_pro = df['professor']
df_stu = df['student']
df_r = df['room']
df_class = df['class']

"""# Population"""

sch = []
for i in range(len(df_class)):
    for j in range(int(df_class['hours'][i] / 4)):
        sch.append(df_class['class_id'][i] + str(int(j)))

for i in range(160 - len(sch)):
    sch.append(str(i))
sch = np.array(sch)

n_pop = 1000
pop = []
for i in range(n_pop):
    pop.append(np.random.permutation(sch))
pop = np.array(pop)

"""# Fitness function (best is closing to 0)"""


def fitness(pop):
    F = []
    for p in pop:
        li_check = []
        conflict = 0
        l_type = None
        for i in range(160):
            if len(p[i]) <= 6:
                li_check.append([i, i])
            else:
                l_type = p[i][0]
                li_check.append(np.array([p[i][1:3], p[i][5:7]]))
                # Check room type is not correct (lab (b) or lecture (a))
                if i % 32 <= 11:
                    if l_type != 'b':
                        conflict += 1
                else:
                    if l_type != 'a':
                        conflict += 1
        li_check = np.array(li_check)
        for i in range(5):
            # Check duplicate student_group and Professor
            for a, b in itertools.combinations(li_check[np.array([k for k in range(i * 32, 32 * (i + 1), 2)])], 2):
                conflict += np.sum(a == b)
            for a, b in itertools.combinations(li_check[np.array([k + 1 for k in range(i * 32, 32 * (i + 1), 2)])], 2):
                conflict += np.sum(a == b)

        F.append(conflict)
    return np.array(F)


"""# Crossover & Mutation"""


def mutate(offs):
    i, j = np.random.permutation(len(offs))[:2]
    offs[[i, j]] = offs[[j, i]]
    return offs


def crossover(p):
    i, j = np.random.permutation(len(p))[:2]
    offs1 = p[i].copy()
    offs2 = p[j].copy()
    temp = list(p[i].copy())
    island = []
    sub = []
    k = 0
    while True:

        if len(sub) == 0:
            sub.append(k)
            temp.remove(p[i][k])

        if p[j][k] == p[i][sub[0]]:
            island.append(sub.copy())
            sub.clear()
            if len(temp) != 0:
                k = np.where(p[i] == temp[0])[0][0]
            else:
                break

        else:
            k = np.where(p[i] == p[j][k])[0][0]
            sub.append(k)
            temp.remove(p[i][k])

    for idx in island:
        if island.index(idx) % 2 == 1:
            offs1[idx], offs2[idx] = p[j][idx], p[i][idx]

    if np.random.rand() < 0.02:
        offs1 = mutate(p[i])
    if np.random.rand() < 0.02:
        offs2 = mutate(p[j])

    return offs1, offs2


"""# Reproduction"""


def reproduction(pop):
    print('Start')
    start_time = time.time()
    F = []
    gen = 0
    result = []
    while True:
        F = fitness(pop)
        idx = F.argsort()
        pop = pop[idx]
        print(f'gen: {gen}, fitness = {F[idx[0]]}')
        gen += 1
        if F[idx[0]] == 0:
            end_time = time.time()
            print("Total time: {:.2f} seconds".format(end_time - start_time))
            result = pop[0]
            break
        for j in range(int(len(pop) / 2), len(pop), 2):
            offs1, offs2 = crossover(pop[:500])
            pop[j] = offs1
            pop[j + 1] = offs2
    return result


best_chro = reproduction(pop)

for i in range(len(best_chro)):
    best_chro[i] = best_chro[i][:-1]


def stu_sche(student):
    # Table format
    table = [['Time / Day', '8.00 - 12.00', '12.00 - 14.00', '14.00 - 18.00'],
             ['Monday', '', '', ''],
             ['Tuesday', '', '', ''],
             ['Wednesday', '', '', ''],
             ['Thursday', '', '', ''],
             ['Friday', '', '', '']]

    id = df_class[df_class['group_name'] == student]['class_id'].values
    idx = np.array([])
    for s in id:
        idx = np.hstack((idx, np.where(best_chro == s)[0]))

    for id in idx:
        id = int(id)
        room = df_r[df_r['id'] == math.floor((id % 32) / 2)]['name'].values[0]
        table[math.ceil(id / 32)][2 * (id % 2) + 1] = best_chro[id] + '\n' + room
    print("Student group : ", student)
    print(tabulate(table, headers='firstrow', tablefmt="fancy_grid", stralign="center"))


def pro_sche(professor):
    # Table format
    table = [['Time / Day', '8.00 - 12.00', '12.00 - 14.00', '14.00 - 18.00'],
             ['Monday', '', '', ''],
             ['Tuesday', '', '', ''],
             ['Wednesday', '', '', ''],
             ['Thursday', '', '', ''],
             ['Friday', '', '', '']]

    id = df_class[df_class['professor'] == professor]['class_id'].values
    idx = np.array([])
    for s in id:
        idx = np.hstack((idx, np.where(best_chro == s)[0]))

    for ide in idx:
        ide = int(ide)
        room = df_r[df_r['id'] == math.floor((ide % 32) / 2)]['name'].values[0]
        table[math.ceil(ide / 32)][2 * (ide % 2) + 1] = best_chro[ide] + ', ' + \
                                                        df_class[df_class['class_id'] == best_chro[ide]][
                                                            'group_name'].values[0] + '\n' + room
    print("Professor : ", professor)
    print(tabulate(table, headers='firstrow', tablefmt="fancy_grid", stralign="center"))


print("Student section")
print("-" * 20)
for stu in df_stu['group_name'].values:
    stu_sche(stu)
print("Professor section")
print("-" * 20)
for pro in df_pro['name'].values:
    pro_sche(pro)
