##############################################################################
# This skeleton was created by Efehan Guner  (efehanguner21@ku.edu.tr)       #
# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
from enum import unique
import glob
import os
import sys
from copy import deepcopy
import numpy as np
import datetime
from Tree import  Tree
import copy
from Lattice import  Lattice

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################

def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset)>0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True



def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    #TODO: complete this code so that a DGH file is read and returned
    # in your own desired format.
    file1 = open(DGH_file, 'r')
    Lines = file1.readlines()
    tab = 0
    lineByCount={}
    # Strips the newline character
    for line in Lines:
        countOfTab= line.count('\t')
        newLine=''.join(line.split())
        lineByCount[newLine] =countOfTab
    prev = None
    rootNode = None

    for key in lineByCount:
        value = lineByCount[key]
        if(prev == None):

            rootNode = Tree()
            rootNode.level = value
            rootNode.data = key
            prev = rootNode
        else:
            howDeep = value
            if howDeep > prev.level:
                node = Tree()
                node.data = key
                node.level = howDeep
                node.parent = prev
                prev.children.append(node)
                prev= node
            elif howDeep == prev.level:
                node = Tree()
                node.data = key
                node.level = howDeep
                node.parent = prev.parent
                prev.parent.children.append(node)
                prev= node
            else:
                while(howDeep < prev.level):
                    prev = prev.parent
                node = Tree()
                node.data = key
                node.level = howDeep
                node.parent = prev.parent
                prev.parent.children.append(node)
                prev= node
    return rootNode

def generilize(dataset_para,clusterTree,qui):
    dataset = copy.deepcopy(dataset_para)
    min = -1
    for data in dataset:

        if clusterTree[qui].BFS(data[qui]).parent != None:
            cur = clusterTree[qui].BFS(data[qui]).level
        else:
            cur = 0
        if cur >= min:
            min = cur
    if min == 0:
        return min,dataset


    for data in dataset:
        to_handle = clusterTree[qui].BFS(data[qui])
        if to_handle.parent != None:
            if to_handle.level == min:
                data[qui]= to_handle.parent.data
    return min,dataset


def generilizeTheAttribute(clusterTree,qui,orderDict,otherDict):
    treeOfQui=clusterTree[qui]
    node1=treeOfQui.BFS(orderDict[qui])
    node2=treeOfQui.BFS(otherDict[qui])

    if(node1.parent == None):
        orderDict[qui] = node1.data
        otherDict[qui]=node1.data
    if(node2.parent == None):
        orderDict[qui] = node2.data
        otherDict[qui]=node2.data

    if(node2.level > node1.level):
        while(node2.level !=node1.level):
            if node2.parent == None:
                orderDict[qui] = node2.data
                otherDict[qui]=node2.data
            node2 = node2.parent

        if(node1.data == node2.data):
            #oldu
            orderDict[qui] = node1.data
            otherDict[qui]=node1.data
        else:
            while(node1.data != node2.data):
                node1= node1.parent
                node2 = node2.parent
             #zaten oldu
            orderDict[qui] = node1.data
            otherDict[qui]=node1.data


    elif (node2.level <= node1.level):
        while(node2.level != node1.level):

            if node1.parent == None:
                orderDict[qui] = node1.data
                otherDict[qui]=node1.data
            node1 = node1.parent

        if(node1.data == node2.data):
            #oldu
            orderDict[qui] = node1.data
            otherDict[qui]=node1.data

        while(node1.data != node2.data):
            node2= node2.parent
            node1 = node1.parent
            #zaten oldu
        orderDict[qui] = node1.data
        otherDict[qui]=node1.data

def cost_LM_Modified(raw_dataset, anonymized_dataset,
            DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = raw_dataset
    anonymized_dataset =anonymized_dataset
    DGHs = read_DGHs(DGH_folder)

    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
           and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    #TODO: complete this function.
    sens,quis=returnQIS(raw_dataset[0],DGH_folder)
    clusterTree=DGHs
    D = len(raw_dataset)
    penalty= 0
    for i in range(D):
        for qui in quis:
            if countNumberOfLeaves(anonymized_dataset[i][qui],DGHs,qui) !=0:
                penalty+=(countNumberOfLeaves(anonymized_dataset[i][qui],DGHs,qui) -1)/((countNumberOfLeavesRoot(DGHs,qui)-1)*len(quis))
    return penalty

def generilizeCluster(clusters ,clusterTree,quis,sen):
    i=0
    for cluster in clusters:

        genarilize1Cluster(cluster,clusterTree,quis,sen)

def genarilize1Cluster(cluster ,clusterTree,quis,sens):
    while(checkIfEquealCluster(cluster,sens,quis) != True):
        for qui in quis:
            i = 0
            for orderDict in cluster:
                i = i+1
                for otherDict in cluster[i:]:
                    #print(qui)
                    #print(orderDict[qui])
                    #print(otherDict[qui])
                    if orderDict[qui] != otherDict[qui] :
                        generilizeTheAttribute(clusterTree,qui,orderDict,otherDict)



def checkIfEquealCluster(clus,sens,quis):
    for qui in quis:
        i = 0
        for toCompare in clus:
            i= i+1
            for compared in clus[i:]:
                value1 = toCompare[qui]
                value2 = compared[qui]
                #print(qui)
                #print(value1)
                #print(value2)
                if value1 != value2:
                    return False
    return True

def checkIfEqueal(clusters,sens):
    for clus in clusters:
        i = 0
        for toCompare in clus:
            i= i+1
            for compared in clus[i:]:
                for key1 in toCompare:
                    value1 = toCompare[key1]
                    value2 = compared[key1]
                    if not(key1 == sens):
                        if value1 != value2:
                            return False
    return True

def returnQIS(datasetListHeader,DGH_folder):
    qui = []
    sensitive = []

    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
         qui.append( DGH_file.split("/")[1].split(".txt")[0])

    for data in  datasetListHeader:
        if data not in qui:
            sensitive.append(data)
    return sensitive, qui

def countNumberOfLeaves(node,tree,qui):
    treeOfQui= tree[qui]
    node_=treeOfQui.BFS(node)
    numbofChild=0
    visited = []
    queue = []
    queue.append(node_)
    visited.append(node_.data)
    while queue:
        s = queue.pop(0)
        for children in s.children:
            if not (children.data  in visited):
                if(len(children.children)==0):
                    numbofChild+=1
                queue.append(children)
                visited.append(children.data)
    return numbofChild

def countNumberOfLeavesRoot(tree,qui):
    node_=tree[qui]
    numbofChild=0
    visited = []
    queue = []
    queue.append(node_)
    visited.append(node_.data)
    while queue:
        s = queue.pop(0)
        for children in s.children:
            if not (children.data  in visited):
                if(len(children.children)==0):
                    numbofChild+=1
                queue.append(children)
                visited.append(children.data)
    return numbofChild


def findEQClusters(dataset_,quis):
    dataset = copy.deepcopy(dataset_)
    cluster=[]
    for data in  dataset :
        data['found'] = False
    s = 0
    for data in dataset:
        s+=1
        eqClus= []
        eqClus.append(data)
        if data['found'] == False:
            for data_after in dataset[s:]:
                if (data_after['found'] == False):
                    limit = len(quis)
                    quiwise = 0
                    for qui in quis:
                        if(data[qui] == data_after[qui]):
                            quiwise +=1
                    if quiwise ==limit:
                        eqClus.append(data_after)
                        data_after['found'] = True
                        data['found']=True
            if(len(eqClus)!= 1):
                cluster.append(eqClus)
    for node in dataset:
        if node['found'] ==False:
            cluster.append([node])
    return  cluster


def checkKanonim (k,dataset,quis):
    cls= findEQClusters(dataset,quis)

    if k <= min(map(len, cls)):
        return True
    else:
        return False


def checkKanonimFile (k,dataset,quis):
    dt = read_dataset(dataset)
    cls= findEQClusters(dt,quis)
    if k <= min(map(len, cls)):
        return True
    else:
        return False






def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file)

    return DGHs


##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    sens,quis=returnQIS(raw_dataset[0],DGH_folder)
    clusterTree=DGHs
    D = len(raw_dataset)
    penalty= 0
    for i in range(D):
        for qui in quis:
            treeOfQui=clusterTree[qui]
            node1=treeOfQui.BFS(raw_dataset[i][qui])
            node2=treeOfQui.BFS(anonymized_dataset[i][qui])
            penalty += np.abs(node2.level-node1.level)
    return penalty;


def cost_LM_of_data(real,anon,DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = real
    anonymized_dataset = anon
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
           and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)
    #TODO: complete this function.
    sens,quis=returnQIS(raw_dataset[0],DGH_folder)
    clusterTree=DGHs
    D = len(raw_dataset)
    penalty= 0
    for i in range(D):
        for qui in quis:
            if countNumberOfLeaves(anonymized_dataset[i][qui],DGHs,qui) !=0:
                penalty+=(countNumberOfLeaves(anonymized_dataset[i][qui],DGHs,qui) -1)/((countNumberOfLeavesRoot(DGHs,qui)-1)*len(quis))
    return penalty


def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)
    #TODO: complete this function.
    sens,quis=returnQIS(raw_dataset[0],DGH_folder)
    clusterTree=DGHs
    D = len(raw_dataset)
    penalty= 0
    for i in range(D):
        for qui in quis:
            if countNumberOfLeaves(anonymized_dataset[i][qui],DGHs,qui) !=0:
                penalty+=(countNumberOfLeaves(anonymized_dataset[i][qui],DGHs,qui) -1)/((countNumberOfLeavesRoot(DGHs,qui)-1)*len(quis))
    return penalty
            

def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str, s: int):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
        s (int): seed of the randomization function
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    sens,qui=returnQIS(raw_dataset[0],DGH_folder)

    for i in range(len(raw_dataset)): ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s) ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize


    D = len(raw_dataset)
    
    #TODO: START WRITING YOUR CODE HERE. Do not modify code in this function above this line.
    # Store your results in the list named "clusters". 
    # Order of the clusters is important. First cluster should be the first EC, second cluster second EC, ...

    clusters = [raw_dataset[i:i + k] for i in range(0, D, k)]
    if D%k != 0 :
        ele = clusters[len(clusters)-1]
        clusters = clusters[:-1]
        clusters[len(clusters)-1]=np.concatenate((clusters[len(clusters)-1], ele))

    generilizeCluster(clusters ,DGHs,qui,sens)



    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D
    for cluster in clusters:        #restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)



def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    sen,quis=returnQIS(raw_dataset[0],DGH_folder)
    mark = {}

    for i in range(len(raw_dataset)): ##set indexing to not lose original places of records
        mark[i] = True
    unmarked = len(raw_dataset)

    if k == 1:
        write_dataset(raw_dataset, output_file)
    else:
        while(2*k <= unmarked):


            dt= raw_dataset[next(iter(mark))]
            del mark[next(iter(mark))]
            unmarked-=1

            #for idx, data in enumerate(raw_dataset_):
            #    dt = data
            #    if dt['marked'] == False:
            #        dt['marked']=True

            #        dt = raw_dataset[idx]
            #        break
            realCluster = []
            realCluster.append(dt)

            cost_arr = np.array([99999999999.0]*(k-1))
            best_ind_array = np.array([-1]*(k-1))

            for idx in mark:
                cl_before = [copy.deepcopy(dt),copy.deepcopy(raw_dataset[idx])]
                cluster = [copy.deepcopy(dt),copy.deepcopy(raw_dataset[idx])]
                genarilize1Cluster(cluster,DGHs,quis,sen)
                arg=np.argmax(cost_arr)
                costt=cost_LM_of_data(cl_before,cluster,DGH_folder)
                if cost_arr[arg] > float(costt):
                    best_ind_array[arg] = idx
                    cost_arr[arg]= costt

            for indd in best_ind_array:

                del mark[indd]
                unmarked-=1
                realCluster.append(raw_dataset[indd])
            genarilize1Cluster(realCluster,DGHs,quis,sen)

        last_clus = []
        if(2*k > unmarked):
            for idx in mark:
                last_clus.append(raw_dataset[idx])
        genarilize1Cluster(last_clus,DGHs,quis,sen)
        write_dataset(raw_dataset, output_file)



        #last_clus = []
        #if(2*k > unmarked):
        #    for idx,datta in enumerate(raw_dataset_):
        #        if datta['marked'] == False:
        #            last_clus.append(raw_dataset[idx])
        #genarilize1Cluster(last_clus,DGHs,quis,sen)
        #write_dataset(raw_dataset, output_file)




            #for idx,datta in enumerate(raw_dataset_):
            #    if datta['marked'] == False:

            #        cl_before = [copy.deepcopy(dt),copy.deepcopy(raw_dataset[idx])]
            #        cluster = [copy.deepcopy(dt),copy.deepcopy(raw_dataset[idx])]
            #        genarilize1Cluster(cluster,DGHs,quis,sen)

            #        arg=np.argmax(cost_arr)
            #        costt=cost_LM_of_data(cl_before,cluster,DGH_folder)

            #        if cost_arr[arg] > float(costt):

            #            best_ind_array[arg] = idx
            #            cost_arr[arg]= costt









# Finally, write dataset to a file
    #write_dataset(anonymized_dataset, output_file)


def bottomup_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Bottom up-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)



    DGHs = read_DGHs(DGH_folder)
    #TODO: complete this function.
    sens,quis=returnQIS(raw_dataset[0],DGH_folder)
    kanonims = []

    lattice= []
    lat_dict = {}
    qui_n = len(quis)
    gene_array= [0]*qui_n

    lat_dict[tuple(gene_array)] = True


    lattice_level=0
    l = Lattice()
    l.parent = None
    l.data = copy.deepcopy(raw_dataset)
    l.level = lattice_level
    l.generalize_array = gene_array
    l.qui = quis
    lattice.append(l)
    flag = True

    while(True):

        #check current level
        if flag:
            for lat in lattice:
                if lat.level == lattice_level:
                    if(True == checkKanonim(k,lat.data,quis)):
                        kanonims.append(lat.data)
            flag = False

        if(len(kanonims)!=0):
            min=999999999999999999
            returned= None
            for kanonim in kanonims:
                cost= cost_LM_Modified(raw_dataset, kanonim,  DGH_folder)
                if cost < min:
                    returned = kanonim
                    min = cost
            write_dataset(returned,output_file)
            break
        lattice_level +=1
        toDrop = 0


        for lat in lattice:
            if lat.level == lattice_level-1:
                toDrop+=1
                modify = [elem for elem in lat.generalize_array]
                idx= 0
                for qui in quis:
                    modify2 = [elem for elem in modify]
                    dont = True
                    modify2[idx] =1+ modify2[idx]

                    if  tuple(modify2) in lat_dict:
                        dont = False


                    if dont:
                        min,gene= generilize(lat.data,DGHs,qui)
                        if min != 0:
                            if(True == checkKanonim(k,lat.data,quis)):
                                flag = True
                            l = Lattice()

                            l.parent = lat
                            l.data = gene
                            l.level = lattice_level
                            l.generalize_array = modify2
                            l.qui = quis
                            lat_dict[tuple(modify2)] = True
                            lattice.append(l)
                    idx+=1
        del lattice[:toDrop]







  #  else:


    # Finally, write dataset to a file
    #write_dataset(anonymized_dataset, output_file)



# Command line argument handling and calling of respective anonymizer:

if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
    print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'bottomup']:
    print("Invalid algorithm.")
    sys.exit(2)

start_time = datetime.datetime.now() ##
print(start_time) ##

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer");
if function == random_anonymizer:
    if len(sys.argv) < 7:
        print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
        print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
        sys.exit(1)
        
    seed = int(sys.argv[6])
    function(raw_file, dgh_path, k, anonymized_file, seed)
else:    
    function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print (f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")

end_time = datetime.datetime.now() ##
print(end_time) ##
print(end_time - start_time)  ##

# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300 5

