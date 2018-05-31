##########################################################################################
####
####             Connected-Component labeling Two-Pass algorithm  (8-connectivity)
#### 
##########################################################################################
####
####             Cluster elevation matrix - iso-elevation regions 
####            
##########################################################################################

import numpy as np
## based on http://en.wikipedia.org/wiki/Connected-component_labeling

## based on
## Connected Component Labeling Algorithms for Gray-Scale Images 
## and Evaluation of Performance using Digital Mammograms, R. Dharshana Yapa
## and  K. Harada

#######################################################################################################

class disjointSet:

    def __init__(self, n):
        self.parent = [0]*n
        self.rank = [0]*n

    def makeSet(self,x):
        self.parent[x] = x
        self.rank[x] = 0

    def union(self,x,y):
        xRoot = self.find(x)
        yRoot = self.find(y)
        
        if(xRoot == yRoot):
            return

        if(self.rank[xRoot] < self.rank[yRoot]):
            self.parent[xRoot] = yRoot
        elif(self.rank[xRoot] > self.rank[yRoot]):
            self.parent[yRoot] = xRoot
        else:
            self.parent[yRoot] = xRoot
            self.rank[xRoot] += 1
            
    def find(self, x):
        if (self.parent[x] != x):
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

class disjointRegions(disjointSet):

    def __init__(self, n):
        self.parent = [0]*n
        self.rank = [0]*n
        self.cell = [0]*n # top left hand corner cell for regions
        self.utils = 0 # effective number of new created label 
        self.neighbors = [[]]*n

    def makeSet(self,x,i,j):
        self.parent[x] = x
        self.rank[x] = 0
        self.cell[x] = (i,j)
        self.utils += 1

    def union(self,x,y):
        xRoot,xCell = self.find(x)
        yRoot,yCell = self.find(y)
        if(xRoot == yRoot):
            return

        if(self.rank[xRoot] < self.rank[yRoot]):
            self.parent[xRoot]  = yRoot 
        elif(self.rank[xRoot] > self.rank[yRoot]):
            self.parent[yRoot]  = xRoot 
        else:
            self.parent[yRoot] = xRoot
            self.rank[xRoot] += 1

    def find(self, x):
        if (self.parent[x] != x):
            self.parent[x] = self.find(self.parent[x])[0]

        return self.parent[x], self.cell[self.parent[x]]  # return representative for region: label, cell

    def topDown(self,x):
        '''
        Return all labels whose parent is x's parent
        '''
        tDlabel  = []
        
        for i in range(self.utils):
            #print ' i ' + str(i) + ' self.find(i)[0] ' + str(self.find(i)[0]) + ' x ' + str(x) + ' ' + str(self.find(x)[0])

            if(self.find(i)[0] == self.find(x)[0]):
                
                tDlabel.append(i)
                 
        return tDlabel

##########################################################################################################

class CC_lab:

    def __init__(self,mat):
        self.labels = []
        self.forest = 0
        self.mat = mat

        
            
    def connectedComponentLabel(self,n_clusters=0):
            '''
            Label regions belonging to the same color level
            Algorithm uses disjoint-set data structure (equivalence table) and
            it is based on
            A Run-Based Two-Scan Labeling Algorithm, L. H, Y. Chao, K. Suzuki, IEEE
            '''
            N, M = len(self.mat), len(self.mat[0])
                 
            if(n_clusters==0):
                n_clusters = N * M

            self.labels = sameDimZeroMatrix(self.mat) # build label matrix with appropriate dimensions
            label = 0 # next label to be assigned

            self.forest = disjointRegions(n_clusters) # forest to record equivalences
            
            for i in range(N):
                
                for j in range(M):

                    neighbors = self.connectedNeighbors(i,j) # neighbors with same value
                    
                    #print 'neighbors ' + str(neighbors) +' for pixel at ' + str(i) + ', '+ str(j)
                    
                    if (neighbors == [[],[],[],[]]): # no neighbors at all
                        #print 'case no neighbors at all'
                        self.labels[i][j] = label # new label
                        self.forest.makeSet(label,i,j)
                        
                        label += 1
                            
                    else:
                         
                        ##find minimum neighbor
                        lab = [] # labels for neighbor pixels that are available

                        for pix in range(4):
                            if(neighbors[pix] != []): # some neighbor in this direction
                                x,y = neighbors[pix][0][0], neighbors[pix][0][1]
                                lab.append(self.labels[x][y])
                                
                        if(len(lab) > 1): # at least two neighbors from north, west, north-west, north-east
                            #print 'more than 1 neighbors'
                            self.labels[i][j] = min(lab)
                            
                            for l in range(len(lab)-1):
                                
                                self.forest.union(lab[l], lab[l+1])  # union labels, maybe they are different      
                            
                        else:
                            #print 'only 1 neighbors'
                            if(neighbors[0] != []):
                                #print 'adopted west'
                                self.labels[i][j] = self.labels[neighbors[0][0][0]][neighbors[0][0][1]]
     
                            elif(neighbors[1] != []): # some north neighbor only
                                #print 'adopted north' 
                                self.labels[i][j] = self.labels[neighbors[1][0][0]][neighbors[1][0][1]]

                            elif(neighbors[2] != []): # some north-west neighbor only
                                #print 'adopted north-west' 
                                self.labels[i][j] = self.labels[neighbors[2][0][0]][neighbors[2][0][1]]
              
                            elif(neighbors[3] != []): # some north-east neighbor only
                                #print 'adopted north-east' 
                                self.labels[i][j] = self.labels[neighbors[3][0][0]][neighbors[3][0][1]] 

                    if(neighbors[0] == [] and i > 0):
                        self.updateNeighbors(i,j,i-1,j) 

                    if(neighbors[1] == [] and j > 0):
                        self.updateNeighbors(i,j,i,j-1)

                    if(neighbors[2] == [] and i > 0 and j > 0):
                        self.updateNeighbors(i,j,i-1,j-1)

                    if(neighbors[3] == [] and i > 0 and j < M-1):
                        self.updateNeighbors(i,j,i-1,j+1)

            # second pass
            for i in range(N):

                for j in range(M):

                    self.labels[i][j] = self.forest.find(self.labels[i][j])[0]



    def updateNeighbors(self, i, j, k, l):
        '''
        Region labeled with label(i,j) and region labeled with label(k,l) are neighbor, update
        self.forest.neighbors twice
        '''
        self.forest.neighbors[self.labels[i][j]] = np.unique([x for x in self.forest.neighbors[self.labels[i][j]]] + [self.labels[k][l]])
        self.forest.neighbors[self.labels[k][l]] = np.unique([x for x in self.forest.neighbors[self.labels[k][l]]] + [self.labels[i][j]])


              
    def neighborRegions(self,labelij,mat):
        '''
        Return neighbor regions for the region with label labelij
        '''
        neighbors = []
        R,C = len(mat), len(mat[0])
 
        parentij = self.forest.find(labelij)[0] # representative for the region which the cell i,j belongs to
        areas = self.forest.topDown(parentij)  # list of areas in the region which the cell i,j belongs to

        for area in areas:
            
            neighboor_ = self.forest.neighbors[area]

            for k in range(len(neighboor_)):
                neighboor_label = self.forest.find(neighboor_[k])[0]
                if(neighboor_label != labelij):
                    neighbors.append(neighboor_label)
             
        return np.unique(neighbors)




    def labelToElevation(self, label_i):
        '''
        From label, retrieve elevation in matrix self.mat
        '''
        x,y = self.forest.cell[label_i]
        return self.mat[x][y]

        
        
    def connectedNeighbors(self,i, j): 
        '''
        Return coordinates for neighbors of pixel i, j that have same value
        as pixel i, j (8-connectivity)
        '''
        neighbors = []
        for z in range(4):
            neighbors.append([])

        N, M = len(self.mat), len(self.mat[0])
        
        if(i >= N or j >= M or i < 0 or j < 0): # exceed dimensions
            return []

        val = self.mat[i][j]
        
        if(i == 0 and j == 0): ## top left-hand corner, no neighbors
            return neighbors

        # i or j is not zero
        if(j > 0):
            
            if(val == self.mat[i][j-1]):
                neighbors[0].append((i, j-1)) ## west
  
        if(i > 0):
            
            if(val == self.mat[i-1][j]):
                neighbors[1].append((i-1, j)) ## north

        if(i > 0 and j > 0): 
            if(val == self.mat[i-1][j-1]):
                neighbors[2].append((i-1, j-1)) ## north-west
        
        if(i > 0 and j < M-1): 
            if(val == self.mat[i-1][j+1]):
                neighbors[3].append((i-1, j+1)) ## north-east
                
        return neighbors


    def matrixFromLabelsList(self, region_values, N, M):
        '''
        Input array region_values contains values for each region in self.labels.
        Returns the matrix with pixel i,j colored with the value for the region which this pixel belongs to.
        '''
        toPlot = []
    
        for i in range(N):
            tmp = []
            for j in range(M):
                tmp.append(region_values[self.forest.find(self.labels[i][j])[0]-1]) # watershed label for the region (i,j) belongs to
            toPlot.append([x for x in tmp])

        return toPlot 

     
def sameDimZeroMatrix(mat):
    '''
    Create matrix with same dimensions as mat with zeros
    '''
    zeros = []
    
    for i in range(len(mat)):
        tmp = []
        for j in range(len(mat[i])):
            tmp.append(0)
        zeros.append([x for x in tmp])
        
    return zeros