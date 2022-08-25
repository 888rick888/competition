from collections import deque
import random
import numpy as np
random.seed(1)
np.random.seed(1)


BUFF_SIZE = 1000000

class SumTree(object):
    data_pointer = 0
    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1) # capacity-1是parent nodes , capacity是leaves
        self.data = np.zeros(capacity, dtype=object)

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        print("-=-=-------------------===========",self.data_pointer)
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0
            
    def update(self, tree_idx, p):  #p就是优先级
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):  #采样环节，代表经验数据的叶子只在树的最底层，采样的时候，优先级（p值）越高的越容易被采样
        parent_idx = 0
        while True:     
            cl_idx = 2 * parent_idx + 1       
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        
                leaf_idx = parent_idx
                break
            else:     
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  
    
class Memory(object):
    def __init__(self, capacity,per, epsilon=0.001, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, td_error_upper=1.):
        self.capacity = capacity
        self.epsilon = epsilon # small value to avoid zero priority
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.td_error_upper = td_error_upper

        self.tree = SumTree(capacity)
        self.num = 0

        self.per = per
        if not per:
            self.buffer = deque()

    def store(self, memory):
        '''memory: [s_t, action[0], reward, s_t1, done]'''
        self.num +=1
        if self.per:
            max_p = max(self.tree.tree[-self.tree.capacity:])
            if max_p == 0. : max_p = self.td_error_upper
            self.tree.add(max_p, memory)

        else:self.buffer.append(memory)
        


    def sample(self, n):
        if self.per:
            batch_idx, batch_data, ISWeights = np.empty((n,), dtype=int), np.empty((n), dtype=object), np.empty(n)
            
            pri_seg = self.tree.total_p / n     #采样分段
            self.beta = min(1., self.beta+self.beta_increment_per_sampling)     #beta为1时抵消Prioritized replay的影响
            min_prob = max(min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p, 0.0001)
            for i in range(n):
                a, b =pri_seg * i, pri_seg * (i+1)
                v = random.uniform(a, b)
                idx, p, data = self.tree.get_leaf(v)
                # print(a, b, v, idx, p, data)
                prob = p / self.tree.total_p
                ISWeights[i], batch_idx[i], batch_data[i] = np.power(prob/min_prob, -self.beta), idx, data
            
            return batch_idx, batch_data, ISWeights

        else:
            barch_data = random.sample(self.buffer, n)
            return 1, barch_data, 1

    def batch_update(self, tree_idx, td_errors):
        for (ti, td) in zip(tree_idx, td_errors):
            td = min(self.td_error_upper, td+self.epsilon)
            p = np.power(td, self.alpha)
            self.tree.update(ti, p)

        
    
    @property
    def pointer(self):
        return self.tree.data_pointer