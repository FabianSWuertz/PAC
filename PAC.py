import numpy as np


### Initialize class PAC
class ProteinAllignmentCheck:
    ### Input is the size of the box
    def __init__(self, size):
        ### Error handling
        assert size > 0, "Box size must be positive"

        ### Initializing variables
        self.size = size
        self.n_boxes_xyz = None
        self.min_xyz = None
        self.n = None
        self.align_1 = None
        self.align_2 = None


    ### Calculates the amount of boxes needed in each direction
    def boxes(self, align_1, align_2):
        ### Error handling
        assert (align_1.shape[0] == 3 or align_1.shape[1] == 3) and len(align_1.shape) == 2, "Shape must be (3, N) or (N, 3)"
        assert align_1.shape == align_2.shape, "Arrays must have the same shape"

        ### Saved for later use
        self.align_1 = align_1
        self.align_2 = align_2  
        
        ### Makes sure the strands have the correct shape
        points_tot = np.hstack([align_1, align_2])

        ### Finds minimum coordinates along each axis
        min_x = np.min((points_tot[0]))
        min_y = np.min((points_tot[1]))
        min_z = np.min((points_tot[2]))
        self.min_xyz = np.array([min_x, min_y, min_z])

        ### Finds maximum coordinates along each axis
        max_x = np.max((points_tot[0]))
        max_y = np.max((points_tot[1]))
        max_z = np.max((points_tot[2]))
        max_xyz = np.array([max_x, max_y, max_z])

        ### Calculation of number of boxes needed in each direction
        self.n_boxes_xyz = np.ceil(((max_xyz) - (self.min_xyz)) / self.size).astype(int)

        return self.n_boxes_xyz


    ### Calculates the box index of which a point belongs to
    def box_idx(self, point):
        ### Error handling
        assert len(point) == 3, "Enter valid x, y, z coordinates"

        box_idx = ((point - self.min_xyz) // self.size).astype(int)

        return box_idx
    
    
    ### Calculates which boxes each point traverses and at what time
    def box_checker(self, align_1, align_2):
        ### Error handling
        assert (align_1.shape[0] == 3 or align_1.shape[1] == 3) and len(align_1.shape) == 2, "Shape must be (3, N) or (N, 3)"
        assert align_1.shape == align_2.shape, "Arrays must have the same shape"

        if align_1.shape[1] == 3:
            align_1 = align_1.T
            align_2 = align_2.T

        n_boxes_xyz = self.boxes(align_1, align_2)

        ### Inizialisze variables
        self.n = align_1.shape[1]
        boxes_traversed = [[] for _ in range(self.n)]
        t_intervals = []
        hashmap = {}
        hashmap_t = {}
        min_xyz = self.min_xyz
        size = self.size


        ### Looping over the N points
        for i in range(self.n):
            ### Access the points
            x1 = align_1[:, i]
            x2 = align_2[:, i]

            hashmap_t[i] = []

            ### Finds their box indices
            init_box1 = self.box_idx(x1)
            init_box2 = self.box_idx(x2)
            
            ### Box indices as tuples to enable hashing
            init_box1_tup = tuple(init_box1)
            init_box2_tup = tuple(init_box2)

            ### Initialize list and set to hold traversed boxes
            boxes_point = [init_box1_tup]
            boxes_point_s = {init_box1_tup}

            ### Direction of movement and sign of movement
            direc = x2 - x1            
            step = np.sign(direc).astype(int)
    
            ### Calculates the time t for when the next boundary is crossed along each axis
            t = (min_xyz + init_box1 * size + (step > 0) * size - x1) / (x2 - x1 + 1e-10)
            ### Calculates the value of which t should be incremented
            t_incr = size / (np.abs(direc) + 1e-10)
            
            ### Initialize variable and boolean value
            a = True
            t_cur = 0

            ### While loop that finds boxes traversed as long as t is valid
            while t_cur <= 1:
                ### Finds minimum t value
                axis = np.argmin(t)
                t_dir = t[axis]
    
                ### Makes sure t value is not bigger than 1
                if t_dir > 1:
                    break
                
                ### Append the current t interval and its index to t intervals
                hashmap_t[i].append([t_cur, t_dir])

                ### Updates the current t value
                t_cur = t_dir

                ### Finds the next traversed box
                init_box1[axis] += step[axis]
                init_box1_tup = tuple(init_box1)

                ### Added to traversed boxes
                if init_box1_tup not in boxes_point_s:
                    boxes_point.append(init_box1_tup)
                    boxes_point_s.add(init_box1_tup)

                ### If the current box is equal to the final box in the traversal are equal, the loop breaks
                if (init_box1 == init_box2).all():
                    a = False
                    break
                
                ### Increment the t value along the axis of movement
                t[axis] += t_incr[axis]

            ### Append the final t interval
            hashmap_t[i].append([t_dir, 1])

            ### Add the current box to the traversed boxes if it is not already equal to the final box
            if a and init_box1_tup not in boxes_point_s:   
                boxes_point.append(init_box2_tup)  
                boxes_point_s.add(init_box2_tup)

            ### Adds the traversed boxes to a hashmap with the points as keys
            for index, box in enumerate(boxes_point):
                if box not in hashmap:   
                    hashmap[box] = []
                hashmap[box].append([i, index])
    
            ### Boxes traversed as a NumPy array
            boxes_traversed[i] = np.array(boxes_point)

        ### Prepares the hashmap and traversed boxes for later use
        self.hashmap = hashmap
        self.hashmap_t = hashmap_t
        self.boxes_nonpad = boxes_traversed
        
        ### Makes the t intervals into a NumPy array
        t_intervals = np.array(t_intervals)
    
        return boxes_traversed, hashmap_t
    
        
    ### Calculates if there is an intersection
    def collision_checker(self, dist, align_1, align_2, time_specific=None, t_star=None):
        ### Error handling
        assert dist >= 0, "Distance cannot be negative"
        assert (align_1.shape[0] == 3 or align_1.shape[1] == 3) and len(align_1.shape) == 2, "Shape must be (3, N) or (N, 3)"
        assert align_1.shape == align_2.shape, "Arrays must have the same shape"

        if time_specific:
            assert t_star!=None, "Pls enter valid t_star"

        ### Makes sure the shape is correct
        if align_1.shape[1] == 3:
            align_1 = align_1.T
            align_2 = align_2.T

        ### Calls the box checker function
        boxes, t = self.box_checker(align_1, align_2)
        size = self.size   
        
        ### Finds minimum and maximum coordinates for the pattern
        min_coor = np.floor((-dist) / size)
        max_coor = np.ceil((dist) / size)
        max_bc = np.ceil(dist / size)

        ### The conditions for how big the pattern should be
        conditions = np.array([dist <= (max_bc - 1) * np.sqrt(3)*size, dist <= (max_bc - 1 ) * np.sqrt(2)*size])

        ### Calculation of the pattern
        if dist <= size or np.all(conditions == False):
            pattern = [(x, y, z)
            for x in range(int(min_coor), int(max_coor) + 1)
            for y in range(int(min_coor), int(max_coor) + 1)
            for z in range(int(min_coor), int(max_coor) + 1)
            ]
        
        elif np.all(conditions):
            pattern = [(x, y, z)
            for x in range(int(min_coor), int(max_coor) + 1)
            for y in range(int(min_coor), int(max_coor) + 1)
            for z in range(int(min_coor), int(max_coor) + 1)
            if (x, y, z) not in {
                ( max_bc, max_bc, max_bc), (-max_bc, -max_bc, -max_bc), ( max_bc, -max_bc, -max_bc), (-max_bc,  max_bc,  max_bc),
                (-max_bc, -max_bc, max_bc), (-max_bc, max_bc, -max_bc), (max_bc, -max_bc, max_bc), (max_bc, max_bc, -max_bc),
                (1, max_bc, max_bc), (1, max_bc, -max_bc), (1, -max_bc, max_bc), (1, -max_bc, -max_bc), (-1, max_bc, max_bc), (-1, max_bc, -max_bc),
                (-1, -max_bc, max_bc), (-1, -max_bc, -max_bc), (max_bc, 1, max_bc), (max_bc, 1, -max_bc), (-max_bc, 1, max_bc), (-max_bc, 1, -max_bc),
                (max_bc, -1, max_bc), (max_bc, -1, -max_bc), (-max_bc, -1, max_bc), (-max_bc, -1, -max_bc), (max_bc, max_bc, 1), (max_bc, -max_bc, 1), 
                (-max_bc, max_bc, 1), (-max_bc, -max_bc, 1), (max_bc, max_bc, -1), (max_bc, -max_bc, -1), (-max_bc, max_bc, -1), (-max_bc, -max_bc, -1), 
                
                (0, max_bc, max_bc), (0, max_bc, -max_bc), (0, -max_bc, max_bc), (0, -max_bc, -max_bc), (-1, max_bc, max_bc), (-1, max_bc, -max_bc),
                (max_bc, 0, max_bc), (max_bc, 0, -max_bc), (-max_bc, 0, max_bc), (-max_bc, 0, -max_bc), 
                (max_bc, max_bc, 0), (max_bc, -max_bc, 0), (-max_bc, max_bc, 0), (-max_bc, -max_bc, 0)
            }
            ]

        elif conditions[0]:
            pattern = [(x, y, z)
            for x in range(int(min_coor), int(max_coor) + 1)
            for y in range(int(min_coor), int(max_coor) + 1)
            for z in range(int(min_coor), int(max_coor) + 1)
            if (x, y, z) not in {
                ( max_bc, max_bc, max_bc), (-max_bc, -max_bc, -max_bc), ( max_bc, -max_bc, -max_bc), (-max_bc,  max_bc,  max_bc),
                (-max_bc, -max_bc, max_bc), (-max_bc, max_bc, -max_bc), (max_bc, -max_bc, max_bc), (max_bc, max_bc, -max_bc)
            }
            ]

        ### Makes the pattern a NumPy array
        pattern = np.array(pattern)

        ### Initializes a sparse matrix of size NxN
        sparse_mat = np.zeros((self.n, self.n))

        ### Creating variables for faster lookup
        hashmap = self.hashmap
        boxes_nonpad = self.boxes_nonpad
        n_boxes_xyz = self.n_boxes_xyz
        hashmap_t = self.hashmap_t
        

        ### Loops over the boxes each point traverses
        for idx in range(self.n):
            ### The boxes traversed by the point
            traversed = boxes_nonpad[idx]

            ### The boxes to check based on the pattern and traversed box
            box_check_v2 = traversed[:, None, :] + pattern[None, :, :]

            ### Creates a mask, so only valid boxes are checked
            mask = np.all((box_check_v2 >= 0) & (box_check_v2 <= n_boxes_xyz), axis=2)
            box_check_v2 = box_check_v2[mask]    

            ### Changes the boxes to tuples to enable hashing
            box_check_v2 = [tuple(i) for i in box_check_v2]

            ### Initializes empty array to contain points in the checked boxes
            points_all = []

            ### Loops over checked boxes and utilizes hashmap to find points in any of the boxes to any time
            if not time_specific:
                for box_check in box_check_v2:
                    if box_check in hashmap:
                        for p in hashmap[box_check]:
                            points_all.append(p[0])

            ### At a specific time
            elif time_specific:
                for box_check in box_check_v2:
                    if box_check in hashmap:
                        for p in hashmap[box_check]:
                            ### Finds which number the box is in the traversal
                            idx = p[1]
                            
                            ### Finds t_values
                            t_vals = hashmap_t[p[0]]

                            ### Finds the one for the correct box
                            t_vals = t_vals[idx]

                            ### Finds start and end of t interval
                            t1 = t_vals[0]
                            t2 = t_vals[1]

                            ### Makes sure moving point is in the box a t_star
                            if t1 <= t_star <= t2:
                                points_all.append(p[0])
                

            ### Flags the found points by setting corresponding index in the sparse matrix to 1
            sparse_mat[idx, points_all] = 1
        
        ### Fill diagonal with zeros, since a point should not be compared with itself
        np.fill_diagonal(sparse_mat, 0)

        ### Finds the indices of flagged points
        i, j = np.where(sparse_mat == 1)

        ### Calculates vector for the calculation of minimum distance between two points during their interpolation
        p0_ij = align_1[:, j] - align_1[:, i] 
        p1_ij = align_2[:, j] - align_2[:, i]
        
        ### Takes the norm
        a = np.linalg.norm(p0_ij, axis=0) 
        b = np.linalg.norm(p1_ij, axis=0)

        ### Finds the dot product
        dot_prod = np.sum(p0_ij * p1_ij, axis=0)

        ### Finds t value at the minimum distance
        t_min = (a*a - dot_prod) / (a*a + b*b - 2*dot_prod)

        ### Clips it to the appropriate interval
        t_min = np.clip(t_min, 0, 1)

        ### Finds the minimum distance
        m = np.sqrt((1 - t_min)**2*a**2 + t_min**2*b**2 + 2*t_min*(1-t_min)*dot_prod)
        
        ### Finds the indices of the minimum distances below the maximum allowed distance
        m_idx = np.where(m <= dist)[0]
        
        ### Finds the index of the points clashing
        i_int = i[m_idx]
        j_int = j[m_idx]
        
        ### Makes a list of points steric clashes
        steric_cl = list(zip(i_int, j_int))

        self.align_1 = align_1
        self.align_2 = align_2
        self.t = t
        self.sparse_mat = sparse_mat
        self.t_min = t_min
        self.m = m
        self.hashmap = hashmap

        ### Changes the list to only contain unique steric clashes
        steric_cl = np.unique(np.array([sorted(i) for i in steric_cl]), axis=0)
    
        return steric_cl, t
    


   