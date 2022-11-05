from HHCART import HHCARTNode, HouseHolderCART

class Node_container:
    '''
    A wrapper object of HHCARTNode to let each node know its parent.
    '''
    def __init__(self, node, depth: int, lc = None, rc = None, p = None):
        '''
        node: An HHCARTNode
        p: parent
        lc: left child (should be another Node_container object, instead of an HHCARTNode)
        rc: right child
        '''
        self.node = node
        self.depth = depth
        self.lc = lc
        self.rc = rc
        self.p = p

class NC_tree:
    '''
    A tree whose nodes are of type `Node_container`. Intended for visualization purposes
    '''
    def __init__(self, Htree: HouseHolderCART):
        '''
        Populate an NC_tree by wrapping an `HouseHolderCART` tree.
        '''
        assert Htree._root is not None
        self.root = Node_container(node = Htree._root, depth = Htree._root.depth)
        Queue = [self.root]  # list implementation of FIFO queue. Slow, might revise in future iteration

        # Construct a tree by BFS.
        while len(Queue) > 0:
            curr = Queue.pop(0)
            node = curr.node
            if node._left_child is None and node._right_child is None: # curr.node is a leaf node
                continue
            if node._left_child is not None:
                new_lc = Node_container(node = node._left_child, depth = node._left_child.depth, p = curr)
                curr.lc = new_lc
                Queue.append(new_lc)
            if node._right_child is not None:
                new_rc = Node_container(node = node._right_child, depth = node._right_child.depth, p = curr)
                curr.rc = new_rc
                Queue.append(new_rc)
            