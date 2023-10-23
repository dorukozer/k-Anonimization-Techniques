class Tree :
    def __init__(self):
        self.parent = None
        self.data = None
        self.children = []
        self.level = None

    def BFS(self,search):
        visited = []
        queue = []
        queue.append(self)
        visited.append(self.data)
        if(self.data == search):
            return self
        while queue:
            s = queue.pop(0)
            for children in s.children:
                if not (children.data  in visited):
                    if(children.data == search):
                        return children
                    queue.append(children)
                    visited.append(children.data)


