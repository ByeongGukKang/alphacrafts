from copy import deepcopy


cdef class cyClsMemoryArena:

    cdef list[object] arena

    def __init__(self, int size, object obj):
        self.arena = [deepcopy(obj) for _ in range(size)] 
    
    cpdef object alloc(self):
        return self.arena.pop()

    cpdef void free(self, object obj):
        self.arena.append(obj)

    def __len__(self):
        return len(self.arena)
