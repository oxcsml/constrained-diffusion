import random

class SwitchObject:
    def __init__(self, objects, **kwargs):
        self.objects = objects
        self.index = 0
        self.update()
        
    def update(self, index=None):
        if index is not None:
            self.index = index
        self.__dict__.update(
            {attr: getattr(self.objects[self.index], attr) 
             for attr in dir(self.objects[self.index])}
        )
    
    def iter_index(self):
        self.update((self.index + 1) % len(self.objects))
        
    def randomize_index(self):
        self.index, = random.choices(
            range(len(self.objects)), 
            weights=[
                len(obj) if hasattr(obj, "__len__") else 1 
                for obj in self.objects
            ]
        )
        self.update()
        
    def __next__(self):
        return self.__next__()
    
    def __len__(self):
        return self.__len__()
