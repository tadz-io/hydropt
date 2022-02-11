import numpy as np

class XarrayWrapper(np.ndarray):

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.coords = input_array.coords
        # Finally, we must return the newly created object:
        return obj

    def _reasemble(self):
        '''
        method should reasemble the array into a xarray
        datastruct
        '''
        pass

    def get_axis(self, axis: str):
        '''
        method should return the axis number given
        a axis name
        '''
        pass

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.coords = getattr(obj, 'coords', None)

class LabeledArray(np.ndarray):
    def __new__(cls, input_array, label=()):
        obj = np.asarray(input_array).view(cls)
        # length of array and label should match
        if len(obj) != len(label):
            raise ValueError(
                'Labels do not match dimension of the array'
            )
        # add the new attribute to the created instance
        obj.label = label

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.label = getattr(obj, 'label', None)