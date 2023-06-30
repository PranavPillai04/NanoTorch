''' Create a wrapper class for data '''

class Value:
    
    def __init__(self, data, _children=(), _op=''):
        # each value will have a data, gradient, and a backward function
        self.data = data
        self.grad = 0
        # used for constructing computational graphs
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    

    def backward(self):
        """
        backward will perform a backpropagation on the computational graph.
        For each value in the graph, it will compute the gradient of the loss
        with respect to that value.

        :return: None
        """

        # topological sort all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and backpropagate
        self.grad = 1
        for v in reversed(topo):
            v._backward()
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
    
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"