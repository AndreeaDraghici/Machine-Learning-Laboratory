from src.clustering import Parser

'''
Conclusion: Metrics are used to monitor and measure the performance of a model (during training and testing), and don't need to be differentiable.
'''

if __name__ == '__main__' :
    myObject = Parser()
    myObject.parse_method()
