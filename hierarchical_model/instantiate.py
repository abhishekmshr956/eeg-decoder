

from importlib.machinery import SourceFileLoader
from pathlib import Path
import yaml
import torch

def submodel_instantiate(path,name,fold):
    """
    path of the folder containing the model
    name of the model (sub folder name)
    """
    params = { 
        'path': Path(path) / name,
        'fold': fold
    }
    # instantiate model
    submodel_file = SourceFileLoader("instantiate",str(params['path'] / "instantiate.py")).load_module()
    model = submodel_file.instantiate(params)
    return model

class HierarchicalModelClass:
    def __init__(self, params):
        self.params = params
        self.path = Path(params['path'])

        # =======  get hierarchical model yaml  =======
        h_yaml_path = self.path / 'hierarchical_model.yaml'
        with open(h_yaml_path, 'r') as file: h_yaml = yaml.safe_load(file)
        self.h_yaml = h_yaml['hierarchical_model']

        # self.path = Path('/data/raspy/trained_models')
        self.tree = self.construct_tree(self.h_yaml)
        self.modelAndText = [] # shows all the model with a text as a list [(model1,text2),(model2,text2)]
        self.tree_to_list(self.tree)

    def construct_tree(self,node):
        """ repurpose hierarchical_model yaml into a model tree """

        # end condition
        if type(node) is int: return
        # recursively create model
        node['model'] = submodel_instantiate(self.path,node['path'],node['fold'])
        for child in node['out'].values():
            self.construct_tree(child)
        
        return node

    def __call__(self,X):
        return self.predict(X)
    
    def predict(self,X):

        # predict by tree method
        if len(X) == 1:
            y =  self.predict_node(X, self.tree) # efficient
            v = torch.zeros(4)
            v[y] = 1
            return v
        else:
            #TODO unfinished cannot handle batchSize > 1
            return self.predict_detail(X) # step by step


    def predict_node(self, X, node):
        if type(node) is int: return node
        i = int(torch.argmax(node['model'](X)))
        return self.predict_node(X, node['out'][i])

    def predict_detail(self,X):
        for model,text in self.modelAndText:
            print(text,model(X).tolist())

    def tree_to_list(self,node):
        if type(node) is int: return str(node)
        
        model = node['model']
        choice = [self.tree_to_list(node['out'][i]) for i in sorted(node['out'].keys())]
        text = f"{node['path']} [{' v '.join(choice)}]"
        self.modelAndText.append([model,text])

        return ''.join(choice)


def instantiate(params):

    return HierarchicalModelClass(params)

if __name__ == '__main__':
    
    # instantiate model
    params = {
        'path':Path('./hierarchical_model')
    }
    model = instantiate(params)

    # predict with model
    y = model(torch.zeros((4,61,200,1)))
    print(y)
    