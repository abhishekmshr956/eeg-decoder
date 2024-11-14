"""
lets you try model on dataset created by config.yaml
the created dataset is stored inside temp folder
"""
import os, random, torch, numpy as np
from shared_utils.dataset import create_dataset
from utils import create_confusion_matrix
from sklearn.metrics import confusion_matrix
from EEGData import EEGData
from importlib.machinery import SourceFileLoader
from pathlib import Path
from torch.utils.data import DataLoader
import tqdm
import yaml

def load_data(name,folds=[0,1,2,3,4],filePath=None):
    """
    given name of data and folds
    given filePath, uses filePath to fetch data
    return loaded_data = [data]xfold
    """
    
    h5_path = Path('hierarchical_model/temp/') / (name + '.h5')
    if not h5_path.exists():

        # load config file from dataset
        if filePath is None: 
            filePath = Path('/data/raspy/trained_models') / name / 'config.yaml'
        with open(filePath, 'r') as file: config = yaml.safe_load(file)
            
        # Set random seed for reproducibility
        if config['random_seed']:
            seed = config['random_seed']
        else:
            random.seed(str(config['data_names']))          # use data names as seed for random module
            seed = random.randint(0, 2**32-1)
        np.random.seed(seed)
        torch.manual_seed(seed)

        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # ============== h5 Dataset Creation ===============
        create_dataset(config, h5_path=h5_path)
    
    validation_datasets = []
    for validation_fold in folds:
        # train_dataset = EEGData(h5_path, folds)
        validation_dataset = EEGData(h5_path, [validation_fold], train=False)
        validation_datasets.append(validation_dataset)
    
    return validation_datasets

def load_model(name,fold):
    """
    given name
    return model

    Note: load model using instantiate!

    """

    # set up some param
    params = {
        'path': Path('/data/raspy/trained_models') / name,
        'fold': fold
    }
    # instantiate model
    instantiate = SourceFileLoader("instantiate",str(params['path'] / "instantiate.py")).load_module()
    model = instantiate.instantiate(params)
    return model

def test_data(data,model,oneByOne=False):
    """
    return accuracy, pred, label
    """
    if oneByOne:
        pred_ys = []
        ys = []
        dataloader = DataLoader(data, batch_size=1)
        for X,y in tqdm.tqdm(dataloader):
            ys.append(int(y))
            pred_y = model(X)
            pred_y = int(torch.argmax(pred_y))
            pred_ys.append(pred_y)
        y = np.array(ys)
        pred_y = np.array(pred_ys)
    else:    
        dataloader = DataLoader(data, batch_size=len(data))
        X,y = next(iter(dataloader))
        pred_y = model(X)
        pred_y = torch.argmax(pred_y,dim=1).cpu()
    accuracy = sum(pred_y == y) / len(data)
    accuracy = float(accuracy)

    return accuracy, pred_y.tolist(), y.tolist()

if __name__ == '__main__':
    # modelName = 'knows_semblance_Hierarchical' # 2s fold 0
    # modelName = 'milkier_angstroms_Hierarchical' # 1s fold 0
    # modelName = 'miscounted_crackled_Hierarchical' # 1s fold 3
    # modelName = 'deism_reprises_EEGNet_2023-08-15_A2_OL_1'
    modelName = 'extolled_beriberi_Hierarchical'
    # modelName = 'retool_sambaed_Hierarchical'
    # dataName = 'deism_reprises_EEGNet_2023-08-15_A2_OL_1' # 2s 
    filePath = 'config.yaml'
    dataName = 'extolled_beriberi_Hierarchical'
    # dataName = 'gassed_overanxious_EEGNet_2023-08-15_A2_OL_1' # 1s
    # modelName = 'gassed_overanxious_EEGNet_2023-08-15_A2_OL_1' # 1s
    fold = 0
    folds = [0,1]
    print("model:",modelName)
    print("data:",dataName)
    print("fold",fold)
    
    model = load_model(modelName,fold)
    data = load_data(dataName,folds=folds,filePath=filePath)

    accuracy, pred, label = test_data(data[fold],model,oneByOne=True)
    print(accuracy)

    # print confusion matrix
    cm = confusion_matrix(pred,label)
    cm_accuracy = cm / np.sum(cm, axis=1, keepdims=True)
    np.set_printoptions(precision=2)
    print(cm_accuracy)
    create_confusion_matrix(pred,label,"hierarchical_model/confusion_matrix.jpeg")

    # accuracy, pred, label = test_data(data[1],model)
    # print(accuracy)

    # accuracy, pred, label = test_data(data[2],model)
    # print(accuracy)

    

