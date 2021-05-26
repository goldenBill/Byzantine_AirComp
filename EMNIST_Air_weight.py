import os
import sys
import time
import math
import random
import pickle
import argparse
import numpy as np

# Load all necessary modules here, for clearness
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms

def get_args():
    parser = argparse.ArgumentParser('Byzantine_Aircomp')
    parser.add_argument('--opt', type=str, default="SGD", help='optimzer')
    parser.add_argument('--agg', type=str, default='gm', help='agg')
    parser.add_argument('--attack', type=str, default=None, help='attack')
    parser.add_argument('--var', type=float, default=None, help='noise variance')
    parser.add_argument('--inherit', type=bool, default=False, help='inherit')
    parser.add_argument('--mark', type=str, default='', help='mark on title')
    parser.add_argument('--use-gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--K', type=int, default=None, help='number of total devices')
    parser.add_argument('--B', type=int, default=None, help='number of Byzantine devices')
    args = parser.parse_args()
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Report function
def log(*k, **kw):
    timeStamp = time.strftime('[%m-%d %H:%M:%S] ', time.localtime())
    print(timeStamp, end='')
    print(*k, **kw)
    sys.stdout.flush()
    
def report(r, rounds, displayInterval, trainLoss, trainAccuracy, valLoss, valAccuracy, var=None):
    varStr = '' if (var == None) else ' var={:.2e}'.format(var)
    log('[{}/{}](interval: {:.0f}) train: loss={:.4f} acc={:.4f} val: loss={:.4f} acc={:.4f}{}'
        .format(r, rounds, displayInterval, trainLoss, trainAccuracy, valLoss, valAccuracy, varStr)
    )

# Linear Regression model
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        out = x.view(x.size(0), -1) # flatten x in [128, 784]
        out = self.linear(out)
        return out
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( #input shape (1,28,28)
            nn.Conv2d(in_channels=1, #input height
                     out_channels=32, #n_filter
                     kernel_size=5, #filter size
                     stride=1, #filter step
                     padding=2 #picture size is no change
                     ), #output shape (32,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) #2x2 sample, output shape (32,14,14)
        )
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, 1, 2), #output shape (64,7,7)
                                  nn.ReLU(),
                                  nn.MaxPool2d(2))
        
        self.fc1 = nn.Sequential(nn.Linear(64*7*7,2048),#two poolings, is 7*7 not 14*14
                                 nn.ReLU())
        self.fc2 = nn.Linear(2048,62)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) #flat (batch_size, 32*7*7)
        x = self.fc1(x)
        x = self.fc2(x) 
        return x

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0.01)
        
# Linear Regression model factory
def modelFactory(SEED=None, device = None):
    if SEED != None:
        setup_seed(SEED)
    model = MLP(28*28, 62)
#     model = CNN()
    model.apply(weights_init)
    return model

def calculateAccuracy(model, loss_func, loader, device):
    loss = 0
    accuracy = 0
    total = 0
    
    for material, targets in loader:
        material, targets = material.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(material)
            l = loss_func(outputs, targets)

        loss += l.item() * len(targets)
        _, predicted = torch.max(outputs.data, dim=1)
        accuracy += (predicted == targets).sum().item()
        total += len(targets)
    
    loss /= total
    accuracy /= total
    
    return loss, accuracy
    
def getVarience(w_local, honestSize):
    w_honest = w_local[:honestSize]
    return torch.mean( ((w_honest - w_honest.mean(dim=0))**2 ).sum(dim=1) )

def gm(wList, options={}):
    """
    Weiszfeld's algorithm as described on Wikipedia.
    """
    in_device = wList.device
    default_options = {'maxiter': 200, 'tol': 1e-5, 'noise_var': None, 'guess': wList.mean(dim=0), 'P_max': 1}
    default_options.update(options)
    options = default_options
    noise_var = options['noise_var']
    P_max = options['P_max']

    # initial guess: centroid
    guess = options['guess']
#     print(1e-4*math.sqrt(wList.shape[1]))
    for _ in range(options['maxiter']):
        scaler = torch.sqrt(torch.mean(guess**2))
        dist_li = torch.norm(wList-guess, dim=1)
        # catch divide by zero
        # TODO: Wikipedia cites how to deal with distance 0
#         dist_li = torch.where(dist_li == 0, torch.ones(1).to(in_device), dist_li)
        dist_li = torch.max(torch.tensor( 1e-4 ).to(in_device), dist_li)
        noise_message = OMA2(torch.cat([wList/dist_li.unsqueeze(-1), scaler/dist_li.unsqueeze(-1)], dim=-1), P_max = P_max, noise_var=noise_var, threshold = (scaler**2)*500)
        noise_numerator = noise_message[:-1]
        noise_denominator = noise_message[-1:]
        guess_next = noise_numerator / noise_denominator * scaler
        guess_movement = (guess - guess_next).norm()
        guess = guess_next
        if guess_movement <= options['tol']:
            return guess
    return guess

def gm2(wList, options={}):
    """
    Weiszfeld's algorithm as described on Wikipedia.
    """
    in_device = wList.device
    default_options = {'maxiter': 200, 'tol': 1e-5, 'guess': wList.mean(dim=0)}
    default_options.update(options)
    options = default_options

    # initial guess: centroid
    guess = options['guess']
    for _ in range(options['maxiter']):
        dist_li = torch.norm(wList-guess, dim=1)
        # catch divide by zero
        # TODO: Wikipedia cites how to deal with distance 0
#         dist_li = torch.where(dist_li == 0, torch.ones(1).to(in_device), dist_li)
        dist_li = torch.max(torch.tensor(1e-4).to(in_device), dist_li)
        guess_next = (wList/dist_li.unsqueeze(-1)).sum(dim=0) / (1/dist_li).sum()
        guess_movement = (guess - guess_next).norm()
        guess = guess_next
        if guess_movement <= options['tol']:
            return guess
    return guess

def mean(wList, options={}):
    return torch.mean(wList, dim=0)

def trimmed_mean(wList, options={}):
    num = wList.shape[0]
    beta = int(wList.shape[0]*0.1)
    return torch.mean(wList.topk(num-beta, dim=0,largest=False)[0].topk(num-2*beta, dim=0,largest=True)[0], dim=0)

def median(wList, options={}):
    return wList.median(dim=0)[0]

def Krum(wList, options={}):
    honestSize = options['honestSize']
    dist = ((wList.unsqueeze(1)-wList.unsqueeze(0))**2).sum(dim = -1)
    k = honestSize - 2 + 1
    topv, _ = dist.topk(k=k, dim=1, largest=False)
    sumdist = topv.sum(dim=1)
    resindex = sumdist.argmin()
    return wList[resindex]

def flatten_list(message):
    wList = [torch.cat([p.flatten() for p in parameters]) for parameters in message]
    wList = torch.stack(wList)
    return wList

def unflatten_vector(vector, model):
    paraGroup = []
    cum = 0
    for p in model.parameters():
        newP = vector[cum:cum+p.numel()]
        paraGroup.append(newP.view_as(p))
        cum += p.numel()
    return paraGroup

def modelSnapshot(model):
    return model.state_dict()

def modelRecovery(state_dict, model):
    return model.load_state_dict(state_dict, strict=True)

def SGD(model, gamma, aggregate, weight_decay, noise_var = None, honestSize=0, byzantineSize=0,
         attack=None, rounds=10, displayInterval=1000, SEED=None, fixSeed=False, loss_func = None,
            train_dataset=None, validate_dataset=None, device=None, batchSize = None, **kw):
    assert byzantineSize == 0 or attack != None
    assert honestSize != 0
    
    if fixSeed:
        setup_seed(SEED)

    nodeSize = honestSize + byzantineSize
    
    # 数据分片
    pieces = [(i*len(train_dataset)) // nodeSize for i in range(nodeSize+1)]
    dataPerNode = [pieces[i+1] - pieces[i] for i in range(nodeSize)]

    # 回复的消息
    message = [
        [torch.zeros_like(para, requires_grad=False) for para in model.parameters()]
        for _ in range(nodeSize)
    ]

    # enumerate loader
#     all_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), pin_memory=True, shuffle=False)
#     all_validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=len(validate_dataset), pin_memory=True, shuffle=False)
    all_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchSize, pin_memory=True, shuffle=False)
    all_validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=batchSize, pin_memory=True, shuffle=False)

    train_dataset_subset = [torch.utils.data.Subset(train_dataset, range(pieces[i], pieces[i+1])) for i in range(nodeSize)]
#     train_loaders_splited = [
#         torch.utils.data.DataLoader(dataset=subset, batch_size=batchSize, shuffle=False)
#         for subset in train_dataset_subset
#     ]
    
    # random sampler
    randomSampler = [torch.utils.data.sampler.RandomSampler(
        subset,
        num_samples=rounds*displayInterval*batchSize, 
        replacement=True
    ) for subset in train_dataset_subset]
    train_random_loaders_splited = [torch.utils.data.DataLoader(
            dataset=train_dataset_subset[i],
            batch_size=batchSize, 
            sampler=randomSampler[i],
    ) for i in range(nodeSize)]
    randomIters = [iter(loader) for loader in train_random_loaders_splited]
    
    # calculate inital loss and accuracy
#     trainLoss, trainAccuracy = calculateAccuracy(model, loss_func, all_train_loader, device)
    trainLoss, trainAccuracy = 0, 0 # save time whithout evaluatation
    valLoss, valAccuracy = calculateAccuracy(model, loss_func, all_validate_loader, device)
    
    trainLossPath = [trainLoss]
    trainAccPath = [trainAccuracy]
    valLossPath = [valLoss]
    valAccPath = [valAccuracy]
    variencePath = []
    
    report(0, rounds, displayInterval, trainLoss, trainAccuracy, valLoss, valAccuracy)

    weight_list = [None]*nodeSize
    # SGD begin
    for r in range(rounds):
        for k in range(displayInterval):
            # honest node update
            weight_list = [None]*nodeSize
            state_dict = modelSnapshot(model)
            for node in range(nodeSize):
                if node < honestSize:
                    material, targets = next(randomIters[node])
                    material, targets = material.to(device), targets.to(device)
                    # prediction
                    outputs = model(material)
                    loss = loss_func(outputs, targets)
                    # backpropagation
                    model.zero_grad()
                    loss.backward()
                    # update
                    for para in model.parameters():
                        para.data.add_(-gamma, para.grad.data + weight_decay*para.data)    
                    weight_list[node] = [para.clone().detach().cpu() for para in model.parameters()]
                    
                else:
                    material, targets = next(randomIters[node])
                    material, targets = material.to(device), targets.to(device)

                    if attack == None:
                        # prediction
                        outputs = model(material)
                        loss = loss_func(outputs, targets)
                        # backpropagation
                        model.zero_grad()
                        loss.backward()
                    elif attack.__name__== 'classflip':
                        # prediction
                        outputs = model(material)
                        loss = loss_func(outputs, 61.0-targets)
                        # backpropagation
                        model.zero_grad()
                        loss.backward()
                    elif attack.__name__== 'dataflip':
                        # prediction
                        outputs = model(1.0-material)
                        loss = loss_func(outputs, targets)
                        # backpropagation
                        model.zero_grad()
                        loss.backward()
                    else: # no attack
                        # prediction
                        outputs = model(material)
                        loss = loss_func(outputs, targets)
                        # backpropagation
                        model.zero_grad()
                        loss.backward()
                    # update
                    for para in model.parameters():
                        para.data.add_(-gamma, para.grad.data + weight_decay*para.data)
                    weight_list[node] = [para.clone().detach().cpu() for para in model.parameters()]
                        
                modelRecovery(state_dict, model)
            
            weight_f = flatten_list(weight_list)
            if attack != None:
                attack(weight_f, byzantineSize)
            modelRecovery(state_dict, model)
            init_point = torch.cat([p.clone().detach().cpu().flatten() for p in model.parameters()])
            options = {'maxiter': 1000, 'tol': 1e-5, 'eta': 1, 'noise_var': noise_var, 'guess': init_point, 'honestSize': honestSize}
            if aggregate.__name__!= 'gm' and noise_var != None:
                OMA(weight_f, noise_var)
            weight_vector = aggregate(weight_f, options)
            weight = unflatten_vector(weight_vector, model)
            
            # update
            for para, new_para in zip(model.parameters(), weight):
                para.data.copy_(new_para.to(device))
                
        var = getVarience(weight_f, honestSize)
        variencePath.append(var)

#         trainLoss, trainAccuracy = calculateAccuracy(model, loss_func, all_train_loader, device)
        trainLoss, trainAccuracy = 0, 0 # save time whithout evaluatation
        valLoss, valAccuracy = calculateAccuracy(model, loss_func, all_validate_loader, device)

        trainLossPath.append(trainLoss)
        trainAccPath.append(trainAccuracy)
        valLossPath.append(valLoss)
        valAccPath.append(valAccuracy)
        
        report(r+1, rounds, displayInterval, trainLoss, trainAccuracy, valLoss, valAccuracy)
    return model, trainLossPath, trainAccPath, valLossPath, valAccPath, variencePath

def classflip(messages, byzantinesize):
    pass

def dataflip(messages, byzantinesize):
    pass

def weightflip(messages, byzantinesize):
    s = torch.sum(messages[0:-byzantinesize], dim=0)
    messages[-byzantinesize:].mul_(-1)
    messages[-byzantinesize:].add_(-2, s / byzantinesize)

def OMA(message, noise_var = 0.01):
    in_device = message.device
    scale = math.sqrt(noise_var)
    mess_shape = message.shape
    channel_real = torch.normal(torch.zeros(mess_shape[0], 1), 1/math.sqrt(2)).to(in_device)
    channel_imag = torch.normal(torch.zeros(mess_shape[0], 1), 1/math.sqrt(2)).to(in_device)
    noise_real = torch.normal(torch.zeros(*mess_shape), scale).to(in_device)
    noise_imag = torch.normal(torch.zeros(*mess_shape), scale).to(in_device)
    de_noise = (channel_real*noise_real + channel_imag*noise_imag) / (channel_real**2 + channel_imag**2)
    message[:].add_( de_noise )
    
def OMA2(message, P_max=10, noise_var=None, threshold=1):
    noise_message = message.clone().detach()
    in_device = message.device
    mess_shape = message.shape
    
    channel_real = torch.normal(torch.zeros(mess_shape[0]), 1/math.sqrt(2)).to(in_device)
    channel_imag = torch.normal(torch.zeros(mess_shape[0]), 1/math.sqrt(2)).to(in_device)
    h_square = ((channel_real)**2 + (channel_imag)**2)
    P_message = noise_message**2/h_square.unsqueeze(-1)
    P_upper = torch.max(torch.mean(P_message, dim=-1), threshold)
    
    P_gain = torch.sqrt(P_max/P_upper)
    noise_message_masked = (noise_message*P_gain.unsqueeze(-1))
    if noise_var != None:
        scale = math.sqrt(noise_var/2)
        de_noise = torch.normal(torch.zeros(mess_shape[1]), scale).to(in_device)
        return noise_message_masked.sum(dim=0).add_( de_noise )
    else:
        return noise_message_masked.sum(dim=0)

def getPara(module, useString=True):
    para = sum([x.nelement() for x in module.parameters()])
    if not useString:
        return para
    elif para >= 2**20:
        return '{:.2f}M'.format(para / 2**20)
    elif para >= 2**10:
        return '{:.2f}K'.format(para / 2**10)
    else:
        return str(para)

def run(optimizer, aggregate, attack, config, noise_var = None, dataSetConfig=None, recordInFile=True, markOnTitle='', device = None):
    # initialize parameters
    _config = config.copy()
    if attack == None:
        _config['byzantineSize'] = 0
    else:
        attack = eval(attack)
    _config['aggregate'] = aggregate
    _config['attack'] = attack
    _config['noise_var'] = noise_var
    
    model = modelFactory(SEED=_config['SEED'], device = device)
    if device != torch.device("cpu"):
        model = nn.DataParallel(model)
    model = model.to(device)

    # record parameters
    attackName = 'baseline' if attack == None else attack.__name__
    # e.g. Resnet50_SARAH(5)_baseline_mean
    title = '{}_{}_{}_{}'.format(
        model.__class__.__name__, 
        optimizer.__name__,
        attackName, 
        aggregate.__name__
    )
    if noise_var != None:
        title = title + '_' + str(noise_var)
    if markOnTitle != '':
        title = title + '_' + markOnTitle
    
    # print running information
    print('[submit task ] ' + _config['CACHE_DIR'] + title)
    print('[running info]')
    print('[network info]   name={} parameters number={}'.format(model.__class__.__name__, getPara(model)))
    print('[optimization]   name={} aggregation={} attack={}'.format(optimizer.__name__, aggregate.__name__, attackName))
    print('[dataset info] name={} trainSize={} validationSize={}'.format(dataSetConfig['name'], len(_config['train_dataset']), len(_config['validate_dataset'])))
    print('[optimizer   ] gamma={} weight_decay={} batchSize={}'.format(_config['gamma'], _config['weight_decay'], _config['batchSize']))
    print('[node number ]   honestSize={}, byzantineSize={}'.format(_config['honestSize'], _config['byzantineSize']))
    print('[running time]   rounds={}, displayInterval={}'.format(_config['rounds'], _config['displayInterval']))
    print('[torch set   ]  device={}, SEED={}, fixSeed={}'.format(device, _config['SEED'], _config['fixSeed']))
    print('-------------------------------------------')
    
    # begin
    log('Optimization begin')
    res = optimizer(model, device=device, **_config)
    [*model, trainLossPath, trainAccPath, valLossPath, valAccPath, variencePath] = res

    config_record = {}
    for key in _config:
        if key in ['train_dataset', 'validate_dataset',]:
            continue
        val = _config[key].__class__.__name__ if hasattr(_config[key], '__call__') else _config[key]
        config_record[key] = val
            
    record = {
        **dataSetConfig,
        **config_record,
        'trainLossPath': trainLossPath, 
        'trainAccPath': trainAccPath, 
        'valLossPath': valLossPath, 
        'valAccPath': valAccPath, 
        'variencePath': variencePath,
    }

    with open(_config['CACHE_DIR'] + title, 'wb') as f:
        pickle.dump(record, f)

def main():
    args = get_args()
    optimizer = args.opt
    aggregate = args.agg
    attack = args.attack
    noise_var = args.var
    inherit = args.inherit
    markOnTitle = args.mark
    use_gpu = args.use_gpu
    K = args.K
    B = args.B
    
    if not torch.cuda.is_available():
        print("GPU is not found.")
        use_gpu = False
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES']='0, 1'
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # MLP
    optConfig = {
        'honestSize':50,
        'byzantineSize': 0,

        'rounds': 100,
        'displayInterval': 10,
        
        'weight_decay': 0.00,

        'fixSeed': True,
        'SEED': 2021,

        'batchSize': 50,
        'shuffle': True,
    }
    if B != None and K != None:
        optConfig['honestSize'] = K - B
        optConfig['byzantineSize'] = B

    # dataset property
    dataSetConfig = {
        'name': 'emnist',
        'dataSet' : 'emnist',
        'dataSetSize': 697932,
        'maxFeature': 784,
    }
    # learning rate
    SAGAConfig = optConfig.copy()
    SAGAConfig['gamma'] = 1e-2
    # store dir
    CACHE_DIR = './EMNIST_Air_weight/'
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    CACHE_DIR += dataSetConfig['name'] + '_K'+str(optConfig['honestSize']+optConfig['byzantineSize']) +'_B' + str(optConfig['byzantineSize'])+'_'
    SAGAConfig['CACHE_DIR'] = CACHE_DIR
    # load dataset
    train_transform = transforms.Compose([
        transforms.ToTensor(), # Convert a PIL Image or numpy.ndarray to tensor.
        # Normalize a tensor image with mean 0.1307 and standard deviation 0.3081
        transforms.Normalize((0.1736,), (0.3317,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1736,), (0.3317,))
    ])
    train_dataset = torchvision.datasets.EMNIST(root='./dataset/', 
                                split="byclass",
                                train=True, 
                                transform=train_transform,
                                download=False)
    validate_dataset = torchvision.datasets.EMNIST(root='./dataset/', 
                                split="byclass",
                               train=False, 
                               transform=test_transform,
                               download=False)

    SAGAConfig['train_dataset'] = train_dataset
    SAGAConfig['validate_dataset'] = validate_dataset

    loss_func = torch.nn.CrossEntropyLoss()
    if use_gpu:
        loss_func = loss_func.cuda()
    else:
        loss_func = loss_func
    SAGAConfig['loss_func'] = loss_func
    
    run(optimizer = eval(optimizer), aggregate = eval(aggregate), attack = attack, noise_var = noise_var, config = SAGAConfig,
        dataSetConfig = dataSetConfig, device = device, markOnTitle=markOnTitle)

if __name__ == "__main__":
    main()