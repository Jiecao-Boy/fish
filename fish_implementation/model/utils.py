import hydra
import os
import torch
import torch.nn as nn
import torchvision.transforms as T

from collections import OrderedDict
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

# from .pretrained import resnet18, alexnet
from utils.constant import  VISION_IMAGE_MEANS, VISION_IMAGE_STDS
# from utils import crop_transform
from utils.augmentation import crop_transform

# Taken from https://github.com/SridharPandian/Holo-Dex/blob/main/holodex/utils/models.py
def create_fc(input_dim, output_dim, hidden_dims, use_batchnorm=False, dropout=None, is_moco=False):
    if hidden_dims is None:
        return nn.Sequential(*[nn.Linear(input_dim, output_dim)])

    layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]

    if use_batchnorm:
        layers.append(nn.BatchNorm1d(hidden_dims[0]))

    if dropout is not None:
        layers.append(nn.Dropout(p = dropout))

    for idx in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
        layers.append(nn.ReLU())

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dims[idx + 1]))

        if dropout is not None:
            layers.append(nn.Dropout(p = dropout))

    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    if is_moco:
        layers.append(nn.BatchNorm1d(hidden_dims[-1], affine=False))
    return nn.Sequential(*layers)

def init_encoder_info(device, out_dir, encoder_type='image', view_num=1, model_type='byol'): # encoder_type: either image or tactile
        # if encoder_type == 'tactile' and  out_dir is None:
        #     encoder = alexnet(pretrained=True, out_dim=512, remove_last_layer=True)
        #     cfg = OmegaConf.create({'encoder':{'out_dim':512}, 'tactile_image_size':224})
        
        # elif encoder_type =='image' and out_dir is None: # Load the pretrained encoder 
        #     encoder = resnet18(pretrain=True, out_dim=512) # These values are set
        #     cfg = OmegaConf.create({"encoder":{"out_dim":512}})
        
        

        cfg = OmegaConf.load(os.path.join(out_dir, '.hydra/config.yaml'))
        bc_model_type = None # TODO: Clean this code
        if model_type == 'byol': # We assume that the model path is byol directly
            model_path = os.path.join(out_dir, f'models/{model_type}_encoder_best.pt')
        elif model_type == 'bc':
            model_path = os.path.join(out_dir, f'models/{model_type}_{encoder_type}_encoder_best.pt')
            bc_model_type = encoder_type
        else:
            model_path = os.path.join(out_dir, f'saved_models/{encoder_type}_encoder_best.pt') 
            bc_model_type = encoder_type

        encoder = load_model(cfg, device, model_path, bc_model_type)
        encoder.eval() 
        
        if encoder_type == 'image':
            def viewed_crop_transform(image):
                return crop_transform(image, camera_view=view_num)
            
            transform = T.Compose([
                T.Resize((480,640)),
                T.Lambda(viewed_crop_transform),
                T.Resize(480),
                T.ToTensor(),
                T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS),
            ]) 
        else:
            transform = None # This is separately set for tactile

        return cfg, encoder, transform


def load_model(cfg, device, model_path, model_type=None):
    # Initialize the model
    # TODO: Make all these initialization more general - init_learner and load_model!
    if cfg.learner_type == 'bc':
        if model_type == 'image':
            model = hydra.utils.instantiate(cfg.encoder.image_encoder)
        elif model_type == 'tactile':
            model = hydra.utils.instantiate(cfg.encoder.tactile_encoder)
        elif model_type == 'last_layer':
            model = hydra.utils.instantiate(cfg.encoder.last_layer)
    elif cfg.learner_type == 'bc_gmm':
        model = hydra.utils.instantiate(cfg.learner.gmm_layer)
    elif 'byol' in cfg.learner_type: # load the encoder
        model = hydra.utils.instantiate(cfg.encoder)
    elif cfg.learner_type == 'temporal_ssl':
        if model_type == 'image':
            model = hydra.utils.instantiate(cfg.encoder.encoder)
        elif model_type == 'linear_layer':
            model = hydra.utils.instantiate(cfg.encoder.linear_layer)  
    else:
        model = hydra.utils.instantiate(cfg.encoder)  

    state_dict = torch.load(model_path, map_location=device) # All the parameters by default gets installed to cuda 0
    new_state_dict = state_dict

    # Modify the state dict accordingly - this is needed when multi GPU saving was done
    ## NOTE: Not using this for now?
    # new_state_dict = modify_multi_gpu_state_dict(state_dict)
    
    if 'byol' in cfg.learner_type:
        new_state_dict = modify_byol_state_dict(new_state_dict)

    # Load the new state dict to the model 
    model.load_state_dict(new_state_dict)
    model = model.to(device)

    return model

def modify_multi_gpu_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict

def modify_byol_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'encoder.net' in k:
            name = k[12:] # Everything after encoder.net
            new_state_dict[name] = v
    return new_state_dict

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)