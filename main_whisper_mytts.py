import argparse
import sys
import os
import numpy as np
import torch
import tensorboard
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from data_utils_SSL import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval, Dataset_in_the_wild_eval, Dataset_myTTS
from modelwhisper import Model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed


def evaluate_accuracy(dev_loader, modelwav2vec, device):
    val_loss = 0.0
    num_total = 0.0
    modelwav2vec.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in dev_loader:
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = modelwav2vec(batch_x)
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        
    val_loss /= num_total
   
    return val_loss


def produce_evaluation_file(dataset, modelwav2vec, device, save_path):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    modelwav2vec.eval()
    
    fname_list = []
    key_list = []
    score_list = []
    
    for batch_x,utt_id in data_loader:
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        
        batch_out = modelwav2vec(batch_x)
        
        batch_score = (batch_out[:, 1]  
                       ).data.cpu().numpy().ravel() 
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))
    
def compute_metrics(dataset, model, device):
    """
    Compute accuracy and other metrics (optional)
    """
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
    model.eval()
    
    num_correct = 0.0
    num_total = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            
            batch_out = model(batch_x)
            batch_pred = batch_out.argmax(dim=1)
            
            num_correct += (batch_pred == batch_y).sum().item()
    
    accuracy = (num_correct / num_total) * 100
    print(f'Accuracy: {accuracy:.2f}% ({num_correct}/{num_total})')
    return accuracy    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/home/a/angelikkd/GitProjects/SSL_Anti-spoofing/tts_mydataset/')
   

     # Essential arguments
#    parser.add_argument('--database_path', type=str, required=True,
#                        help='Path to evaluation dataset')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--eval_output', type=str, required=True,
                        help='Path to save evaluation scores')
    
    # Dataset selection
    parser.add_argument('--track', type=str, default='LA', 
                        choices=['LA', 'PA', 'DF', 'In-the-Wild', 'Glow', 'Tacotron'], 
                        help='Evaluation track')
    
    # Optional arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--compute_accuracy', action='store_true',
                        help='Compute accuracy (requires labels)')

    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                    default=True,
                    help='use cudnn-deterministic? (default true)')

    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                    default=False,
                    help='use cudnn-benchmark? (default false)')

    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed, args)
    
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))
    
    # Load model
    print('Loading model from: {}'.format(args.model_path))
    model = Model(args, device)  # You might need to pass dummy args
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print('Model loaded successfully')
    nb_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {nb_params:,}')
    
    # Prepare evaluation dataset
    if args.track == 'Glow':
        # For Glow dataset
        file_eval = genSpoof_list(
            dir_meta=os.path.join(args.database_path, "glow_tts/merged_ALL1000_ids.txt"),
            is_train=False, 
            is_eval=True
        )
        print(f'Number of evaluation trials: {len(file_eval)}')
        
        eval_set = Dataset_myTTS(
            list_IDs=file_eval,
            base_dir=os.path.join(args.database_path, "glow_tts/merged_ALL1000_wavs")
        )
        
    elif args.track == 'Tacotron':
        # For Tacotron dataset
        file_eval = genSpoof_list(
            dir_meta=os.path.join(args.database_path, "tactron2dcc_tts/merged_ALL1000_ids.txt"),
            is_train=False, 
            is_eval=True
        )
        print(f'Number of Tacotron evaluation trials: {len(file_eval)}')
        

        eval_set = Dataset_myTTS(  
            list_IDs=file_eval,
            base_dir=os.path.join(args.database_path, "tactron2dcc_tts/merged_ALL1000_wavs")
        )    
    # Run evaluation
    print('Starting evaluation...')
    produce_evaluation_file(eval_set, model, device, args.eval_output)
    
    # Optionally compute accuracy (if labels are available)
    if args.compute_accuracy and hasattr(eval_set, 'labels'):
        accuracy = compute_metrics(eval_set, model, device)
    
    print('Evaluation completed!')
        