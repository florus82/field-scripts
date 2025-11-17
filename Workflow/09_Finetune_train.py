import sys
sys.path.append('/media/')
from helperToolz.helpsters import *
from helperToolz.feevos.rocksdbutils_copy import *

from tqdm import tqdm
from torch.amp import autocast, GradScaler


origin = 'Docker'

if origin == 'Docker':
    prefix = '/workspace/'
else:
    prefix = '/data/Aldhani/eoagritwin/'

# setfine-tune dataset
db_name = 'Fine_tuner'

# set model that is fine-tuned
model_check = 'AI4_RGB_exclude_True_38'

# freezing strategies

# option 0 == non-freeze

# option 1: freeze everything (Entire encoder (self.features) + 3D stage inside head3D)
# New dataset is small
# New labels are similar to original
# for maximum stability
# avoid catastrophic forgetting
### Effect:
# Only the segmentation head learns the new domain
# Very safe, very stable
# Slower improvements but minimal overfitting


# option 2: freeze encode, train full head
# New dataset is medium-sized
# Distribution changes slightly
# moderate adaptation
### Effect:
# Fast adaptation
# Keeps backbone stable
# Good for domain shift (new time frames, new regions)


# option 3: Freeze early encoder blocks only (features.stage 1 +2)
# Dataset is large
# Low-level features remain the same (edge, texture)
# High-level structure needs adaptation
### Effect:
# Preserves basic feature extraction
# Updates higher-level semantics


# option 4:
# Dataset is very large
# Or drastically different (e.g. RGB → SAR, different spectral bands)

FREEZER = 2


# Set this to False for training
#DEBUG=True
DEBUG=False

def mtsk_loss(preds, labels,criterion, NClasses=1):                   
    # Multitasking loss,    segmentation / boundaries/ distance     
                                                                    
    pred_segm  = preds[:,:NClasses]                                 
    pred_bound = preds[:,NClasses:2*NClasses]                       
    pred_dists = preds[:,2*NClasses:3*NClasses]                     
                                                                    
                                                                    
                                                                    
    # Multitasking loss                                             
    label_segm  = labels[:,:NClasses]                               
    label_bound = labels[:,NClasses:2*NClasses]                     
    label_dists = labels[:,2*NClasses:3*NClasses]                   
                                                                    
                    
    #print(preds.shape, labels.shape)

    loss_segm  = criterion(pred_segm,   label_segm)                 
    loss_bound = criterion(pred_bound, label_bound)                 
    loss_dists = criterion(pred_dists, label_dists)                 
                                                                                                                                        
    return (loss_segm+loss_bound+loss_dists)/3.0     


# create output dictionary
keys = ['Epoch', 'Iteration','Loss', 'Mode']
vals = [list() for _ in range(len(keys))]
res  = dict(zip(keys, vals))

keys = ['Epoch', 'MCC']
vals = [list() for _ in range(len(keys))]
res2  = dict(zip(keys, vals))


def monitor_epoch(model, epoch, datagen_valid, NClasses=1):
    metric_target = Classification(num_classes=NClasses, task='binary').to(0)
    model.eval()

    valid_pbar = tqdm(datagen_valid, desc=f"Validating Epoch {epoch}", position=1, leave=False)
    for idx, data in enumerate(valid_pbar):
        images, labels = data
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        with torch.inference_mode():
            preds_target = model(images)

        criterionV = ftnmt_loss()
        lossi = mtsk_loss(preds_target, labels, criterionV, NClasses)

        res['Epoch'].append(epoch)
        res['Iteration'].append(idx)
        res['Loss'].append(lossi.item())
        res['Mode'].append('Valid')


        pred_segm = preds_target[:, :NClasses]
        label_segm = labels[:, :NClasses]

        metric_target(pred_segm, label_segm)
     
        if DEBUG and idx > 5:
            break
    
    metric_kwargs_target = metric_target.compute()
    

    kwargs = {'epoch': epoch}
    for k, v in metric_kwargs_target.items():
        kwargs[k] = v.cpu().numpy()
    return kwargs


def train(args):
    # dummy variable to keep track of mcc
    conti = [1,2]
    mcc_dum = 0
    num_epochs = args.epochs
    batch_size = args.batch_size

    torch.manual_seed(0)
    local_rank = 0
    # torch.cuda.set_device(local_rank)
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    NClasses = 1
    nf = 96
    verbose = True
    model_config = {'in_channels': 4,
                    'spatial_size_init': (128, 128),
                    'depths': [2, 2, 5, 2],
                    'nfilters_init': nf,
                    'nheads_start': nf // 4,
                    'NClasses': NClasses,
                    'verbose': verbose,
                    'segm_act': 'sigmoid'}

    model = ptavit3d_dn(**model_config).to(local_rank)

    # set checkpoint
    checkpoint = torch.load(f'{prefix}fields/output/models/model_state_{model_check}.pth',
                        map_location='cuda')
    model.load_state_dict(checkpoint, strict=True)
    print("Loaded pretrained weights.")

    if FREEZER == 1:
        for name, param in model.named_parameters():
            if "head3D" in name and "head_cmtsk" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif FREEZER == 2:
        for name, param in model.named_parameters():
            if "features" in name:
                param.requires_grad = False

    elif FREEZER == 3:
        for name, param in model.named_parameters():
            if "features.stage1" in name or "features.stage2" in name:
                param.requires_grad = False

    elif FREEZER == 4:
        for param in model.parameters():
            param.requires_grad = True
    else:
        pass



    criterion = ftnmt_loss()
    criterion_features = ftnmt_loss(axis=[-3, -2, -1])
    #optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3, eps=1.e-6)
    optimizer = torch.optim.RAdam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    eps=1.e-6
)
    scaler = GradScaler()

    train_dataset = RocksDBDataset(f'{prefix}fields/output/rocks_db/{db_name}.db/train.db', transform=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    valid_dataset = RocksDBDataset(f'/{prefix}fields/output/rocks_db/{db_name}.db/valid.db', transform=None)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    start = datetime.now()
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)
    for epoch in epoch_pbar:
        tot_loss = 0
        model.train() # train function from ptavit3d_dn(torch.nn.Module) is called
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", position=1, leave=False)
        for i, data in enumerate(train_pbar):
            if DEBUG and i > 5:
                break

            images, labels = data
            images = images.to(local_rank, non_blocking=True)
            labels = labels.to(local_rank, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                preds_target = model(images)
                loss = mtsk_loss(preds_target, labels, criterion, NClasses)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tot_loss += loss.item()
            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

               # for export
            res['Epoch'].append(epoch)
            res['Iteration'].append(i)
            res['Loss'].append(loss.item())
            res['Mode'].append('Train')

        kwargs = monitor_epoch(model, epoch, valid_loader, NClasses)
        kwargs['tot_train_loss'] = tot_loss
        # for export
        res2['Epoch'].append(epoch)
        res2['MCC'].append(kwargs['mcc'])

        # check if mcc higher than ever observed
        if kwargs['mcc'] > mcc_dum:
            mcc_dum = kwargs['mcc']
            conti[0] = model.state_dict()
            conti[1] = epoch
        
     
        #res.append 
        if verbose:
            output_str = ', '.join(f'{k}:: {v}, |===|, ' for k, v in kwargs.items())
            epoch_pbar.write(output_str)


    if verbose:
        print("Training completed in: " + str(datetime.now() - start))

    
    torch.save(conti[0], f'{prefix}fields/output/models/model_state_{db_name}_{conti[1]}_on_{model_check}.pth') # 

    df  = pd.DataFrame(data = res)
    df.to_csv(f'{prefix}fields/output/loss/loss_{db_name}_on_{model_check}.csv', sep=',',index=False)

    df  = pd.DataFrame(data = res2)
    df.to_csv(f'{prefix}fields/output/loss/MCC_{db_name}_on_{model_check}.csv', sep=',',index=False)


def main():
    class Args:
        def __init__(self):
            self.epochs = 50
            self.batch_size = 3 # H100 test - 94GB GPU memory

    args = Args()
        
    train(args)

if __name__ == '__main__':
    main()



### test at some point

# def train(args):
#     # progressive unfreezing phase lengths
#     phase_1_epochs = 5
#     phase_2_epochs = 10
#     total_epochs = args.epochs
#     assert total_epochs > phase_1_epochs + phase_2_epochs, \
#         "Increase epochs: not enough left for phase 3."

#     # dummy variable to keep track of best MCC
#     conti = [1, 2]
#     best_mcc = 0

#     torch.manual_seed(0)
#     local_rank = 0
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     NClasses = 1
#     nf = 96
#     verbose = True

#     model_config = {
#         'in_channels': 5,
#         'spatial_size_init': (128, 128),
#         'depths': [2, 2, 5, 2],
#         'nfilters_init': nf,
#         'nheads_start': nf // 4,
#         'NClasses': NClasses,
#         'verbose': verbose,
#         'segm_act': 'sigmoid'
#     }

#     model = ptavit3d_dn(**model_config).to(local_rank)

#     # -----------------------------------------------------------
#     # Load pretrained checkpoint for fine-tuning
#     # -----------------------------------------------------------
#     pretrained_path = "/data/fields/output/models/model_state_AI4_RGB_NDVI_exclude_False_XX.pth"
#     checkpoint = torch.load(pretrained_path, map_location=device)
#     model.load_state_dict(checkpoint, strict=True)
#     print(f"Loaded pretrained model from {pretrained_path}")

#     # -----------------------------------------------------------
#     # Freezing helper functions
#     # -----------------------------------------------------------
#     def freeze_all_encoder(model):
#         for name, param in model.named_parameters():
#             if "features" in name:
#                 param.requires_grad = False
#             else:
#                 param.requires_grad = True

#     def unfreeze_top_encoder(model):
#         for name, param in model.named_parameters():
#             # unfreeze deeper blocks (usually stage3 + stage4)
#             if ("features.stage3" in name) or ("features.stage4" in name):
#                 param.requires_grad = True

#     def unfreeze_entire_encoder(model):
#         for param in model.parameters():
#             param.requires_grad = True

#     # -----------------------------------------------------------
#     # Data
#     # -----------------------------------------------------------
#     train_dataset = RocksDBDataset(
#         f'/data/fields/output/rocks_db/{db_name}.db/train.db', transform=None
#     )
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
#                               shuffle=False, num_workers=4, pin_memory=True)

#     valid_dataset = RocksDBDataset(
#         f'/data/fields/output/rocks_db/{db_name}.db/valid.db', transform=None
#     )
#     valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
#                               shuffle=False, num_workers=4, pin_memory=True)

#     criterion = ftnmt_loss()
#     scaler = GradScaler()

#     # -----------------------------------------------------------
#     # Start training
#     # -----------------------------------------------------------
#     start = datetime.now()
#     epoch_pbar = tqdm(range(total_epochs), desc="Epochs", position=0)

#     # initially freeze encoder (Phase 1)
#     freeze_all_encoder(model)
#     optimizer = torch.optim.RAdam(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=1e-4, eps=1e-6
#     )
#     current_phase = 1
#     print("PHASE 1: Training head only")

#     for epoch in epoch_pbar:

#         # -------------------------------------------------------
#         # Phase transitions
#         # -------------------------------------------------------
#         if epoch == phase_1_epochs:
#             print("Switching to PHASE 2: unfreeze top encoder layers")
#             unfreeze_top_encoder(model)
#             optimizer = torch.optim.RAdam(
#                 filter(lambda p: p.requires_grad, model.parameters()),
#                 lr=5e-5, eps=1e-6
#             )
#             current_phase = 2

#         if epoch == phase_1_epochs + phase_2_epochs:
#             print("Switching to PHASE 3: unfreeze entire encoder")
#             unfreeze_entire_encoder(model)
#             optimizer = torch.optim.RAdam(
#                 model.parameters(),
#                 lr=1e-5, eps=1e-6
#             )
#             current_phase = 3

#         # -------------------------------------------------------
#         # Training loop
#         # -------------------------------------------------------
#         model.train()
#         tot_loss = 0
#         train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", position=1, leave=False)

#         for i, data in enumerate(train_pbar):
#             images, labels = data
#             images = images.to(device, non_blocking=True)
#             labels = labels.to(device, non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)

#             with autocast(device_type='cuda', dtype=torch.bfloat16):
#                 preds = model(images)
#                 loss = mtsk_loss(preds, labels, criterion, NClasses)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             tot_loss += loss.item()
#             train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

#             # logging
#             res['Epoch'].append(epoch)
#             res['Iteration'].append(i)
#             res['Loss'].append(loss.item())
#             res['Mode'].append('Train')

#         # -------------------------------------------------------
#         # Validation + MCC tracking
#         # -------------------------------------------------------
#         kwargs = monitor_epoch(model, epoch, valid_loader, NClasses)
#         kwargs['tot_train_loss'] = tot_loss

#         res2['Epoch'].append(epoch)
#         res2['MCC'].append(kwargs['mcc'])

#         if kwargs['mcc'] > best_mcc:
#             best_mcc = kwargs['mcc']
#             conti[0] = model.state_dict()
#             conti[1] = epoch

#         if verbose:
#             output_str = ', '.join(f'{k}: {v}, ' for k, v in kwargs.items())
#             epoch_pbar.write(output_str)

#     # -----------------------------------------------------------
#     # Save best model + logs
#     # -----------------------------------------------------------
#     if verbose:
#         print("Training completed in:", datetime.now() - start)

#     save_path = f'/data/fields/output/models/model_state_{db_name}_{conti[1]}.pth'
#     torch.save(conti[0], save_path)
#     print(f"Saved best model to {save_path}")

#     pd.DataFrame(res).to_csv(f'/data/fields/output/loss/loss_{db_name}.csv', index=False)
#     pd.DataFrame(res2).to_csv(f'/data/fields/output/loss/MCC_{db_name}.csv', index=False)
