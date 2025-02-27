import sys
sys.path.append('./')
import random
random.seed(42)
import argparse
from tqdm import tqdm
import os
import torch.optim as optim
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from lightning.fabric.strategies import DDPStrategy
from transformers import AutoTokenizer
from lightning import Fabric
from dataset import MyDataset
from models import SimCLR_Classifier_SCL
from torch.utils.tensorboard import SummaryWriter

def collate_fn(batch):
    batch_dict = default_collate(batch) 
    articles = batch_dict[0] 
    articles = [str(article) if article is not None else '' for article in articles]
    labels = batch_dict[1] 
    details = batch_dict[2] 
    encoded_batch = tokenizer.batch_encode_plus(
        articles,
        return_tensors="pt",
        max_length=512,
        padding='max_length',
        truncation=True,
    )
    return {
        'input_ids': encoded_batch['input_ids'],
        'attention_mask': encoded_batch['attention_mask'],
        'labels': labels,
        'details': details
    }
def train(opt):
    torch.set_float32_matmul_precision("medium")
    if opt.device_num>1:
        ddp_strategy = DDPStrategy(find_unused_parameters=True)
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=opt.device_num,strategy=ddp_strategy)#
    else:
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=opt.device_num)
    fabric.launch()

    if opt.simple_classifier:
        human_train=MyDataset(opt.human_dataset_path+"/human_train.jsonl",need_details=True)
        human_eval=MyDataset(opt.human_dataset_path+"/human_dev.jsonl",need_details=True)
        machine_train=MyDataset(opt.gpt_dataset_path+"/gpt_none_train.jsonl",need_details=True)
        machine_eval=MyDataset(opt.gpt_dataset_path+"/gpt_none_dev.jsonl",need_details=True)
        train_dataset=human_train+machine_train
        eval_dataset=human_eval+machine_eval
        
        opt.classifier_dim=2

    train_dataloder = DataLoader(train_dataset, batch_size=opt.per_gpu_batch_size,\
                                     num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=True,collate_fn=collate_fn)
    
    val_dataloder = DataLoader(eval_dataset, batch_size=opt.per_gpu_eval_batch_size,\
                            num_workers=opt.num_workers, pin_memory=True,shuffle=True,drop_last=False,collate_fn=collate_fn)
    
    if opt.simple_classifier:
        #设置为训练模式
        model = SimCLR_Classifier_SCL(opt,fabric).train()
    else:
        pass
    train_dataloder,val_dataloder=fabric.setup_dataloaders(train_dataloder,val_dataloder)

    if fabric.global_rank == 0 :
        for num in range(10000):
            if os.path.exists(os.path.join(opt.savedir,'{}_v{}'.format(opt.name,num)))==False:
                opt.savedir=os.path.join(opt.savedir,'{}_v{}'.format(opt.name,num))
                os.makedirs(opt.savedir)
                break
        if os.path.exists(os.path.join(opt.savedir,'runs'))==False:
            os.makedirs(os.path.join(opt.savedir,'runs'))
        writer = SummaryWriter(os.path.join(opt.savedir,'runs'))
        opt_dict = vars(opt)
        with open(os.path.join(opt.savedir,'config.yaml'), 'w') as file:
            yaml.dump(opt_dict, file, sort_keys=False)


    num_batches_per_epoch = len(train_dataloder)
    warmup_steps=opt.warmup_steps
    lr = opt.lr
    total_steps = opt.total_epoch * num_batches_per_epoch- warmup_steps
    optimizer = optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)

    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, total_steps, eta_min=lr/10)
    model, optimizer = fabric.setup(model, optimizer)
    max_avg_rec=0
    for epoch in range(opt.total_epoch):
        model.train()
        avg_loss=0
        pbar = enumerate(train_dataloder)
        pbar = tqdm(pbar, total=len(train_dataloder))
        print(('\n' + '%11s' *(5)) % ('Epoch', 'GPU_mem', 'Cur_loss', 'avg_loss','lr'))

        for i,batch in pbar:
            optimizer.zero_grad()
            current_step=epoch*num_batches_per_epoch+i
            if current_step < warmup_steps:
                current_lr = lr * current_step / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            current_lr = optimizer.param_groups[0]['lr']
            
            
            encoded_batch=batch['input_ids']
            attention_mask=batch['attention_mask']
            labels=batch['labels']

            if opt.simple_classifier:
                label = torch.tensor(labels)
                loss,logits = model(batch,label)
            else:
                pass
            avg_loss=(avg_loss*i+loss.item())/(i+1)
            fabric.backward(loss)
            optimizer.step()
            if current_step >= warmup_steps:
                schedule.step()

            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            if fabric.global_rank == 0:
                pbar.set_description(
                    ('%11s' * 2 + '%11.4g' * 3) %
                    (f'{epoch + 1}/{opt.total_epoch}', mem, loss.item(),avg_loss, current_lr)
                )        
                if current_step%10==0:
                    writer.add_scalar('lr', current_lr, current_step)
                    writer.add_scalar('loss', loss.item(), current_step)
                    writer.add_scalar('avg_loss', avg_loss, current_step)
        
        with torch.no_grad():
            test_loss=0
            model.eval()
            pbar=enumerate(val_dataloder)
            if fabric.global_rank == 0 :
                pbar = tqdm(pbar, total=len(val_dataloder))
                print(('\n' + '%11s' *(5)) % ('Epoch', 'GPU_mem', 'Cur_acc', 'avg_acc','loss'))
            
            right_num, tot_num= 0,0
            for i, batch in pbar:
                encoded_batch=batch['input_ids']
                attention_mask=batch['attention_mask']
                labels=batch['labels']
                if opt.simple_classifier:
                    logits = model(batch, labels)[1]
                else:
                    pass
                _, predicted = torch.max(logits, dim=1)
                correct = (predicted == label).sum().item()
                right_num += correct
                tot_num += label.size(0)
                accuracy = correct / label.size(0)
                avg_acc=0.0
                avg_acc = (avg_acc * i + accuracy) / (i + 1)

                if fabric.global_rank == 0:
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * 2) %
                        (f'{epoch + 1}/{opt.total_epoch}', mem, accuracy, avg_acc)
                    )

            if fabric.global_rank == 0:
                writer.add_scalar('eval_loss', test_loss, epoch)
                writer.add_scalar('eval_acc', avg_acc, epoch)

            if avg_acc > max_avg_rec:
                max_avg_rec = avg_acc
                model.save_pretrained(os.path.join(opt.savedir, 'best_model'))
                print(f"New best model saved with accuracy: {avg_acc:.4f}")

    if fabric.global_rank == 0:
        model.save_pretrained(os.path.join(opt.savedir, 'final_model'))
        print(f"Final model saved with accuracy: {avg_acc:.4f}")
    torch.cuda.empty_cache()
    fabric.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_num', type=int, default=1, help="GPU number to use")
    parser.add_argument('--projection_size', type=int, default=768, help="Pretrained model output dim")
    parser.add_argument("--temperature", type=float, default=0.07, help="contrastive loss temperature")
    parser.add_argument('--num_workers', type=int, default=8, help="num_workers for dataloader")
    parser.add_argument("--per_gpu_batch_size", default=16, type=int, help="Batch size per GPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=16, type=int, help="Batch size per GPU for evaluation."
    )
    parser.add_argument('--human_dataset_path', type=str, default='/home/yx/yx_search/search/DeTeCtive-new/mydata/human', help="train,valid,test,test_ood")
    parser.add_argument('--gpt_dataset_path', type=str, default='/home/yx/yx_search/search/DeTeCtive-new/mydata/gptnone', help="train,valid,test,test_ood")

    '''abcd代表四个分类器的权重！损失函数为加权损失和'''
    parser.add_argument('--a', type=float, default=1)
    parser.add_argument('--b', type=float, default=1) 
    parser.add_argument('--c', type=float, default=1)
    parser.add_argument('--d', type=float, default=1,help="单词句子段落其他")
    parser.add_argument('--e', type=float, default=1)
    parser.add_argument('--f', type=float, default=1,help="classifier loss weight")

    parser.add_argument('--simple_classifier',type=bool,default=True,help="是否使用简单分类器")

    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay")
    parser.add_argument("--total_epoch", type=int, default=50, help="Total number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--resum", type=bool, default=False)
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--savedir", type=str, default="./runs")
    parser.add_argument("--name", type=str, default="trival")
    parser.add_argument('--model_name', type=str, default='/data/Content Moderation/unsup-simcse-roberta-base')
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1")
    parser.add_argument("--beta2", type=float, default=0.98, help="beta2")
    parser.add_argument("--eps", type=float, default=1e-6, help="eps")
    opt=parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
    train(opt)
