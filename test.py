import torch
import argparse
from torch.utils.data import DataLoader
from src.text_embedding import TextEmbeddingModel
from utils.My_utils import load_MyData
from utils.Turing_utils import load_Turing
from src.dataset  import PassagesDataset
from lightning import Fabric
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
from transformers import AutoTokenizer
from utils.utils import compute_metrics
from lightning.fabric.strategies import DDPStrategy
from src.simclr import SimCLR_Classifier,SimCLR_Classifier_SCL
import pandas as pd
def collate_fn(batch):
    # 首先使用default_collate处理大部分情况
    text,label,attack_method,attack_method_set,id = default_collate(batch)
    encoded_batch = tokenizer.batch_encode_plus(
        text,
        return_tensors="pt",
        max_length=512,
        padding='max_length',
        truncation=True,
        )
    return encoded_batch,label,attack_method,attack_method_set,id

def test(opt):
    torch.set_float32_matmul_precision("medium")
    if opt.device_num > 1:
        ddp_strategy = DDPStrategy(find_unused_parameters=True)
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", 
                      devices=opt.device_num, strategy=ddp_strategy)
    else:
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed", 
                      devices=opt.device_num)
    fabric.launch()

    model = TextEmbeddingModel(opt.model_name).cuda()
    state_dict = torch.load(opt.model_path, map_location=fabric.device)
    # new_state_dict={}
    # for key in state_dict.keys():
    #     if key.startswith('model.'):
    #         new_key='model.'+key
    #         new_state_dict[new_key]=state_dict[key]
    # model.load_state_dict(new_state_dict)
    model.load_state_dict(state_dict)

    if opt.mode=='Turing':
        test_database = load_Turing(opt.test_dataset_path)[opt.test_dataset_name]
    elif opt.mode=='temp'or'checkgpt':
        #load_MyData返回的是一个字典
        test_database = load_MyData(opt.test_dataset_path)[opt.test_dataset_name]
    
    test_dataset = PassagesDataset(test_database,mode=opt.mode,need_ids=True)

    if opt.only_classifier:
        opt.a=opt.b=opt.c=0
        opt.d=1
        opt.one_loss=True

    if opt.one_loss:
        model = SimCLR_Classifier_SCL(opt, fabric)
    else:
        model = SimCLR_Classifier(opt, fabric)

    test_dataloder = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=True,drop_last=True,collate_fn=collate_fn,shuffle=True)
    test_loader = fabric.setup_dataloaders(test_dataloder)

    model = fabric.setup(model)
    model.eval()

    
    with torch.no_grad():
        total_samples = 0
        total_human_rec = 0.0
        total_machine_rec = 0.0
        total_avg_rec = 0.0
        total_acc = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        first_write = True
        for batch in tqdm(test_loader, desc="Testing"):
            encoded_batch, label, attack_method, attack_method_set, id = batch
            encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
            loss,out,k_out,k_outlabel = model(
    batch=encoded_batch,indices1=attack_method,
indices2=attack_method_set,label=label,id=id
)
            batch_size = len(k_outlabel)
            preds = torch.argmax(out, dim=1)
            k_outlabel = k_outlabel.cpu().numpy().astype(str)
            preds = preds.cpu().numpy().astype(str)
            id = id.cpu().numpy().astype(str)
            detail_df=pd.DataFrame({
                'id':id,
                'true_label':k_outlabel,
                'pred_label':preds
            })
            detail_df.to_csv(
            opt.detail_result_path,
            mode='a' if not first_write else 'w',
            header=first_write,
            index=False
        )
            first_write = False  # 首次写入后关闭header
            human_rec, machine_rec, avg_rec, acc, precision, recall, f1 = compute_metrics(k_outlabel, preds,id)

            total_human_rec += human_rec * batch_size
            total_machine_rec += machine_rec * batch_size
            total_avg_rec += avg_rec * batch_size
            total_acc += acc * batch_size
            total_precision += precision * batch_size
            total_recall += recall * batch_size
            total_f1 += f1 * batch_size
            total_samples += batch_size

            # print(f"HumanRec: {human_rec}, MachineRec: {machine_rec}, AvgRec: {avg_rec}, Acc:{acc}, Precision:{precision}, Recall:{recall}, F1:{f1}")
            # human_recs.append(human_rec)
            # machine_recs.append(machine_rec)
            # avg_recs.append(avg_rec)
            # accs.append(acc)
            # precisions.append(precision)
            # recalls.append(recall)
            # f1_scores.append(f1)
    final_human_rec = total_human_rec / total_samples
    final_machine_rec = total_machine_rec / total_samples
    final_avg_rec = total_avg_rec / total_samples
    final_acc = total_acc / total_samples
    final_precision = total_precision / total_samples
    final_recall = total_recall / total_samples
    final_f1 = total_f1 / total_samples

    # 打印最终指标
    print("\nFinal Weighted Metrics:")
    print(f"HumanRec: {final_human_rec:.2f}")
    print(f"MachineRec: {final_machine_rec:.2f}")
    print(f"AvgRec: {final_avg_rec:.2f}")
    print(f"Accuracy: {final_acc:.2f}")
    print(f"Precision: {final_precision:.2f}")
    print(f"Recall: {final_recall:.2f}")
    print(f"F1: {final_f1:.2f}")

    # 保存结果
    data = {
        'HumanRec': [final_human_rec],
        'MachineRec': [final_machine_rec],
        'AvgRec': [final_avg_rec],
        'Acc': [final_acc],
        'Precision': [final_precision],
        'Recall': [final_recall],
        'F1': [final_f1]
    }
    df = pd.DataFrame(data)
    df.to_csv(opt.save_result_path, index=False, mode='w')
    print(f"Test done. Saved weighted average results to {opt.save_result_path}")
    



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default="/data/Content Moderation/unsup-simcse-roberta-base", help="Model name")
  parser.add_argument('--test_dataset_path', type=str, default="/home/yx/yx_search/search/DeTeCtive-temp/detect_data/temp", help="Test dataset path")
  parser.add_argument('--test_dataset_name', type=str, default="test")
  parser.add_argument('--model_path', type=str, default="/home/yx/yx_search/search/DeTeCtive-temp/runs/simple-roberta-detector_v0/model_best.pth", help="Model path")
  parser.add_argument('--save_result_path', type=str, default="/home/yx/yx_search/search/DeTeCtive-temp/result.csv", help="Save result path")
  parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
  parser.add_argument('--num_workers', type=int, default=4, help="Number of workers")
  parser.add_argument('--device_num', type=int, default=1, help="Number of devices")
  parser.add_argument('--mode', type=str, default="temp", help="Mode")
  parser.add_argument("--one_loss",action='store_true',help="only use single contrastive loss")
  parser.add_argument("--only_classifier", action='store_true',help="only use classifier, no contrastive loss")
  parser.add_argument("--temperature", type=float, default=0.07, help="contrastive loss temperature")
  parser.add_argument('--a', type=float, default=1)
  parser.add_argument('--b', type=float, default=1) 
  parser.add_argument('--c', type=float, default=1)
  parser.add_argument('--d', type=float, default=1,help="classifier loss weight")
  parser.add_argument('--classifier_dim', type=int, default=2,help="classifier out dim")
  parser.add_argument('--projection_size', type=int, default=768, help="Pretrained model output dim")
  parser.add_argument("--resum", type=bool, default=False)
  parser.add_argument('--detail_result_path',type=str,default="/home/yx/yx_search/search/DeTeCtive-temp/detail_result.csv",help="detail result path")
  opt = parser.parse_args()
  tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
  test(opt)
