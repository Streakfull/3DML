from datasets.shape_net import ShapeNet
from tqdm.notebook import tqdm
import torch
import json
import numpy as np

class Evaluation():
     def __init__(self, dataloader, model, device, save_dir):
        self.dataloader = dataloader
        self.classes = ShapeNet.class_names
        self.results_dict = self.init_results_dict()
        self.device = device
        self.save_dir = save_dir
        self.model = model
        self.model.eval()
        
     
     def init_results_dict(self):
         results_dict = {}
         for class_name in self.classes:
                results_dict[class_name] = {
                    "total_iou": 0,
                    "total_instances": 0,
                    "top_ious": [],
                    "avg_iou": None
                }
         return results_dict
    
    
    
     def evaluate(self):
        for batch_idx, batch in tqdm(enumerate(self.dataloader)):
             ShapeNet.move_batch_to_device(batch, self.device)
             with torch.no_grad():
                self.model.inference(batch)
                self.map_results(batch)
        import pdb;pdb.set_trace()
    
     def map_results(self, batch):
         metrics = self.model.get_metrics(no_reduction=True)['iou']
         for idx, class_name in enumerate(batch["class"]):
                shape_iou = metrics[idx].item()
                shape_id = batch["id"][idx]
                self.results_dict[class_name]["total_iou"] += shape_iou
                self.results_dict[class_name]["total_instances"] += 1
                self.check_top_iou(class_name, shape_id, shape_iou)
         self.calculate_averages()
         import pdb;pdb.set_trace()

     
    
     def check_top_iou(self, class_name, shape_id, shape_iou):
           dict_entry = self.results_dict[class_name]
           top_ious = dict_entry["top_ious"]
           payload = { "shape_id": shape_id, "iou": shape_iou }
           if(len(top_ious) < 5):
             top_ious.append(payload)
             return
           min_value = np.inf
           min_index = 0
           for idx, iou in enumerate(top_ious):
                iou = top_ious[idx]["iou"]
                if(iou < min_value):
                    min_value = iou
                    min_index = idx
           if(shape_iou > min_value):
              top_ious[min_index] = payload
     
     def save_dict(self):
        with open(f"{self.save_dir}/evaluation.json", 'w') as fp:
            json.dump(self.results_dict, fp)
        print("Saved:" f"{self.save_dir}/evaluation.json")
        
     
     def calculate_averages(self):
        avgs = []
        for key in self.results_dict.keys():
            total_instances = self.results_dict[key]["total_instances"]
            total_iou = self.results_dict[key]["total_iou"]
            if(total_instances==0):
                continue
            avg = total_iou / total_instances
            self.results_dict[key]["avg_iou"] = avg
            avgs.append(avg)
        
        self.results_dict["overall_average"] = np.mean(np.array(avgs))
            
        
         