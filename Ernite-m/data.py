# 读取数据
import json
data_items_train = json.load(open("./data/dataset_items_train.json"))
data_items_val = json.load(open("./data/dataset_items_val.json"))
data_items_test = json.load(open("./data/dataset_items_test.json"))

import paddle
from paddle.vision import transforms as T
from paddle.io import Dataset
import json
from urllib.parse import urlparse
from PIL import Image
import os 
import imghdr

def process_string(input_str):
    input_str = input_str.replace('&#39;', ' ')
    input_str = input_str.replace('<b>','')
    input_str = input_str.replace('</b>','')
    #input_str = unidecode(input_str)  
    return input_str
    
class NewsContextDatasetEmbs(Dataset):
    def __init__(self, context_data_items_dict, queries_root_dir, split):
        self.context_data_items_dict = context_data_items_dict
        self.queries_root_dir = queries_root_dir
        self.idx_to_keys = list(context_data_items_dict.keys())
        self.transform =T.Compose([
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        self.split=split
    def __len__(self):
        return len(self.context_data_items_dict)   


    def load_img_pil(self,image_path):
        if imghdr.what(image_path) == 'gif': 
            try:
                with open(image_path, 'rb') as f:
                    img = Image.open(f)
                    return img.convert('RGB')
            except:
                return None 
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    def load_imgs_direct_search(self,item_folder_path,direct_dict):   
        list_imgs_tensors = []
        count = 0   
        keys_to_check = ['images_with_captions','images_with_no_captions','images_with_caption_matched_tags']
        for key1 in keys_to_check:
            if key1 in direct_dict.keys():
                for page in direct_dict[key1]:
                    image_path = os.path.join(item_folder_path,page['image_path'].split('/')[-1])
                    try:
                        pil_img = self.load_img_pil(image_path)
                    except Exception as e:
                        print(e)
                        print(image_path)
                    if pil_img == None: continue
                    transform_img = self.transform(pil_img)
                    count = count + 1 
                    list_imgs_tensors.append(transform_img)
        stacked_tensors = paddle.stack(list_imgs_tensors, axis=0)
        return stacked_tensors
    def load_captions(self,inv_dict):
        captions = ['']
        pages_with_captions_keys = ['all_fully_matched_captions','all_partially_matched_captions']
        for key1 in pages_with_captions_keys:
            if key1 in inv_dict.keys():
                for page in inv_dict[key1]:
                    if 'title' in page.keys():
                        item = page['title']
                        item = process_string(item)
                        captions.append(item)
                    
                    if 'caption' in page.keys():
                        sub_captions_list = []
                        unfiltered_captions = []
                        for key2 in page['caption']:
                            sub_caption = page['caption'][key2]
                            sub_caption_filter = process_string(sub_caption)
                            if sub_caption in unfiltered_captions: continue 
                            sub_captions_list.append(sub_caption_filter) 
                            unfiltered_captions.append(sub_caption) 
                        captions = captions + sub_captions_list 
                    
        pages_with_title_only_keys = ['partially_matched_no_text','fully_matched_no_text']
        for key1 in pages_with_title_only_keys:
            if key1 in inv_dict.keys():
                for page in inv_dict[key1]:
                    if 'title' in page.keys():
                        title = process_string(page['title'])
                        captions.append(title)
        return captions

    def load_captions_weibo(self,direct_dict):
        captions = ['']
        keys = ['images_with_captions','images_with_no_captions','images_with_caption_matched_tags']
        for key1 in keys:
            if key1 in direct_dict.keys():
                for page in direct_dict[key1]:
                    if 'page_title' in page.keys():
                        item = page['page_title']
                        item = process_string(item)
                        captions.append(item)
                    if 'caption' in page.keys():
                        sub_captions_list = []
                        unfiltered_captions = []
                        for key2 in page['caption']:
                            sub_caption = page['caption'][key2]
                            sub_caption_filter = process_string(sub_caption)
                            if sub_caption in unfiltered_captions: continue 
                            sub_captions_list.append(sub_caption_filter) 
                            unfiltered_captions.append(sub_caption) 
                        captions = captions + sub_captions_list 
        #print(captions)
        return captions
        #加载img文件夹
    def load_queries(self,key):
        caption = self.context_data_items_dict[key]['caption']
        image_path = os.path.join(self.queries_root_dir,self.context_data_items_dict[key]['image_path'])
        pil_img = self.load_img_pil(image_path)
        transform_img = self.transform(pil_img)
        return transform_img, caption
    def __getitem__(self, idx):
        #print(idx)
        #print(self.context_data_items_dict)      
        #idx = idx.tolist()               
        key = self.idx_to_keys[idx]
        #print(key)
        item=self.context_data_items_dict.get(str(key))
        #print(item)
        # 如果为test没有label属性
        #print(self.split)
        if self.split=='train' or self.split=='val':
            label = paddle.to_tensor(int(item['label']))
            direct_path_item = os.path.join(self.queries_root_dir,item['direct_path'])
            inverse_path_item = os.path.join(self.queries_root_dir,item['inv_path'])
            inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json')))
            direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json')))
            captions= self.load_captions(inv_ann_dict)
            captions += self.load_captions_weibo(direct_dict)
            imgs = self.load_imgs_direct_search(direct_path_item,direct_dict)     
            qImg,qCap =  self.load_queries(key)
            sample = {'label': label, 'caption': captions,'imgs': imgs,  'qImg': qImg, 'qCap': qCap}
        else:
            direct_path_item = os.path.join(self.queries_root_dir,item['direct_path'])
            inverse_path_item = os.path.join(self.queries_root_dir,item['inv_path'])
            inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json')))
            direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json')))
            captions= self.load_captions(inv_ann_dict)
            captions += self.load_captions_weibo(direct_dict)
            imgs = self.load_imgs_direct_search(direct_path_item,direct_dict)     
            qImg,qCap =  self.load_queries(key)
            sample = {'caption': captions,'imgs': imgs,  'qImg': qImg, 'qCap': qCap}
        #print(sample)
        #print(len(captions)) 
        #print(type(imgs))
        #print(imgs.size)
        #print(imgs.shape)  
        return sample,  len(captions), imgs.shape[0]

train_dataset = NewsContextDatasetEmbs(data_items_train, './data','train')
val_dataset = NewsContextDatasetEmbs(data_items_val,'./data','val')
test_dataset = NewsContextDatasetEmbs(data_items_test,'./data','test')

import paddle 
def collate_context_bert_train(batch):
    #print(batch)
    samples = [item[0] for item in batch]
    max_captions_len = max([item[1] for item in batch])
    max_images_len = max([item[2] for item in batch])
    qCap_batch = []
    qImg_batch = []
    img_batch = []
    cap_batch = []
    labels = [] 
    for j in range(0,len(samples)):  
        sample = samples[j]    
        labels.append(sample['label'])
        captions = sample['caption']
        cap_len = len(captions)
        for i in range(0,max_captions_len-cap_len):
            captions.append("")
        if len(sample['imgs'].shape) > 2:
            padding_size = (max_images_len-sample['imgs'].shape[0], sample['imgs'].shape[1], sample['imgs'].shape[2], sample['imgs'].shape[3])
        else:
            padding_size = (max_images_len-sample['imgs'].shape[0],sample['imgs'].shape[1])
        padded_mem_img = paddle.concat((sample['imgs'], paddle.zeros(padding_size)),axis=0)
        #print(1)
        img_batch.append(padded_mem_img)#pad证据图片
        cap_batch.append(captions)
        qImg_batch.append(sample['qImg'])#[3, 224, 224]
        qCap_batch.append(sample['qCap'])     
    #print(labels)   
    #print(img_batch)
    img_batch = paddle.stack(img_batch, axis=0)
    qImg_batch = paddle.stack(qImg_batch, axis=0)
    labels = paddle.stack(labels, axis=0) 
    #print(3)  
    return labels, cap_batch, img_batch, qCap_batch, qImg_batch

def collate_context_bert_test(batch):
    samples = [item[0] for item in batch]
    max_captions_len = max([item[1] for item in batch])
    max_images_len = max([item[2] for item in batch])
    qCap_batch = []
    qImg_batch = []
    img_batch = []
    cap_batch = []
    for j in range(0,len(samples)):  
        sample = samples[j]    
        captions = sample['caption']
        cap_len = len(captions)
        for i in range(0,max_captions_len-cap_len):
            captions.append("")
        if len(sample['imgs'].shape) > 2:
            padding_size = (max_images_len-sample['imgs'].shape[0],sample['imgs'].shape[1],sample['imgs'].shape[2],sample['imgs'].shape[3])
        else:
            padding_size = (max_images_len-sample['imgs'].shape[0],sample['imgs'].shape[1])
        padded_mem_img = paddle.concat((sample['imgs'], paddle.zeros(padding_size)),axis=0)
        img_batch.append(padded_mem_img)
        cap_batch.append(captions)
        qImg_batch.append(sample['qImg'])
        qCap_batch.append(sample['qCap'])        
    img_batch = paddle.stack(img_batch, axis=0)
    qImg_batch = paddle.stack(qImg_batch, axis=0)
    return cap_batch, img_batch, qCap_batch, qImg_batch

# load DataLoader
from paddle.io import DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn = collate_context_bert_train, return_list=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn = collate_context_bert_train,  return_list=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn = collate_context_bert_test, return_list=True)

