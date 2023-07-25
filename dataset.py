import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


EMOTION_LABEL= ['Anxiety', 'Peace', 'Weariness', 'Happiness', 'Anger']
DRIVER_BEHAVIOR_LABEL = ['Smoking', 'Making Phone', 'Looking Around', 'Dozing Off', 'Normal Driving', 'Talking', 'Body Movement']
SCENE_CENTRIC_CONTEXT_LABEL= ['Traffic Jam', 'Waiting', 'Smooth Traffic']
VEHICLE_BASED_CONTEXT_LABEL= ['Parking', 'Turning', 'Backward Moving', 'Changing Lane', 'Forward Moving']


class CarDataset(Dataset):
   
    def __init__(self, csv_file, transform=None):
     
        self.path = pd.read_csv(csv_file)
        self.transform = transform
        self.resize_height = 224
        self.resize_width = 224
        self.body_height = 112
        self.body_width = 112
        self.face_height = 64
        self.face_width = 64

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frames_path,label_path = self.path.iloc[idx]
        label_json = json.load(open(label_path))
        pose_list = label_json['pose_list']
        buffer, buffer_front, buffer_left, buffer_right, buffer_body, buffer_face, keypoints = self.load_frames(frames_path, pose_list)
        context = torch.cat([buffer, buffer_front, buffer_left, buffer_right], dim=0)
        context = self.randomflip(context)
        context = self.to_tensor(context)

        buffer_body = self.to_tensor(buffer_body)
        buffer_face = self.to_tensor(buffer_face)
        keypoints = keypoints.permute(2, 0, 1).contiguous()
        
        emotion_label = EMOTION_LABEL.index((label_json['emotion_label'].capitalize()))
        driver_behavior_label = DRIVER_BEHAVIOR_LABEL.index((label_json['driver_behavior_label']))
        scene_centric_context_label = SCENE_CENTRIC_CONTEXT_LABEL.index((label_json['scene_centric_context_label']))
        vehicle_based_context_label = VEHICLE_BASED_CONTEXT_LABEL.index((label_json['vehicle_based_context_label']))
        
        sample = {
            'context':context,
            'body':buffer_body,
            'face':buffer_face,
            'keypoints':torch.stack([keypoints],dim=-1),
            "emotion_label": emotion_label,
            "driver_behavior_label": driver_behavior_label,
            "scene_centric_context_label": scene_centric_context_label,
            "vehicle_based_context_label": vehicle_based_context_label
            }
 
        keypoint =  sample['keypoints']
        context = sample['context'] 
        body = sample['body'] 
        face = sample['face']  
        emotion_label = sample['emotion_label']  
        behavior_label = sample['driver_behavior_label']
        context_label = sample['scene_centric_context_label']
        vehicle_label = sample['vehicle_based_context_label']

        return keypoint, context, body, face, emotion_label, behavior_label, context_label, vehicle_label 

    def load_frames(self, file_dir,pose_list):

        incar_path = os.path.join(file_dir, 'incarframes')
        front_frames = os.path.join(file_dir, 'frontframes')
        left_frames = os.path.join(file_dir, 'leftframes')
        right_frames = os.path.join(file_dir, 'rightframes')

        frames = [os.path.join(incar_path, img) for img in os.listdir(incar_path) if img.endswith('.jpg')]
        front_frames = [os.path.join(front_frames, img) for img in os.listdir(front_frames) if img.endswith('.jpg')]
        left_frames = [os.path.join(left_frames, img) for img in os.listdir(left_frames) if img.endswith('.jpg')]
        right_frames = [os.path.join(right_frames, img) for img in os.listdir(right_frames) if img.endswith('.jpg')]

        frames.sort(key=lambda x:int(os.path.basename(x).split('.')[0]))
        front_frames.sort(key=lambda x:int(os.path.basename(x).split('.')[0]))
        left_frames.sort(key=lambda x:int(os.path.basename(x).split('.')[0]))
        right_frames.sort(key=lambda x:int(os.path.basename(x).split('.')[0]))
      
        buffer,buffer_front,buffer_left,buffer_right = [],[],[],[]
        buffer_body,buffer_face,keypoints = [],[],[]
        
        for i, frame_name in enumerate(frames):
            if not i == 0 and not i%3 == 2:
                continue
            if i >= 45:
                break

            img  = cv2.imread(frame_name)
            front_img = cv2.imread(front_frames[i])
            left_img = cv2.imread(left_frames[i])
            right_img = cv2.imread(right_frames[i])
            body = pose_list[i]['result'][0]['bbox']
            face = pose_list[i]['result'][0]['face_bbox']
            keypoint = np.array(pose_list[i]['result'][0]['keypoints']).reshape(-1,3)

            img_body = img[int(body[1]):int(body[1]+max(body[3],20)), int(body[0]):int(body[0]+max(body[2], 10))]
            img_face = img[int(face[1]):int(face[1]+max(face[3],10)), int(face[0]):int(face[0]+max(face[2], 10))]


            if img.shape[0]!=self.resize_height or img.shape[1]!=self.resize_width:
                img = cv2.resize(img, (self.resize_width, self.resize_height))
            if front_img.shape[0]!=self.resize_height or front_img.shape[1]!=self.resize_width:
                front_img = cv2.resize(front_img, (self.resize_width, self.resize_height))
            if left_img.shape[0]!=self.resize_height or left_img.shape[1]!=self.resize_width:
                left_img = cv2.resize(left_img, (self.resize_width, self.resize_height))
            if right_img.shape[0]!=self.resize_height or right_img.shape[1]!=self.resize_width:
                right_img = cv2.resize(right_img, (self.resize_width, self.resize_height))

            if img_body.shape[0]!=self.body_height or img_body.shape[1]!=self.body_width:
                img_body = cv2.resize(img_body, (self.body_width, self.body_height))
            try:
                if img_face.shape[0]!=self.face_height or img_face.shape[1]!=self.face_width:
                    img_face = cv2.resize(img_face, (self.face_width, self.face_height))
            except:
                img_face = img_body

  
            buffer.append(torch.from_numpy(img).float())
            buffer_front.append(torch.from_numpy(front_img).float())
            buffer_left.append(torch.from_numpy(left_img).float())
            buffer_right.append(torch.from_numpy(right_img).float())
            
            buffer_body.append(torch.from_numpy(img_body).float())
            buffer_face.append(torch.from_numpy(img_face).float())
            keypoints.append(torch.from_numpy(keypoint).float())

        return torch.stack(buffer), torch.stack(buffer_front), torch.stack(buffer_left), torch.stack(buffer_right), torch.stack(buffer_body), torch.stack(buffer_face), torch.stack(keypoints)

    def randomflip(self, buffer):

        if np.random.random() < 0.5:
          
            buffer  = torch.flip(buffer,dims=[1])
        
        if np.random.random() < 0.5:
           
            buffer  = torch.flip(buffer,dims=[2])

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer


    def to_tensor(self, buffer):
  
        return buffer.permute(3, 0, 1, 2).contiguous()

if __name__== "__main__":
    train_dataset = CarDataset(csv_file='training.csv')
    val_dataset = CarDataset(csv_file='validation.csv')
    test_dataset = CarDataset(csv_file='testing.csv')

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=16, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=16, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16, drop_last=False)
    for i_batch, sample in enumerate(val_dataloader):
        keypoint, context, body, face, emotion_label, behavior_label, context_label, vehicle_label = sample
        posture =  keypoint[:,:,:,:26,:]  
        gesture = keypoint[:,:,:,94:,:]
        print('posture:{}, gesture:{}, context:{}, body:{}, face:{}, emotion_label:{}, behavior_label:{}, context_label:{}, vehicle_label:{}' \
        .format(posture.shape, gesture.shape, context.shape, body.shape, face.shape, emotion_label.shape, behavior_label.shape, context_label.shape, vehicle_label.shape))
        

