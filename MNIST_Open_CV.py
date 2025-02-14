import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import os


class MNIST_NN(nn.Module):

    def __init__(self):
        super().__init__()

        #Convolutional Layers
        self.conv1 = nn.Conv2d(1,10,5,1)
        self.conv2 = nn.Conv2d(10,20,5,1)

        #Fully connected Layers
        self.fc1 = nn.Linear(320, 128)
        self.fc2 = nn.Linear(128 , 64)
        self.fc3 = nn.Linear(64 , 10)

    def forward(self,x):

        #Convolutional Layers and Pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)

        x = x.view(-1,320)

        #Fully connected Layers

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x,dim = 1)
    
def checkpoint():

    checkpoint = torch.load(r'models\best.pt',weights_only=True)
    return checkpoint

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MNIST_NN()
    model.to(device)
    model.load_state_dict(checkpoint())
    model.eval()


    cap = cv2.VideoCapture(0)

    while True:

        _,frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
 
        contours,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
                
                if cv2.contourArea(contour) > 200:

                    (x, y, w, h) = cv2.boundingRect(contour)
                    rect_0 = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
                    # rect_1 = cv2.rectangle(frame.copy(), (x, y), (x + w, y + h), (0, 255, 255), -1)
                    # frame_1 =cv2.addWeighted(rect_1,0.5, frame, .5, 0)
                    
                    # Calculating ROI with padding
                    x2 = h//8
                    y2 = h//10
                    roi = thresh[y-y2:y+h+y2, x-x2:x+w+x2]
                    
                    try:
                        roi = cv2.resize(roi, (20, 20))
                        roi = cv2.dilate(roi, (6, 6))
                        roi = cv2.copyMakeBorder(roi, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        roi = roi.astype('float32') / 255.0  #try np.invert(), in case if black background and white text
                        
                        roi_tensor = torch.FloatTensor(roi).unsqueeze(0).unsqueeze(0).to(device) 
                        
                        with torch.no_grad():
                            output = model(roi_tensor)
                            predicted = output.argmax().item()
                        
                        cv2.putText(frame, str(predicted), (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0,255), 2)

                    except:
                        continue

        cv2.imshow("frame",frame)
        cv2.imshow("thresh",thresh)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()