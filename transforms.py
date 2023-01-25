import torch
import torchvision
import cv2
import os


class VideoFolderPathToTensor(object): #renamed the class -- added 'Augmented' to the end

    def __init__(self, state, max_len=None):
        self.max_len = max_len
        self.state=state

    def __call__(self, path):

        file_names = sorted([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        frames_path = [os.path.join(path, f) for f in file_names]

        frame = cv2.imread(frames_path[0])
        height, width, channels = frame.shape
        num_frames = len(frames_path)

        transform_train= torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomApply([ # randomly apply color jitter
              torchvision.transforms.ColorJitter(brightness=.5, hue=.3), 
              ] , p=1.0),
            torchvision.transforms.RandomApply([ # randomly apply Gaussian blur
              torchvision.transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)),
            ], p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            torchvision.transforms.Resize([112, 112])])

        transform_test= torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            torchvision.transforms.Resize([112, 112])])

        # EXTRACT_FREQUENCY = 1
        EXTRACT_FREQUENCY = 15

        # num_time_steps = int(num_frames / EXTRACT_FREQUENCY)

        num_time_steps = 16
        # num_time_steps = 4

        # (3 x T x H x W), https://pytorch.org/docs/stable/torchvision/models.html
        frames = torch.FloatTensor(channels, num_time_steps, 112, 112)

        for index in range(0, num_time_steps):
            frame = cv2.imread(frames_path[index+EXTRACT_FREQUENCY])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)  # (H x W x C) to (C x H x W)
            frame = frame / 255
            if frame.shape[2] != 112:
                frame = frame[:, :, 80:560]
            if self.state=='train':
                frame = transform_train(frame)
            else:
                frame = transform_test(frame)
            frames[:, index, :, :] = frame.float()

        #return frames.permute(1, 0, 2, 3)
        return frames