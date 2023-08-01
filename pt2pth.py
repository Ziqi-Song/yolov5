"""
Save pretrained weight into .pth
Note:
    weights from torch.hub is different from the downloaded one
"""

import torch


# Use weight from torch.hub
# pretrained_model = torch.hub.load("ultralytics/yolov5", "yolov5s")
# pretrain_state_dict = pretrained_model.state_dict()
# for idx, key in enumerate(pretrain_state_dict.keys()):
#     if idx == 0:
#         print(f"{idx} {key} {pretrain_state_dict[key].shape}")
#         print(pretrain_state_dict[key].cpu().detach().numpy()[0, 0, ::])

# Use weight from downloaded file
ckpt = torch.load("/Users/songziqi/Documents/Projects/yolov5/yolov5x.pt", map_location='cpu')
pretrained_model = ckpt['model']
state_dict = ckpt['model'].state_dict()
for idx, key in enumerate(state_dict.keys()):
    print("{:<5} {:<50} {:<30}".format(str(idx), str(key), str(state_dict[key].shape)))
    # if idx == 0:
        # print(f"{idx} {key} {state_dict[key].shape}")
        # print(state_dict[key].cpu().detach().numpy()[0, 0, ::])

checkpoint_path = "./yolov5x.pth"
torch.save({
            'model_state_dict': pretrained_model.state_dict(),
        }, checkpoint_path)




