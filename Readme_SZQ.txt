训练（注意修改coco128.yaml里面的数据集路径）：
python train.py --img 640 --epochs 1 --data coco128.yaml --weights yolov5s.pt --batch-size 1

测试
python val.py --weights yolov5s.pt --data coco128.yaml --img 640 --batch-size 2
