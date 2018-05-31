# train step 1
python ../../tools/train.py --solver solver-s.prototxt --weights ../../vgg16_20M_mc.caffemodel --gpu 0
python ../../tools/test-ms.py --model models/model-s_iter_8000.caffemodel --images list/train_aug_id.txt --dir /workspace/hzl/data/VOCdevkit/voc12 --output /workspace/hzl/data/VOCdevkit/voc12/DSRGOutput --gpu 0 --smooth true
#python ../../tools/evaluate.py --pred /workspace/hzl/data/VOCdevkit/voc12/DSRGOutput --gt /workspace/hzl/data/VOCdevkit/voc12/SegmentationClass --test_ids list/val_id.txt --save_path sec_seed_s.txt --class_num 21

# train step 2 (retrain)
python ../../tools/train.py --solver solver-f.prototxt --weights models/model-s_iter_8000.caffemodel --gpu 0
python ../../tools/test-ms-f.py --model models/model-f_iter_20000.caffemodel --images list/val_id.txt --dir /workspace/hzl/data/VOCdevkit/voc12 --output DSRG_final_output --gpu 0 --smooth true
python ../../tools/evaluate.py --pred DSRG_final_output --gt /workspace/hzl/data/VOCdevkit/voc12/SegmentationClass --test_ids list/val_id.txt --save_path sec_seed_a_8000.txt --class_num 21

# test
python ../../tools/test-ms-f.py --model models/model-f_iter_20000.caffemodel --images list/test_id.txt --dir /workspace/hzl/wsis-mask/data/voc12 --output DSRG_final_test_output --gpu 1 --smooth true
