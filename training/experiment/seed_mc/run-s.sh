PASCAL_DIR=/workspace/hzl/data/VOCdevkit/voc12
GPU=1

# train step 1 (DSRG training)
python ../../tools/train.py --solver solver-s.prototxt --weights ../../vgg16_20M_mc.caffemodel --gpu ${GPU}
python ../../tools/test-ms.py --model models/model-s_iter_8000.caffemodel --images list/train_aug_id.txt --dir ${PASCAL_DIR} --output ${PASCAL_DIR}/DSRGOutput --gpu ${GPU} --smooth true

# train step 2 (retrain)
python ../../tools/train.py --solver solver-f.prototxt --weights models/model-s_iter_8000.caffemodel --gpu ${GPU}
#python ../../tools/test-ms-f.py --model models/model-f_iter_20000.caffemodel --images list/val_id.txt --dir /workspace/hzl/data/VOCdevkit/voc12 --output DSRG_final_output --gpu 0 --smooth true

# test
python ../../tools/test-ms-f.py --model models/model-f_iter_20000.caffemodel --images list/test_id.txt --dir ${PASCAL_DIR} --output DSRG_final_test_output --gpu ${GPU} --smooth true
python ../../tools/evaluate.py --pred DSRG_final_test_output --gt ${PASCAL_DIR}/SegmentationClass --test_ids list/test_id.txt --save_path DSRG_result_final.txt --class_num 21
