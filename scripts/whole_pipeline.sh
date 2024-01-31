# extract features and save knn scores
python3 extract_feature_and_knn_score.py \
    --data_root data \
    --batch_size 64 \
    --dataset cifar10 \
    --model_name ViT \
    --model_path classifier_ckpts/cifar10/ViT600.pth \
    --save_only_ood_scores True \
    --top_k 50 \
    --number_tta 9 \
    --use_multithreading True \
#    --test_data_type natural gaussian_noise

# python extract_feature_and_knn_score.py --data_root data --batch_size 64 --dataset cifar10 --model_name ViT --model_path classifier_ckpts/cifar10/ViT600.pth --save_only_ood_scores True --top_k 50 --number_tta 9 --use_multithreading True --no_state_dict True

# run DAC (compare with ETS or SPL)
#combination_method=ETS # ETS or SPL
python3 run_experiment.py \
    --dataset cifar10 \
    --save_outputs_dir outputs/cifar10/resnet18 \
    --save_outputs_type ood_score \
    --ood_scoring_layers_list maxpool layer1 layer2 layer3 layer4 logits \
    --combination_method ETS \
    --test_data_type test

python run_experiment.py --dataset cifar10 --save_outputs_dir outputs/cifar10/ViT --save_outputs_type ood_score --ood_scoring_layers_list 'transformer.layers.0.0.fn.to_out.0' 'transformer.layers.1.0.fn.to_out.0' 'transformer.layers.2.0.fn.to_out.0' 'transformer.layers.3.0.fn.to_out.0' 'transformer.layers.4.0.fn.to_out.0' 'transformer.layers.5.0.fn.to_out.0' --combination_method ETS --test_data_type test