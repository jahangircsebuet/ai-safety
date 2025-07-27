# test multijail dataset
CUDA_VISIBLE_DEVICES=2 python test_vicuna_and_other_model.py --model_name guanaco --multijail True
CUDA_VISIBLE_DEVICES=2 python test_vicuna_and_other_model.py --model_name falcon --multijail True
CUDA_VISIBLE_DEVICES=2 python test_vicuna_and_other_model.py --model_name dolphin --multijail True


