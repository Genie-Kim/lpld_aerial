
CUDA_VISIBLE_DEVICES=3 python tools/train_main.py \
    --config-file configs/sfda/sfda_dota2uavdt_debug.yaml \
    --model-dir source_model/dota_source_v11/best_mAP.pth

configs/sfda/sfda_dota2uavdt_IRG.yaml
configs/sfda/sfda_dota2uavdt_LPLD.yaml
configs/sfda/sfda_dota2uavdt_MT.yaml
configs/sfda/sfda_dota2visdrone_IRG.yaml
configs/sfda/sfda_dota2visdrone_LPLD.yaml
configs/sfda/sfda_dota2visdrone_MT.yaml
configs/sfda/sfda_gtav10k2dotagta_IRG.yaml
configs/sfda/sfda_gtav10k2dotagta_LPLD.yaml
configs/sfda/sfda_gtav10k2dotagta_MT.yaml

####################################################################

CUDA_VISIBLE_DEVICES=2 python tools/train_main.py \
    --config-file configs/sfda/sfda_dota2uavdt_LPLD.yaml \
    --model-dir source_model/dota_source_v11/best_mAP.pth

# 사실 그냥 sfda yaml로 돌려도 상관없음.
# box plot과 동시에 이미지 별 성능 csv 생성.
CUDA_VISIBLE_DEVICES=2 python tools/visualize_bbox.py \
    --config-file configs/sfda/ablation/ablation-sfda_dota2uavdt.yaml \
    --model-dir source_model/dota_source_v11/best_mAP.pth

# 사실 그냥 sfda yaml로 돌려도 상관없음.
# 데이터셋 전체 map 값만 output함. 빠르게 확인할 때 사용.
CUDA_VISIBLE_DEVICES=2 python tools/test_main.py \
    --config-file configs/sfda/ablation/ablation-sfda_dota2uavdt.yaml \
    --model-dir source_model/dota_source_v11/best_mAP.pth

# 사실 그냥 sfda yaml로 돌려도 상관없음.
CUDA_VISIBLE_DEVICES=2 python tools/visualize_tpfn.py \
    --config-file configs/sfda/ablation/ablation-sfda_dota2uavdt.yaml \
    --model-dir source_model/dota_source_v11/best_mAP.pth


# 무조건 ablation yaml로 돌려야함.
CUDA_VISIBLE_DEVICES=2 python tools/visualize_histogram.py \
    --config-file configs/sfda/ablation/ablation-sfda_dota2uavdt.yaml \
    --model-dir source_model/dota_source_v11/best_mAP.pth

# 무조건 ablation yaml로 돌려야함.
CUDA_VISIBLE_DEVICES=2 python tools/visualize_pl.py \
    --config-file configs/sfda/ablation/ablation-sfda_dota2uavdt.yaml \
    --model-dir source_model/dota_source_v11/best_mAP.pth


