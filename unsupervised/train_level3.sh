#! /bin/bash

rootdir=/raid/cs18btech11039/work/Sem_Seg/Domain-adaptation
arch=${rootdir}/archs/drn

cd $arch

datadir=${rootdir}/data/all_sources
savedir=${rootdir}/Baselines/unsupervised/source_only/level3/all/saves
listdir=${datadir}/level3_list

CUDA_VISIBLE_DEVICES=3,4,5,7  python3 segment_tb.py train -l ${listdir} -d ${datadir} -c 26 -s 896 \
    --arch drn_d_22 --batch-size 24 --epochs 20 --lr 0.001 --momentum 0.9 \
    --save_path ${savedir} 2>&1 | tee ${savedir}/level3_all
