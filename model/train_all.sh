#!/bin/bash
source activate tslib

#基本模块的有效性分析
# abla12 取消时间编码 1
# abla9 取消位置编码 1
# abla4 取消了聚类 1
# abla7 取消时序特征提取 1
# abla10 取消空间特征提取 1

#聚类消融实验
# abla6 使用位置编码和路由中心进行聚类（静态聚类）1
# abla8 使用节点路由和静态路由中心进行聚类 1

#路由机制的进一步分析
# abla1 取消了空间信息扩散模块（忽视时间步尺度上的的空间聚合） 1
# abla2 根据路由节点进行时间步级别的空间交互（忽视时滞） 1
# abla11 不进行节点路由增强 1

#空间聚合方式的比较分析
# abla3 逐个时间步计算相似度，进行空间加权（时间步级别的空间相关性，鲁棒性评估） 1
# abla5 使用位置编码作为节点路由，静态权重（类似GWN，全序列级别的空间相关性） 1

# abla13 abla3的基础上，不聚类（仅用于统计计算量） 1


# "PEMS04"  "PEMS07(M)" "PEMS07(L)" "METRLA" "PEMSBAY"
# "PEMS03" "PEMS04"  "PEMS08" "PEMS07(M)" "METRLA"
# "PEMS07" "PEMS07(L)"
datasets=("PEMS07(L)")
modes=("WEIGHT")
#  "SRCA_abla4"   "SRCA_abla1" "SRCA_abla2" "SRCA_abla3" "SRCA_abla5" "SRCA_abla6" "SRCA_abla7" "SRCA_abla8" "SRCA_abla9"  "SRCA_abla10" "SRCA_abla11" "SRCA_abla12"
models=("SRCA_abla1" "SRCA_abla2" "SRCA_abla3" "SRCA_abla5" "SRCA_abla6" "SRCA_abla7" "SRCA_abla8" "SRCA_abla9"  "SRCA_abla10" "SRCA_abla11" "SRCA_abla12")  # 添加更多模型

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do  # 遍历不同模型
      for mode in "${modes[@]}"; do
            echo "Training on $dataset with mode=$mode using $model..."
            python train.py -d $dataset -g 0 --mode $mode --model $model
            if [ $? -eq 0 ]; then
                echo "Successfully trained on $dataset with mode=$mode using $model"
            else
                echo "Error training on $dataset with mode=$mode using $model"
            fi
        done
    done
done