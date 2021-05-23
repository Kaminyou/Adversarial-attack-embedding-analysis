modellist="resnet18 resnet50 densenet121 wide_resnet50_2"
for model in $modellist
do
    python3 generate.py --model_name $model --attack_name deepfool >> deepfool.log
done
