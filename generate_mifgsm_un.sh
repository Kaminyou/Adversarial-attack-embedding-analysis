#modellist="resnet18 resnet50 densenet121 wide_resnet50_2"
modellist="wide_resnet50_2"

for model in $modellist
do
    python3 generate.py --model_name $model --attack_name mifgsm_un >> mifigsm_un.log
done
