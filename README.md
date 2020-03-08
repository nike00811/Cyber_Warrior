# Cyber_Warrior

執行訓練

```
python run_training.py \
--model_name bert-base-multilingual-cased \
--train_file ../dataset/WI_v1.json \
--epoch_time 100 \
--learning_rate 3e-5 \
--output_name ./training \
--batch_size 2 \
--seed 20 \
--save_epochs 5 \
```



使用模型

```
python use_model_v2.py --model_path ./Net_Army_v6/training_epoch_20/
```

![](https://i.imgur.com/I3lnvqN.png)
