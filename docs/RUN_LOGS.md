运行日志（关键片段）

cnn_main.py：训练与评估
- epoch: 0 train_correct: 36046 train_loss: 1566.1191 valid_correct 9852 valid_loss: 290.4605
- ...
- epoch: 9 train_correct: 43533 train_loss: 598.9821 valid_correct 10881 valid_loss: 151.3737
- Confusion_matrix: （略，参见终端输出）
- Overall accuracy on test set:  0.8802
- 模型保存：`./model1.pth`

evasion_attack.py：FGSM攻击
- Epsilon: 0      Test Accuracy = 8802 / 10000 = 0.8802
- Epsilon: 0.05   Test Accuracy = 3769 / 10000 = 0.3769
- Epsilon: 0.1    Test Accuracy = 1506 / 10000 = 0.1506
- Epsilon: 0.15   Test Accuracy = 752 / 10000 = 0.0752
- Epsilon: 0.2    Test Accuracy = 424 / 10000 = 0.0424
- Epsilon: 0.25   Test Accuracy = 269 / 10000 = 0.0269
- Epsilon: 0.3    Test Accuracy = 203 / 10000 = 0.0203
- 攻击曲线输出：`saved/accuracy_epsilon.png`
