### baseline 完成

### 3.21 
加入checkpoint机制进行训练，cfg如下
>   # model config
    OUTPUT_STRIDE = 16
    ASPP_OUTDIM = 256
    SHORTCUT_DIM = 48
    SHORTCUT_KERNEL = 1
    NUM_CLASSES = 8

    # train config
    EPOCHS = 30
    WEIGHT_DECAY = 1.0e-4
    SAVE_PATH = "logs"
    BASE_LR = 0.0006
学习率调整策略如下
>   def adjust_lr(optimizer, epoch):
        if epoch == 0:
            lr = 1e-3
        elif epoch == 2:
            lr = 1e-2
        elif epoch == 100:
            lr = 1e-3
        elif epoch == 150:
            lr = 1e-4
        else:
            return
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

分辨率和ROI如下
>   image_size=(1024, 384), offset=690
测试结果和中间档保存在 "test_example/logs_2"中
Epoch:25, mask loss is 0.0553，并未明显收敛。根据正交化调参原则，应先朝降低训练集loss的方向进行改进。

### 3.21
恢复上次断点继续训练至50轮,设置与上次相同
发现loss下降不理想,根据学习率调整准备继续训练，另外断点恢复机制debug完：先将模型加入gpu，再加载optim
> 