from model import NetWork
from data import train_dataloader, val_dataloader

# 声明模型
model = NetWork("image")
print(model)

# 训练配置
epochs = 2
num_training_steps = len(train_dataloader) * epochs
warmup_steps = int(num_training_steps * 0.1)
print(num_training_steps, warmup_steps)
# 定义 learning_rate_scheduler，负责在训练过程中对 lr 进行调度
lr_scheduler = LinearDecayWithWarmup(1e-6, num_training_steps, warmup_steps)
# 训练结束后，存储模型参数
save_dir = "checkpoint/"
best_dir = "best_model"
# 创建保存的文件夹
os.makedirs(save_dir, exist_ok=True)
os.makedirs(best_dir, exist_ok=True)

decay_params = [
    p.name
    for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

# 定义 Optimizer
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=1.2e-4,
    apply_decay_param_fun=lambda x: x in decay_params,
)

# 交叉熵损失
criterion = paddle.nn.loss.CrossEntropyLoss()

# 评估的时候采用准确率指标
metric = paddle.metric.Accuracy()


# 定义线下评估 评价指标为acc 线上评估是f1score
@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        labels, cap_batch, img_batch, qCap_batch, qImg_batch = batch
        logits = model(qCap=qCap_batch, qImg=qImg_batch, caps=cap_batch, imgs=img_batch)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
    model.train()
    metric.reset()
    return np.mean(losses), accu


# 定义训练
def do_train(model, criterion, metric, val_dataloader, train_dataloader):
    print("train run start")
    global_step = 0
    tic_train = time.time()
    best_accuracy = 0.0
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_dataloader, start=1):
            labels, cap_batch, img_batch, qCap_batch, qImg_batch = batch
            probs = model(
                qCap=qCap_batch, qImg=qImg_batch, caps=cap_batch, imgs=img_batch
            )
            loss = criterion(probs, labels)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1
            # 每间隔 100 step 输出训练指标
            if global_step % 100 == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                    % (
                        global_step,
                        epoch,
                        step,
                        loss,
                        acc,
                        10 / (time.time() - tic_train),
                    )
                )
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            # 每间隔一个epoch 在验证集进行评估
            if global_step % len(train_dataloader) == 0:
                eval_loss, eval_accu = evaluate(
                    model, criterion, metric, val_dataloader
                )
                save_param_path = os.path.join(
                    save_dir + str(epoch), "model_state.pdparams"
                )
                paddle.save(model.state_dict(), save_param_path)
                if best_accuracy < eval_accu:
                    best_accuracy = eval_accu
                    # 保存模型
                    save_param_path = os.path.join(best_dir, "model_best.pdparams")
                    paddle.save(model.state_dict(), save_param_path)


do_train(model, criterion, metric, val_dataloader, train_dataloader)
