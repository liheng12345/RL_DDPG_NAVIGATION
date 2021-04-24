from torch.utils.tensorboard import SummaryWriter
###tensorboard --logdir=logs

write = SummaryWriter("logs")
for i in range(100):
    write.add_scalar("y=x", i, i)
write.close()