import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type="instance", num_groups=1):
    if norm_type == "batch":
        # 一个Batch内不同样本间的标准化
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        # 一个样本内的标准化
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, affine=True, num_groups=num_groups)
    elif norm_type == "none":
        norm_layer = NoNorm
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_norm_args(norm_layer, nfeats_list):
    if hasattr(norm_layer, "__name__") and norm_layer.__name__ == "NoNorm":
        norm_args = [{"fake": True} for f in nfeats_list]
    elif norm_layer.func.__name__ == "GroupNorm":
        norm_args = [{"num_channels": f} for f in nfeats_list]
    elif norm_layer.func.__name__ == "BatchNorm":
        norm_args = [{"num_features": f} for f in nfeats_list]
    else:
        raise NotImplementedError(
            "normalization layer [%s] is not found" % norm_layer.func.__name__
        )
    return norm_args


# 这个是什么用途？
class NoNorm(nn.Module):  # todo with abstractclass and pass
    def __init__(self, fake=True):
        self.fake = fake
        super(NoNorm, self).__init__()

    def forward(self, x):
        return x

    def __call__(self, x):
        return self.forward(x)


# transformer需要调整学习率吗？和CNN有区别吗？
# 可参考https://www.msra.cn/zh-cn/news/features/pre-ln-transformer#:~:text=Warm-up%20%E6%98%AF%E5%8E%9F%E5%A7%8B%20Transformer%20%E7%BB%93%E6%9E%84%E4%BC%98%E5%8C%96%E6%97%B6%E7%9A%84%E4%B8%80%E4%B8%AA%E5%BF%85%E5%A4%87%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4%E7%AD%96%E7%95%A5%E3%80%82%20Transformer,%E7%BB%93%E6%9E%84%E5%AF%B9%E4%BA%8E%20warm-up%20%E7%9A%84%E8%B6%85%E5%8F%82%E6%95%B0%EF%BC%88%E6%8C%81%E7%BB%AD%E8%BD%AE%E6%95%B0%E3%80%81%E5%A2%9E%E9%95%BF%E6%96%B9%E5%BC%8F%E3%80%81%E5%88%9D%E5%A7%8B%E5%AD%A6%E4%B9%A0%E7%8E%87%E7%AD%89%EF%BC%89%E9%9D%9E%E5%B8%B8%E6%95%8F%E6%84%9F%EF%BC%8C%E8%8B%A5%E8%B0%83%E6%95%B4%E4%B8%8D%E6%85%8E%EF%BC%8C%E5%BE%80%E5%BE%80%E4%BC%9A%E4%BD%BF%E5%BE%97%E6%A8%A1%E5%9E%8B%E6%97%A0%E6%B3%95%E6%AD%A3%E5%B8%B8%E6%94%B6%E6%95%9B%E3%80%82%20%E5%9B%BE1%EF%BC%9A%E5%B8%B8%E8%A7%81%E7%9A%84%20warm-up%20%E5%AD%A6%E4%B9%A0%E7%8E%87%E7%AD%96%E7%95%A5
def get_scheduler(optimizer, opt):
    if opt.lr_policy == "lambda":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(
                opt.niter_decay + 1
            )
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1
        )
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", opt.lr_policy
        )
    return scheduler


# 如果是MedMeshCNN中，是否可以通过weight改进模型？
def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def define_classifier(
    input_nc, ncf, ninput_edges, nclasses, opt, gpu_ids, arch, init_type, init_gain
):
    net = None
    if arch == "mconvnet":
        net = MeshConvNet(
            norm_layer,
            input_nc,
            ncf,
            nclasses,
            ninput_edges,
            opt.pool_res,
            opt.fc_n,
            opt.resblocks,
        )
    elif arch == "meshunet":
        down_convs = [input_nc] + ncf
        up_convs = ncf[::-1] + [nclasses]
        pool_res = [ninput_edges] + opt.pool_res
        net = MeshEncoderDecoder(
            pool_res, down_convs, up_convs, blocks=opt.resblocks, transfer_data=True
        )
    # Transformer
    elif arch == "transformer":
        net = MeshTransformer(
            # input
            num_layers=6,
            model_dim=512,
            num_heads=8,
            ffn_dim=2048,
            dropout=0.2,
        )
    else:
        raise NotImplementedError("Encoder model name [%s] is not recognized" % arch)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_loss(opt):
    if opt.dataset_mode == "classification":
        loss = torch.nn.CrossEntropyLoss()
    elif opt.dataset_mode == "segmentation":
        weights = torch.FloatTensor(opt.weighted_loss)
        loss = torch.nn.CrossEntropyLoss(weights, ignore_index=-1)
    return loss
