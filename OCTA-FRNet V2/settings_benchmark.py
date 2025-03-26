from models.frnet import *


class ObjectCreator:
    def __init__(self, cls, args) -> None:
        self.cls_net = cls
        self.args = args

    def __call__(self):
        return self.cls_net(**self.args)

models = {
"FRNet-base": ObjectCreator(cls=FRNet, args=dict(
        ch_in=1, ch_out=1, cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock
    )),
"FRNet": ObjectCreator(cls=FRNet, args=dict(
        ch_in=1, ch_out=1, cls_init_block=RRCNNBlock, cls_conv_block=RecurrentConvNeXtBlock
    )),
    # 其他模型
}



