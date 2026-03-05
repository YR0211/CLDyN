import torch.nn as nn
from models.A2SI import A2SI

class RSC(nn.Module):
    def __init__(self, num_blocks, embed_dim, inC, filter_num, set_num, vector_num):
        super(RSC, self).__init__()
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            setattr(self, f"a2si{i}", A2SI(embed_dim=embed_dim, inC=inC, filter_num=filter_num, set_num=set_num, vector_num=vector_num))

    def forward_obo(self, basis_vector_bank, task_f, img_f, block_index: int):
        assert block_index <= self.num_blocks - 1 and block_index >= 0
        block = getattr(self, f"a2si{block_index}", "forward")
        out = block(basis_vector_bank, task_f, img_f)
        return out