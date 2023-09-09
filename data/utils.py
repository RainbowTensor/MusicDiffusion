import torch
from .consts import EMPTY_INDEX

def generate_target(indices, valid_instr):
    sampled_target_instr_idx = torch.multinomial(valid_instr, 1)

    target_mask = torch.zeros_like(indices)

    batch_select = torch.arange(target_mask.size(0), dtype=torch.long)
    instr_select = sampled_target_instr_idx.chunk(chunks=target_mask.size(0), dim=0)

    target_mask[batch_select, instr_select] = 1

    target_empty = torch.empty_like(indices).fill_(EMPTY_INDEX)
    target = torch.where(
        target_mask == 1,
        indices,
        target_empty
    )

    return target, target_mask

def generate_source(indices, target_mask, valid_instr):
    # target_mask_reduced = target_mask.sum(-1).bool().float()

    sampled_source_instrs = torch.randint(0, 2, (indices.size(0), 6), dtype=torch.float, device=indices.device)
    sampled_source_instrs = torch.logical_and(valid_instr, sampled_source_instrs).to(torch.float)
    # sampled_source_instrs = (sampled_source_instrs - target_mask_reduced).clamp(0)
    source_mask = sampled_source_instrs[:, :, None].expand_as(indices)
    source_mask = torch.logical_or(source_mask, target_mask).long()

    source_empty = torch.empty_like(indices).fill_(EMPTY_INDEX)
    source = torch.where(
        source_mask == 1,
        indices,
        source_empty
    )

    return source, source_mask