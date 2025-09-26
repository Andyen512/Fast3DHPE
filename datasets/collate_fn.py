from torch.utils.data._utils.collate import default_collate

def collate_keep_seqname(batch):
    # batch: list of samples; each sample is a tuple of len 3 or 4
    # 兼容 3DHP(4项) / H36M(3项)
    if len(batch[0]) == 4:
        cams, b3d, b2d, seqs = zip(*batch)
    else:
        cams, b3d, b2d = zip(*batch)
        seqs = [None] * len(batch)

    cams = default_collate(cams)
    b3d  = default_collate(b3d)
    b2d  = default_collate(b2d)
    # seqs 保持为 list，不做 collate（允许元素是 str 或 None）
    return cams, b3d, b2d, list(seqs)