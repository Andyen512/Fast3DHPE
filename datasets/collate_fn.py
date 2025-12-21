from torch.utils.data._utils.collate import default_collate

def collate_keep_seqname(batch):
    """
    batch: list of samples
    sample: (cam, chunk_3d, chunk_2d, seqname/None, action)
    """
    cams, b3d, b2d, seqs, acts = zip(*batch)  

    cams = default_collate(cams)
    
    if all(x is None for x in b3d):
        b3d_batch = None
    else:
        b3d_batch = default_collate(b3d)

    b2d = default_collate(b2d)

    seqs = list(seqs)
    acts = list(acts)

    return cams, b3d_batch, b2d, seqs, acts