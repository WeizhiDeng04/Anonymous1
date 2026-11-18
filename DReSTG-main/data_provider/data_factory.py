# ----------------------------------------------------------------------------
# DReSTG data provider: verified DReSTG native Reference + single clean Persistent path.
# ----------------------------------------------------------------------------
from data_provider.data_loader import Dataset_ETTh, Dataset_ETTm, Dataset_Custom
from data_provider.ref_dataset import DReSTGReferenceDataset, reference_collate
from torch.utils.data import DataLoader
from utils.ref_cache import base_contract, screen_contract


data_dict = {
    'ETTh1': Dataset_ETTh,
    'ETTh2': Dataset_ETTh,
    'ETTm1': Dataset_ETTm,
    'ETTm2': Dataset_ETTm,
    'custom': Dataset_Custom,
}


def _dataset_id(args):
    explicit = str(getattr(args, 'drestg_dataset_id', '') or '')
    if explicit:
        return explicit
    if args.data != 'custom':
        return str(args.data)
    raise ValueError('DReSTG custom datasets require --drestg_dataset_id (Weather/Electricity/Traffic)')


def _base_contract(args, flag):
    return base_contract(
        dataset=_dataset_id(args), split=str(flag), seq_len=int(args.seq_len),
        c=int(args.enc_in), l=int(args.coarse_patch_count), n=None,
        patch_len=int(args.fine_patch_len), stride=int(args.fine_patch_stride),
        padding_patch=str(args.padding_patch), eps=float(args.eps),
        scale=bool(args.scale), data_path=str(args.data_path), target=str(args.target),
    )


def _screen_contract(args, flag):
    return screen_contract(
        dataset=_dataset_id(args), split=str(flag), n=None, c=int(args.enc_in),
        mass_ratio=float(args.softk_mass_ratio),
    )


def data_provider(args, flag):
    Data = data_dict[args.data]
    shuffle_flag = False if flag == 'test' else True
    drop_last = True
    batch_size = args.batch_size

    base = Data(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        data_path=args.data_path,
        target=args.target,
        scale=args.scale,
    )
    wrapped = DReSTGReferenceDataset(
        base,
        args.drestg_reference_root,
        _dataset_id(args),
        flag,
        expected_base=_base_contract(args, flag),
        expected_screen=_screen_contract(args, flag),
    )
    print(flag, len(wrapped), 'DReSTG reference=', args.drestg_reference_root,
          'screen=raw-positive-softk', 'filter=none')
    loader = DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=reference_collate,
    )
    return wrapped, loader
