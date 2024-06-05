# -*- coding: utf-8 -*-

import os
import pickle
import sys
from typing import Any, Dict, Iterable, List, Optional, Union, Tuple
import unicodedata
import urllib
import zipfile
import mmap
import struct
from collections import defaultdict

import torch


def ispunct(token):
    return all(unicodedata.category(char).startswith('P') for char in token)


def isfullwidth(token):
    return all(unicodedata.east_asian_width(char) in ['W', 'F', 'A'] for char in token)


def islatin(token):
    return all('LATIN' in unicodedata.name(char) for char in token)


def isdigit(token):
    return all('DIGIT' in unicodedata.name(char) for char in token)


def tohalfwidth(token):
    return unicodedata.normalize('NFKC', token)


def bracket_translate(token):
    # Convert parentheses, brackets and converts them to PTB symbols.
    CONVERT_PARENTHESES = {
        "(": "-LRB-",
        ")": "-RRB-",
        "[": "-LSB-",
        "]": "-RSB-",
        "{": "-LCB-",
        "}": "-RCB-",
    }
    return CONVERT_PARENTHESES.get(token, token)


def get_non_eos_index(words, pad_index=0):
    """Get the index of non-eos tokens.

    Args:
        words (~torch.Tensor): [batch_size, n+2, *]
            adaptive to both WordField and SubwordField.
        pad_index (int): the index of padding token.

    Returns:
        index (~torch.Tensor): [batch_size, n+1]
    """
    batch_size, seq_len = words.shape[0], words.shape[1]
    # [batch_size, n+2, *]
    word_mask = words.ne(pad_index)
    # [batch_size, n+2]
    word_mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
    # [batch_size]
    lens = word_mask.sum(dim=-1) - 1
    # [batch_size, n+2]
    eos_mask = lens.new_tensor(range(seq_len)) != lens.view(-1, 1)
    # word index which is not equal to the index of eos
    # [batch_size, n+1]
    index = eos_mask.nonzero()[:, 1].reshape(batch_size, seq_len-1)

    return index


def stripe(x, n, w, offset=(0, 0), dim=1):
    r"""
    Returns a diagonal stripe of the tensor.

    Args:
        x (~torch.Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 1 if returns a horizontal stripe; 0 otherwise.

    Returns:
        a diagonal stripe of the tensor.

    Examples:
        >>> x = torch.arange(25).view(5, 5)
        >>> x
        tensor([[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24]])
        >>> stripe(x, 2, 3)
        tensor([[0, 1, 2],
                [6, 7, 8]])
        >>> stripe(x, 2, 3, (1, 1))
        tensor([[ 6,  7,  8],
                [12, 13, 14]])
        >>> stripe(x, 2, 3, (1, 1), 0)
        tensor([[ 6, 11, 16],
                [12, 17, 22]])
    """

    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[0, 0].numel()
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=stride,
                        storage_offset=(offset[0]*seq_len+offset[1])*numel)


def pad(tensors, padding_value=0, total_length=None, padding_side='right'):
    size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors)
                             for i in range(len(tensors[0].size()))]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[slice(-i, None) if padding_side == 'left' else slice(0, i) for i in tensor.size()]] = tensor
    return out_tensor


def download(url, reload=False):
    path = os.path.join(os.path.expanduser('~/.cache/supar'), os.path.basename(urllib.parse.urlparse(url).path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if reload:
        os.remove(path) if os.path.exists(path) else None
    if not os.path.exists(path):
        sys.stderr.write(f"Downloading: {url} to {path}\n")
        try:
            torch.hub.download_url_to_file(url, path, progress=True)
        except urllib.error.URLError:
            raise RuntimeError(f"File {url} unavailable. Please try other sources.")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as f:
            members = f.infolist()
            path = os.path.join(os.path.dirname(path), members[0].filename)
            if len(members) != 1:
                raise RuntimeError('Only one file(not dir) is allowed in the zipfile.')
            if reload or not os.path.exists(path):
                f.extractall(os.path.dirname(path))
    return path


def binarize(
    data: Union[List[str], Dict[str, Iterable]],
    fbin: str = None,
    merge: bool = False
) -> Tuple[str, torch.Tensor]:
    start, meta = 0, defaultdict(list)
    # the binarized file is organized as:
    # `data`: pickled objects
    # `meta`: a dict containing the pointers of each kind of data
    # `index`: fixed size integers representing the storage positions of the meta data
    with open(fbin, 'wb') as f:
        # in this case, data should be a list of binarized files
        if merge:
            for file in data:
                if not os.path.exists(file):
                    raise RuntimeError("Some files are missing. Please check the paths")
                mi = debinarize(file, meta=True)
                for key, val in mi.items():
                    val[:, 0] += start
                    meta[key].append(val)
                with open(file, 'rb') as fi:
                    length = int(sum(val[:, 1].sum() for val in mi.values()))
                    f.write(fi.read(length))
                start = start + length
            meta = {key: torch.cat(val) for key, val in meta.items()}
        else:
            for key, val in data.items():
                for i in val:
                    bytes = pickle.dumps(i)
                    f.write(bytes)
                    meta[key].append((start, len(bytes)))
                    start = start + len(bytes)
            meta = {key: torch.tensor(val) for key, val in meta.items()}
        pickled = pickle.dumps(meta)
        # append the meta data to the end of the bin file
        f.write(pickled)
        # record the positions of the meta data
        f.write(struct.pack('LL', start, len(pickled)))
    return fbin, meta


def debinarize(
    fbin: str,
    pos_or_key: Optional[Union[Tuple[int, int], str]] = (0, 0),
    meta: bool = False
) -> Union[Any, Iterable[Any]]:
    with open(fbin, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        if meta or isinstance(pos_or_key, str):
            length = len(struct.pack('LL', 0, 0))
            mm.seek(-length, os.SEEK_END)
            offset, length = struct.unpack('LL', mm.read(length))
            mm.seek(offset)
            if meta:
                return pickle.loads(mm.read(length))
            # fetch by key
            objs, meta = [], pickle.loads(mm.read(length))[pos_or_key]
            for offset, length in meta.tolist():
                mm.seek(offset)
                objs.append(pickle.loads(mm.read(length)))
            return objs
        # fetch by positions
        offset, length = pos_or_key
        mm.seek(offset)
        return pickle.loads(mm.read(length))
