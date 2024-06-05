# -*- coding=utf-8 -*-

import os
from datetime import datetime

import torch
import torch.nn as nn

from supar.models.const2dep import Const2DepModel, Const2DepEmbeddingModel
from supar.parsers.dep import CRFDependencyParser, Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, pad, unk
from supar.utils.field import BoolChartField, Field, RawField, SubwordField
from supar.utils.fn import ispunct
from supar.utils.parallel import is_master
from supar.utils.logging import get_logger, progress_bar, init_logger
from supar.utils.metric import AttachmentMetric, DateMetric, LossMetric
from supar.utils.transform.const2dep import UnlabeledDepTree

logger = get_logger(__name__)


class Const2DepParser(CRFDependencyParser):
    r"""
    The implementation of first-order CRF Dependency Parser :cite:`zhang-etal-2020-efficient`.
    """

    NAME = 'const-2-dep'
    MODEL = Const2DepModel

    def __init__(self, *args, **kwargs):
        Parser.__init__(self, *args, **kwargs)
        self.WORD, self.CHAR, self.BERT = self.transform.WORD
        self.TAG = self.transform.POS

    def train(self, train, dev, test, buckets=32, batch_size=5000, update_steps=1,
              punct=False, mbr=False, tree=True, proj=True, partial=True, verbose=True, **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            punct (bool):
                If ``False``, ignores the punctuation during evaluation. Default: ``False``.
            mbr (bool):
                If ``True``, performs MBR decoding. Default: ``True``.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating training configs.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000, punct=False,
                 mbr=False, tree=True, proj=True, partial=True, verbose=True, **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            punct (bool):
                If ``False``, ignores the punctuation during evaluation. Default: ``False``.
            mbr (bool):
                If ``True``, performs MBR decoding. Default: ``True``.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating evaluation configs.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, lang=None, buckets=8, batch_size=5000, prob=False,
                mbr=False, tree=True, proj=True, verbose=True, **kwargs):
        r"""
        Args:
            data (list[list] or str):
                The data for prediction, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            mbr (bool):
                If ``True``, performs MBR decoding. Default: ``True``.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating prediction configs.

        Returns:
            A :class:`~supar.utils.Dataset` object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    def test(self, data, pred=None, lang=None, buckets=8, batch_size=5000,
             prob=False, tree=True, proj=True, partial=True, verbose=True, **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            punct (bool):
                If ``False``, ignores the punctuation during evaluation. Default: ``False``.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating evaluation configs.

        Returns:
            The loss scalar and evaluation results.
        """
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        if args.prob:
            self.transform.append(Field('probs'))
        self.transform.append(Field('arcs'))

        logger.info("Loading the data")
        dataset = Dataset(self.transform, data, proj=args.proj)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Test the dataset")
        start = datetime.now()
        preds = self._test(dataset.loader)
        elapsed = datetime.now() - start
        for name, value in preds.items():
            setattr(dataset, name, value)
        if pred is not None and is_master():
            logger.info(f"Saving predicted results to {pred}")
            os.makedirs(os.path.dirname(pred) or './', exist_ok=True)
            self.transform.save(pred, dataset.sentences)
        logger.info(f"{elapsed}s elapsed, {len(dataset)/elapsed.total_seconds():.2f} Sents/s")

        return dataset

    @classmethod
    def load(cls, path, reload=False, **kwargs):
        r"""
        Loads a parser with data fields and pretrained model parameters.

        Args:
            path (str):
                - a string with the shortcut name of a pretrained model defined in ``supar.MODEL``
                  to load from cache or download, e.g., ``'crf-dep-en'``.
                - a local path to a pretrained model, e.g., ``./<path>/model``.
            reload (bool):
                Whether to discard the existing cache and force a fresh download. Default: ``False``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating training configs and initializing the model.

        Examples:
            >>> from supar import Parser
            >>> parser = Parser.load('crf-dep-en')
            >>> parser = Parser.load('./ptb.crf.dep.lstm.char')
        """

        return super().load(path, reload, **kwargs)

    def _train(self, loader):
        self.model.train()

        bar = progress_bar(loader)

        for i, (words, *feats, trees, c_span_mask, combine_mask) in enumerate(bar, 1):
            print(words[0])
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            # print(mask.shape)
            # for tree in trees:
            #     print(tree.pformat(margin=1e10))
            #     word_list = tree.leaves()
            #     print(len(word_list))
            #     print(word_list)
            #     print(self.WORD.transform([word_list])[0])
            s_arc = self.model(words, feats)
            loss, s_arc = self.model.loss(s_arc, mask,
                                          c_span_mask if self.args.c_span_mask else None,
                                          combine_mask if self.args.combine_mask else None,
                                          self.args.mbr,
                                          self.args.partial)
            loss = loss / self.args.update_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
        logger.info(f"{bar.postfix}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, LossMetric()

        for words, *feats, trees, c_span_mask, combine_mask in loader:
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc = self.model(words, feats)
            loss, s_arc = self.model.loss(s_arc, mask,
                                          c_span_mask if self.args.c_span_mask else None,
                                          combine_mask if self.args.combine_mask else None,
                                          self.args.mbr,
                                          self.args.partial)
            # arc_preds = self.model.decode(s_arc, mask,
            #                               c_span_mask if self.args.c_span_mask else None,
            #                               combine_mask if self.args.combine_mask else None)
            total_loss += loss.item()
            # FIXME: compare with gold trees
            # metric(arc_preds, mask)
        total_loss /= len(loader)
        metric(total_loss)

        return total_loss, metric

    @torch.no_grad()
    def _test(self, loader):
        self.model.eval()

        results = {'arcs': [], 'probs': [] if self.args.prob else None}

        for words, *feats, trees, c_span_mask, combine_mask in loader:
            # print(trees[0].pformat(margin=1e10))
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc = self.model(words, feats)
            # arc_preds = self.model.decode(s_arc, mask,
            #                               c_span_mask if self.args.c_span_mask else None,
            #                               combine_mask if self.args.combine_mask else None)
            arc_preds = self.model.decode(s_arc, mask,
                                          c_span_mask,
                                          combine_mask)
            results['arcs'].extend(arc_preds[mask].split(lens))
            # FIXME: compare with gold trees
            # metric(arc_preds, mask)
        results['arcs'] = [seq.tolist() for seq in results['arcs']]

        return results

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()
        raise NotImplementedError
        preds = {'arcs': [], 'rels': [], 'probs': [] if self.args.prob else None}
        for words, *feats, trees, c_span_mask, combine_mask in progress_bar(loader):
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc = self.model(words, feats)
            if self.args.mbr:
                s_arc = self.model.crf(s_arc, mask, mbr=True)
            arc_preds, rel_preds = self.model.decode(s_arc, mask, self.args.tree, self.args.proj)
            preds['arcs'].extend(arc_preds[mask].split(lens))
            preds['rels'].extend(rel_preds[mask].split(lens))
            if self.args.prob:
                arc_probs = s_arc if self.args.mbr else s_arc.softmax(-1)
                preds['probs'].extend([prob[1:i+1, :i+1].cpu() for i, prob in zip(lens, arc_probs.unbind())])
        preds['arcs'] = [seq.tolist() for seq in preds['arcs']]
        preds['rels'] = [self.REL.vocab[seq.tolist()] for seq in preds['rels']]

        return preds

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary.
                Required if taking words as encoder input.
                Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        TAG, CHAR, BERT = None, None, None
        if args.encoder != 'lstm':
            from transformers import (AutoTokenizer, GPT2Tokenizer,
                                      GPT2TokenizerFast)
            t = AutoTokenizer.from_pretrained(args.bert)
            WORD = SubwordField('words',
                                pad=t.pad_token,
                                unk=t.unk_token,
                                bos=t.bos_token or t.cls_token,
                                fix_len=args.fix_len,
                                tokenize=t.tokenize,
                                fn=None if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast)) else lambda x: ' '+x)
            WORD.vocab = t.get_vocab()
        else:
            WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
            if 'tag' in args.feat:
                TAG = Field('tags', bos=bos)
            if 'char' in args.feat:
                CHAR = SubwordField('chars', pad=pad, unk=unk, bos=bos, fix_len=args.fix_len)
            if 'bert' in args.feat:
                from transformers import (AutoTokenizer, GPT2Tokenizer,
                                          GPT2TokenizerFast)
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = SubwordField('bert',
                                    pad=t.pad_token,
                                    unk=t.unk_token,
                                    bos=t.bos_token or t.cls_token,
                                    fix_len=args.fix_len,
                                    tokenize=t.tokenize,
                                    fn=None if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast)) else lambda x: ' '+x)
                BERT.vocab = t.get_vocab()

        TREE = RawField('trees')
        C_SPAN_MASK = BoolChartField('c_span_mask', use_vocab=False, fn=UnlabeledDepTree.get_c_span_mask)
        COMBINE_MASK = BoolChartField('combine_mask', use_vocab=False, fn=UnlabeledDepTree.get_combine_mask)
        transform = UnlabeledDepTree(WORD=(WORD, CHAR, BERT), POS=TAG, TREE=TREE, CONST=(C_SPAN_MASK, COMBINE_MASK))

        train = Dataset(transform, args.train)
        if args.encoder == 'lstm':
            WORD.build(train, args.min_freq, (Embedding.load(args.embed, args.unk) if args.embed else None), lambda x: x / torch.std(x))
            if TAG is not None:
                TAG.build(train)
            if CHAR is not None:
                CHAR.build(train)
        args.update({
            'n_words': len(WORD.vocab) if args.encoder != 'lstm' else WORD.vocab.n_init,
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed if hasattr(WORD, 'embed') else None).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)


class Const2DepEmbeddingParser(Const2DepParser):

    NAME = 'const-2-dep-embedding'
    MODEL = Const2DepEmbeddingModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.WORD, self.CHAR, self.BERT = self.transform.WORD
        self.TAG = self.transform.POS

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary.
                Required if taking words as encoder input.
                Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        TAG, CHAR, BERT = None, None, None
        from transformers import (AutoTokenizer, GPT2Tokenizer, GPT2TokenizerFast)
        t = AutoTokenizer.from_pretrained(args.bert)
        WORD = SubwordField('words',
                            pad=t.pad_token,
                            unk=t.unk_token,
                            bos=t.bos_token or t.cls_token,
                            fix_len=args.fix_len,
                            tokenize=t.tokenize,
                            fn=None if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast)) else lambda x: ' '+x)
        WORD.vocab = t.get_vocab()

        if args.feat is not None and 'tag' in args.feat:
            TAG = Field('tags', bos=bos)

        TREE = RawField('trees')
        C_SPAN_MASK = BoolChartField('c_span_mask', use_vocab=False, fn=UnlabeledDepTree.get_c_span_mask)
        COMBINE_MASK = BoolChartField('combine_mask', use_vocab=False, fn=UnlabeledDepTree.get_combine_mask)
        transform = UnlabeledDepTree(WORD=(WORD, CHAR, BERT), POS=TAG, TREE=TREE, CONST=(C_SPAN_MASK, COMBINE_MASK))

        train = Dataset(transform, args.train)
        if TAG is not None:
            TAG.build(train)

        args.update({
            'n_words': len(WORD.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)
