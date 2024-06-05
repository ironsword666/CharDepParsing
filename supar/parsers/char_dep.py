# -*- coding: utf-8 -*-

import os
from datetime import datetime

import dill
import supar
import torch
import torch.nn as nn
from supar.models import CharCRFDependencyModel, \
                         ExplicitCharCRFDependencyModel, \
                         LatentCharCRFDependencyModel, \
                         ClusterCharCRFDependencyModel, \
                         Coarse2FineCharCRFDependencyModel, SplitCoarse2FineCharCRFDependencyModel
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset
from supar.utils.common import bos, pad, unk, nul, BOS, PAD, UNK, NUL, ROOT, clusters, rel_cluster_map, get_rel2cluster
from supar.utils.field import RawField, Field, NGramField, ChartField, BoolChartField
from supar.utils.fn import ispunct, download
from supar.utils.logging import get_logger, progress_bar, init_logger
from supar.utils.parallel import is_master
from supar.utils.metric import CharAttachmentMetric, WordStructMetric
from supar.utils.transform.conll import CharCoNLL, Coarse2FineCharCoNLL, ExplicitCharCoNLL, MixedCharCoNLL, MixedCoarse2FineCharCoNLL
from supar.utils.transform.conll import LatentCharCoNLL

logger = get_logger(__name__)


class CharCRFDependencyParser(Parser):
    r"""
    The implementation of Biaffine Dependency Parser :cite:`dozat-etal-2017-biaffine`.
    """

    NAME = 'char-crf-dependency'
    MODEL = CharCRFDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.CHAR, self.BICHAR, self.TRICHAR, self.BERT = self.transform.FORM
        self.TAG = self.transform.CPOS

    def train(self, train, dev, test, buckets=32, batch_size=5000, update_steps=1,
              punct=False, tree=True, proj=True, partial=False, verbose=True, **kwargs):
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

    def evaluate(self, data, buckets=8, batch_size=5000,
                 punct=False, tree=True, proj=True, partial=False, verbose=True, **kwargs):
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
        logger.info("Loading the data")
        dataset = Dataset(self.transform, data)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Evaluating the dataset")
        start = datetime.now()
        loss, metric = self._evaluate(dataset.loader)
        elapsed = datetime.now() - start
        logger.info(f"loss: {loss:.4f} - {metric}")
        logger.info(f"{elapsed}s elapsed, {len(dataset)/elapsed.total_seconds():.2f} Sents/s")

        return loss, metric

    def test(self, data, pred=None, lang=None, buckets=8, batch_size=5000,
             prob=False, tree=True, proj=True, partial=False, verbose=True, **kwargs):
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
        self.transform.append(Field('id'))
        self.transform.append(Field('word'))
        self.transform.append(Field('head'))
        self.transform.append(Field('dep'))
        self.transform.append(Field('intra_word_struct'))
        self.transform.append(Field('intra_word_labels'))

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

    def predict(self, data, pred=None, lang=None, buckets=8, batch_size=5000, prob=False,
                tree=True, proj=True, verbose=True, **kwargs):
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

        # return super().predict(**Config().update(locals()))
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.eval()
        if args.prob:
            self.transform.append(Field('probs'))
        self.transform.append(Field('id'))
        self.transform.append(Field('word'))
        self.transform.append(Field('head'))
        self.transform.append(Field('dep'))
        self.transform.append(Field('intra_word_struct'))
        self.transform.append(Field('intra_word_labels'))

        logger.info("Loading the data")
        dataset = Dataset(self.transform, data, lang=lang)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Making predictions on the dataset")
        start = datetime.now()
        preds = self._predict(dataset.loader)
        elapsed = datetime.now() - start

        for name, value in preds.items():
            setattr(dataset, name, value)
        if pred is not None and is_master():
            logger.info(f"Saving predicted results to {pred}")
            os.makedirs(os.path.dirname(pred) or './', exist_ok=True)
            self.transform.save(pred, dataset.sentences)
        logger.info(f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s")

        return dataset

    @classmethod
    def load(cls, path, reload=False, **kwargs):
        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path if os.path.exists(path) else download(supar.MODEL.get(path, path), reload=reload))
        sd = state['state_dict']
        cls = supar.PARSER[state['name']] if cls.NAME is None else cls
        args = state['args'].update(args)
        model = cls.MODEL(**args)
        if state.get('pretrained_embed_dict', None) is not None:
            model.load_pretrained(state['pretrained_embed_dict'])
        model.load_state_dict(state['state_dict'], False)
        model.to(args.device)
        transform = state['transform']
        return cls(args, model, transform)

    @classmethod
    def load_from_word_parser(cls, path, transform_path, reload=False, **kwargs):
        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path if os.path.exists(path) else download(supar.MODEL.get(path, path), reload=reload))
        transform_state = torch.load(transform_path if os.path.exists(transform_path) else download(supar.MODEL.get(transform_path, transform_path), reload=reload))
        sd = state['state_dict']
        cls = supar.PARSER[state['name']] if cls.NAME is None else cls
        args = state['args'].update(args)
        args = transform_state['args'].update(args)
        model = cls.MODEL(**args)
        if state.get('pretrained_embed_dict', None) is not None:
            model.load_pretrained(state['pretrained_embed_dict'])
        model.load_state_dict(state['state_dict'], False)
        model.to(args.device)
        transform = transform_state['transform']
        return cls(args, model, transform)

    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        args = model.args
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained_embed_dict = {name: state_dict.pop(f'{name}.weight', None)
                                 for name in ('pretrained_char_embed', 'pretrained_bichar_embed', 'pretrained_trichar_embed')}
        state = {'name': self.NAME,
                 'args': args,
                 'state_dict': state_dict,
                 'pretrained_embed_dict': pretrained_embed_dict,
                 'transform': self.transform}
        torch.save(state, path, pickle_module=dill)

    def _train(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _evaluate(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _test(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _predict(self, loader):
        raise NotImplementedError

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        raise NotImplementedError


class ExplicitCharCRFDependencyParser(CharCRFDependencyParser):

    NAME = 'explicit-char-crf-dependency'
    MODEL = ExplicitCharCRFDependencyModel

    def __init__(self, *args, **kwargs):
        Parser.__init__(self, *args, **kwargs)

        self.CHAR, self.BICHAR, self.TRICHAR, self.BERT = self.transform.FORM
        self.TAG = self.transform.CPOS
        self.ARC, self.REL = self.transform.PHEAD, self.transform.PDEPREL

    def train(self, train, dev, test, buckets=32, batch_size=5000, update_steps=1,
              punct=False, tree=True, proj=True, partial=False, verbose=True, **kwargs):
        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000,
                 punct=False, tree=True, proj=True, partial=False, verbose=True, **kwargs):
        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, lang=None, buckets=8, batch_size=5000, prob=False,
                tree=True, proj=True, verbose=True, **kwargs):
        return super().predict(**Config().update(locals()))

    def test(self, data, pred=None, lang=None, buckets=8, batch_size=5000,
             prob=False, tree=True, proj=True, partial=False, verbose=True, **kwargs):
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
        self.transform.append(Field('id'))
        self.transform.append(Field('word'))
        self.transform.append(Field('head'))
        self.transform.append(Field('dep'))
        self.transform.append(Field('intra_word_struct'))
        self.transform.append(Field('intra_word_labels'))

        logger.info("Loading the data")
        dataset = Dataset(self.transform, data)
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
        return super().load(path, reload, **kwargs)

    def _train(self, loader):
        self.model.train()

        bar = progress_bar(loader)

        for i, batch in enumerate(bar, 1):
            chars, *feats, words, heads, arcs, rels = batch
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(chars, feats)
            loss, s_arc = self.model.loss(s_arc, s_rel, arcs, rels, mask, mbr=self.args.mbr, partial=self.args.partial)
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

        if not hasattr(self.args, 'intra_word_rel_indexes'):
            setattr(self.args, 'intra_word_rel_indexes', [self.args.nul_index])

        total_loss, metric = 0, CharAttachmentMetric()

        preds = {'arcs': [], 'rels': []}

        for batch in loader:
            chars, *feats, words, heads, arcs, rels = batch
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc, s_rel = self.model(chars, feats)
            loss, s_arc = self.model.loss(s_arc, s_rel, arcs, rels, mask, mbr=self.args.mbr, partial=self.args.partial)

            # transform word-level text to char-level text
            texts = [[unk] + [char for word in seq for char in word] for seq in words]
            nul_mask = rels.eq(self.REL.vocab[nul]) & mask if self.args.use_gold_seg else None
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, nul_mask=nul_mask, orientation=self.args.orientation, tree=self.args.tree, proj=self.args.proj)
            arc_preds = [seq.tolist() for seq in arc_preds[mask].split(lens)]
            rel_preds = [seq.tolist() for seq in rel_preds[mask].split(lens)]
            # transform char-level predictions to word-level predictions
            preds = [CharCoNLL.recover_words(arc_pred, rel_pred, self.args.intra_word_rel_indexes) for arc_pred, rel_pred in zip(arc_preds, rel_preds)]
            # pred_words = [[f"{''.join(text[b:e])} <-({self.REL.vocab[rel]}) {''.join(text[head[0]:head[1]])}" for (b, e), head, rel in pred] for pred, text in zip(preds, texts)]
            # remove punctuations
            preds = [[((b, e), head, rel, ispunct(''.join(text[b:e]))) for (b, e), head, rel in pred] for pred, text in zip(preds, texts)]

            arc_golds = [seq.tolist() for seq in arcs[mask].split(lens)]
            rel_golds = [seq.tolist() for seq in rels[mask].split(lens)]
            # transform char-level predictions to word-level predictions
            golds = [CharCoNLL.recover_words(arc_gold, rel_gold, self.args.intra_word_rel_indexes) for arc_gold, rel_gold in zip(arc_golds, rel_golds)]
            # gold_words = [[f"{''.join(text[b:e])} <-({self.REL.vocab[rel]}) {''.join(text[head[0]:head[1]])}" for (b, e), head, rel in pred] for pred, text in zip(golds, texts)]
            # remove punctuations
            golds = [[((b, e), head, rel, ispunct(''.join(text[b:e]))) for (b, e), head, rel in gold] for gold, text in zip(golds, texts)]

            total_loss += loss.item()
            metric(preds, golds)
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _test(self, loader):
        self.model.eval()

        if not hasattr(self.args, 'intra_word_rel_indexes'):
            setattr(self.args, 'intra_word_rel_indexes', [self.args.nul_index])

        results = {'id': [], 'word': [], 'head': [], 'dep': [], 'intra_word_struct': [], 'intra_word_labels': []}

        for batch in loader:
            chars, *feats, words, heads, arcs, rels = batch
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc, s_rel = self.model(chars, feats)
            loss, s_arc = self.model.loss(s_arc, s_rel, arcs, rels, mask, mbr=self.args.mbr, partial=self.args.partial)

            # transform word-level text to char-level text
            texts = [[unk] + [char for word in seq for char in word] for seq in words]
            nul_mask = rels.eq(self.REL.vocab[nul]) & mask if self.args.use_gold_seg else None
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, nul_mask=nul_mask, orientation=self.args.orientation, tree=self.args.tree, proj=self.args.proj)
            arc_preds = [seq.tolist() for seq in arc_preds[mask].split(lens)]
            rel_preds = [seq.tolist() for seq in rel_preds[mask].split(lens)]
            # transform char-level predictions to word-level predictions
            preds = [CharCoNLL.recover_words(arc_pred, rel_pred, self.args.intra_word_rel_indexes) for arc_pred, rel_pred in zip(arc_preds, rel_preds)]
            texts = [[bos] + [char for word in seq for char in word] for seq in words]

            for text, arc_pred, rel_pred, pred in zip(texts, arc_preds, rel_preds, preds):
                word_boundaries, head_boundaries, dependency_relations = list(zip(*pred))
                # add $root
                word_boundaries = [(0, 1)] + list(word_boundaries)
                results['id'].append([f"{i+1}" for i in range(len(word_boundaries)-1)])
                # get all word
                results['word'].append([''.join(text[begin:end]) for begin, end in word_boundaries[1:]])
                # get all head
                results['head'].append([word_boundaries.index(boundary) for boundary in head_boundaries])
                # get all dep relation
                results['dep'].append([self.REL.vocab[rel] for rel in dependency_relations])
                # get all intra word structure
                results['intra_word_struct'].append(['_'.join([str(arc_pred[i]-begin+1) if arc_pred[i] >= begin and arc_pred[i] < end else '0' for i in range(begin-1, end-1)]) for begin, end in word_boundaries[1:]])
                # get all intra word labels
                label_seq = [[self.REL.vocab[rel_pred[i]] if arc_pred[i] >= begin and arc_pred[i] < end else ROOT for i in range(begin-1, end-1)] for begin, end in word_boundaries[1:]]
                results['intra_word_labels'].append(['_'.join(label[2:] if label.startswith('c-') else label for label in labels) for labels in label_seq])
        return results

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.args.embed_dict).to(args.device)
            return parser

        logger.info("Building the fields")
        BICHAR, TRICHAR, BERT = None, None, None
        if args.encoder != 'lstm':
            from transformers import AutoTokenizer
            t = AutoTokenizer.from_pretrained(args.bert)
            CHAR = NGramField('chars',
                              pad=t.pad_token,
                              unk=t.unk_token,
                              bos=t.bos_token or t.cls_token)
            CHAR.vocab = t.get_vocab()
        else:
            CHAR = NGramField('chars', pad=PAD, unk=UNK, bos=BOS, ngram=1)
            if 'bichar' in args.feat:
                BICHAR = NGramField('bichars', pad=PAD, unk=UNK, bos=BOS, ngram=2)
            if 'trichar' in args.feat:
                TRICHAR = NGramField('trichars', pad=PAD, unk=UNK, bos=BOS, ngram=3)
            if 'bert' in args.feat:
                from transformers import AutoTokenizer
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = NGramField('bert',
                                  pad=t.pad_token,
                                  unk=t.unk_token,
                                  bos=t.bos_token or t.cls_token)
                BERT.vocab = t.get_vocab()

        WORD = RawField('words')
        HEAD = RawField('heads')
        ARC = Field('arcs', bos=bos, use_vocab=False, fn=ExplicitCharCoNLL.get_arcs)
        REL = Field('rels', bos=nul)
        transform = ExplicitCharCoNLL(args.orientation, FORM=(CHAR, BICHAR, TRICHAR, BERT), LEMMA=WORD, HEAD=HEAD, PHEAD=ARC, PDEPREL=REL)

        train = Dataset(transform, args.train)
        if args.encoder == 'lstm':
            dev = Dataset(transform, args.dev)
            test = Dataset(transform, args.test)
            datasets = [train, dev, test]
            char_embed = CHAR.tailor_embed(datasets, args.char_embed) if args.char_embed else None
            CHAR.build(train, args.min_freq, char_embed, norm=lambda x: x / torch.std(x))
            if BICHAR is not None:
                bichar_embed = BICHAR.tailor_embed(datasets, args.bichar_embed) if args.bichar_embed else None
                BICHAR.build(train, min_freq=3, embed=bichar_embed, norm=lambda x: x / torch.std(x))
            if TRICHAR is not None:
                trichar_embed = TRICHAR.tailor_embed(datasets, args.trichar_embed) if args.trichar_embed else None
                TRICHAR.build(train, min_freq=5, embed=trichar_embed, norm=lambda x: x / torch.std(x))
        REL.build(train)

        args.update({
            'n_chars': len(CHAR.vocab) if args.encoder != 'lstm' else CHAR.vocab.n_init,
            'n_rels': len(REL.vocab),
            'n_bichars': len(BICHAR.vocab) if BICHAR is not None else None,
            'n_trichars': len(TRICHAR.vocab) if TRICHAR is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': CHAR.pad_index,
            'unk_index': CHAR.unk_index,
            'bos_index': CHAR.bos_index,
            'nul_index': REL.vocab[nul],
            'intra_word_rel_indexes': [index for rel, index in REL.vocab.items() if rel == nul or rel.startswith('c-')],
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained({'pretrained_char_embed': CHAR.embed if hasattr(CHAR, 'embed') else None,
                                                   'pretrained_bichar_embed': BICHAR.embed if hasattr(BICHAR, 'embed') else None,
                                                   'pretrained_trichar_embed': TRICHAR.embed if hasattr(TRICHAR, 'embed') else None}).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)


class LatentCharCRFDependencyParser(CharCRFDependencyParser):

    NAME = 'latent-char-crf-dependency'
    MODEL = LatentCharCRFDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.CHAR, self.BICHAR, self.TRICHAR, self.BERT = self.transform.FORM
        self.TAG = self.transform.CPOS
        self.ARC, self.REL, *_ = self.transform.PHEAD

    def train(self, train, dev, test, buckets=32, batch_size=5000, update_steps=1,
              punct=False, tree=True, proj=True, partial=False, verbose=True, **kwargs):
        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000,
                 punct=False, tree=True, proj=True, partial=False, verbose=True, **kwargs):
        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, lang=None, buckets=8, batch_size=5000, prob=False,
                tree=True, proj=True, verbose=True, **kwargs):
        return super().predict(**Config().update(locals()))

    def _train(self, loader):
        self.model.train()

        bar = progress_bar(loader)

        for i, batch in enumerate(bar, 1):
            if self.args.combine_mask:
                chars, *feats, words, heads, arcs, rels, span_mask, combine_mask = batch
            else:
                chars, *feats, words, heads, arcs, rels, span_mask = batch
            intra_mask, inter_mask = span_mask[:, 0, ...], span_mask[:, 1, ...]
            c_span_mask = intra_mask | inter_mask
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(chars, feats)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, c_span_mask=c_span_mask if self.args.span_mask else None, combine_mask=combine_mask if self.args.combine_mask else None)
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

        if not hasattr(self.args, 'intra_word_rel_indexes'):
            setattr(self.args, 'intra_word_rel_indexes', [self.args.nul_index])

        total_loss, metric = 0, CharAttachmentMetric()
        # struct_metric = WordStructMetric()
        # total_conflict = 0

        for batch in loader:
            if self.args.combine_mask:
                chars, *feats, words, heads, arcs, rels, span_mask, combine_mask = batch
            else:
                chars, *feats, words, heads, arcs, rels, span_mask = batch
            intra_mask, inter_mask = span_mask[:, 0, ...], span_mask[:, 1, ...]
            c_span_mask = intra_mask | inter_mask
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc, s_rel = self.model(chars, feats)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, c_span_mask=c_span_mask if self.args.span_mask else None, combine_mask=combine_mask if self.args.combine_mask else None)

            texts = [[bos] + [char for word in seq for char in word] for seq in words]
            if self.args.use_gold_tree:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, intra_word_rel_indexes=self.args.intra_word_rel_indexes, c_span_mask=c_span_mask, combine_mask=combine_mask, intra_mask=intra_mask, arcs=arcs)
            elif self.args.use_gold_seg:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, intra_word_rel_indexes=self.args.intra_word_rel_indexes, c_span_mask=c_span_mask, combine_mask=combine_mask, intra_mask=intra_mask)
            elif self.args.constraint_decoding:
                arc_preds, rel_preds = self.model.constraint_decode(s_arc, s_rel, mask, intra_word_rel_indexes=self.args.intra_word_rel_indexes)
            else:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, intra_word_rel_indexes=self.args.intra_word_rel_indexes)
            arc_preds = [seq.tolist() for seq in arc_preds[mask].split(lens)]
            rel_preds = [seq.tolist() for seq in rel_preds[mask].split(lens)]
            # recover char-level predictions to word-level predictions
            preds = [CharCoNLL.recover_words(arc_pred, rel_pred, self.args.intra_word_rel_indexes) for arc_pred, rel_pred in zip(arc_preds, rel_preds)]
            # total_conflict += sum([1 if CharCoNLL.check_conflict(arc_pred, rel_pred, self.args.intra_word_rel_indexes) else 0 for arc_pred, rel_pred in zip(arc_preds, rel_preds)])
            # struct_metric([[((begin, end), [arc_pred[i-1] for i in range(begin, end)]) for (begin, end), heads, rel in pred] for pred, arc_pred in zip(preds, arc_preds)], texts)
            # pred_words = [[f"{''.join(text[b:e])} <-({self.REL.vocab[rel]}) {''.join(text[head[0]:head[1]])}" for (b, e), head, rel in pred] for pred, text in zip(preds, texts)]
            # remove punctuations
            # preds = [[((b, e), head, rel) for (b, e), head, rel in pred if not ispunct(''.join(text[b:e]))] for pred, text in zip(preds, texts)]
            preds = [[((b, e), head, rel, ispunct(''.join(text[b:e]))) for (b, e), head, rel in pred] for pred, text in zip(preds, texts)]

            golds = [LatentCharCoNLL.build_relations(ws, hs, rs) for ws, hs, rs in zip(words, heads, rels.tolist())]
            # gold_words = [[f"{''.join(text[b:e])} <-({self.REL.vocab[rel]}) {''.join(text[head[0]:head[1]])}" for (b, e), head, rel in gold] for gold, text in zip(golds, texts)]
            # remove punctuations
            golds = [[((b, e), head, rel, ispunct(''.join(text[b:e]))) for (b, e), head, rel in gold] for gold, text in zip(golds, texts)]

            total_loss += loss.item()
            metric(preds, golds)

        total_loss /= len(loader)
        # logger.info(f"total conflict sentence: {total_conflict}")
        # logger.info(f"word structures: {struct_metric}")

        return total_loss, metric

    @torch.no_grad()
    def _test(self, loader):
        self.model.eval()

        if not hasattr(self.args, 'intra_word_rel_indexes'):
            setattr(self.args, 'intra_word_rel_indexes', [self.args.nul_index])

        results = {'id': [], 'word': [], 'head': [], 'dep': [], 'intra_word_struct': [], 'intra_word_labels': []}

        for batch in loader:
            if self.args.combine_mask:
                chars, *feats, words, heads, arcs, rels, span_mask, combine_mask = batch
            else:
                chars, *feats, words, heads, arcs, rels, span_mask = batch
            intra_mask, inter_mask = span_mask[:, 0, ...], span_mask[:, 1, ...]
            c_span_mask = intra_mask | inter_mask
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc, s_rel = self.model(chars, feats)

            if self.args.use_gold_tree:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, intra_word_rel_indexes=self.args.intra_word_rel_indexes, c_span_mask=c_span_mask, combine_mask=combine_mask, intra_mask=intra_mask, arcs=arcs)
                marginals = self.model.crf.marginalize(s_arc, mask)
                constraint_marginals = self.model.crf.marginalize(s_arc, mask, target=arcs, c_span_mask=c_span_mask, combine_mask=combine_mask)
            elif self.args.use_gold_seg:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, intra_word_rel_indexes=self.args.intra_word_rel_indexes, c_span_mask=c_span_mask, combine_mask=combine_mask, intra_mask=intra_mask)
            elif self.args.constraint_decoding:
                arc_preds, rel_preds = self.model.constraint_decode(s_arc, s_rel, mask, intra_word_rel_indexes=self.args.intra_word_rel_indexes)
            else:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, intra_word_rel_indexes=self.args.intra_word_rel_indexes)
            arc_preds = [seq.tolist() for seq in arc_preds[mask].split(lens)]
            rel_preds = [seq.tolist() for seq in rel_preds[mask].split(lens)]
            # recover char-level predictions to word-level predictions
            preds = [CharCoNLL.recover_words(arc_pred, rel_pred, self.args.intra_word_rel_indexes) for arc_pred, rel_pred in zip(arc_preds, rel_preds)]
            texts = [[bos] + [char for word in seq for char in word] for seq in words]
            # for word, text, marginal, constraint_marginal in zip(words, texts, marginals, constraint_marginals):
            #     n_char = len(text)
            #     if n_char > 5 and n_char < 10:
            #         print(word)
            #         print(text)
            #         print(marginal[:n_char, :n_char])
            #         print(constraint_marginal[:n_char, :n_char])

            for text, arc_pred, rel_pred, pred in zip(texts, arc_preds, rel_preds, preds):
                word_boundaries, head_boundaries, dependency_relations = list(zip(*pred))
                # add $root
                word_boundaries = [(0, 1)] + list(word_boundaries)
                results['id'].append([f"{i+1}" for i in range(len(word_boundaries)-1)])
                # get all word
                results['word'].append([''.join(text[begin:end]) for begin, end in word_boundaries[1:]])
                # get all head
                results['head'].append([word_boundaries.index(boundary) for boundary in head_boundaries])
                # get all dep relation
                results['dep'].append([self.REL.vocab[rel] for rel in dependency_relations])
                # get all intra word structure
                results['intra_word_struct'].append(['_'.join([str(arc_pred[i]-begin+1) if arc_pred[i] >= begin and arc_pred[i] < end else '0' for i in range(begin-1, end-1)]) for begin, end in word_boundaries[1:]])
                # get all intra word labels
                label_seq = [[self.REL.vocab[rel_pred[i]] if arc_pred[i] >= begin and arc_pred[i] < end else ROOT for i in range(begin-1, end-1)] for begin, end in word_boundaries[1:]]
                results['intra_word_labels'].append(['_'.join(label[2:] if label.startswith('c-') else label for label in labels) for labels in label_seq])
        return results

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        if not hasattr(self.args, 'intra_word_rel_indexes'):
            setattr(self.args, 'intra_word_rel_indexes', [self.args.nul_index])

        results = {'id': [], 'word': [], 'head': [], 'dep': [], 'intra_word_struct': [], 'intra_word_labels': []}

        for batch in loader:
            chars, *feats, words = batch
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc, s_rel = self.model(chars, feats)

            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            arc_preds = [seq.tolist() for seq in arc_preds[mask].split(lens)]
            rel_preds = [seq.tolist() for seq in rel_preds[mask].split(lens)]
            # recover char-level predictions to word-level predictions
            preds = [CharCoNLL.recover_words(arc_pred, rel_pred, self.args.intra_word_rel_indexes) for arc_pred, rel_pred in zip(arc_preds, rel_preds)]
            texts = [[bos] + [char for word in seq for char in word] for seq in words]

            for text, arc_pred, rel_pred, pred in zip(texts, arc_preds, rel_preds, preds):
                word_boundaries, head_boundaries, dependency_relations = list(zip(*pred))
                # add $root
                word_boundaries = [(0, 1)] + list(word_boundaries)
                results['id'].append([f"{i+1}" for i in range(len(word_boundaries)-1)])
                # get all word
                results['word'].append([''.join(text[begin:end]) for begin, end in word_boundaries[1:]])
                # get all head
                results['head'].append([word_boundaries.index(boundary) for boundary in head_boundaries])
                # get all dep relation
                results['dep'].append([self.REL.vocab[rel] for rel in dependency_relations])
                # get all intra word structure
                results['intra_word_struct'].append(['_'.join([str(arc_pred[i]-begin+1) if arc_pred[i] >= begin and arc_pred[i] < end else '0' for i in range(begin-1, end-1)]) for begin, end in word_boundaries[1:]])
                # get all intra word labels
                label_seq = [[self.REL.vocab[rel_pred[i]] if arc_pred[i] >= begin and arc_pred[i] < end else ROOT for i in range(begin-1, end-1)] for begin, end in word_boundaries[1:]]
                results['intra_word_labels'].append(['_'.join(label[2:] if label.startswith('c-') else label for label in labels) for labels in label_seq])

        return results

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
            parser.model.load_pretrained(parser.args.embed_dict).to(args.device)
            return parser

        logger.info("Building the fields")
        BICHAR, TRICHAR, BERT = None, None, None
        if args.encoder != 'lstm':
            from transformers import AutoTokenizer
            t = AutoTokenizer.from_pretrained(args.bert)
            CHAR = NGramField('chars',
                              pad=t.pad_token,
                              unk=t.unk_token,
                              bos=t.bos_token or t.cls_token)
            CHAR.vocab = t.get_vocab()
        else:
            CHAR = NGramField('chars', pad=pad, unk=unk, bos=bos, lower=True, ngram=1)
            if 'bichar' in args.feat:
                BICHAR = NGramField('bichars', pad=pad, unk=unk, bos=bos, lower=True, ngram=2)
            if 'trichar' in args.feat:
                TRICHAR = NGramField('trichars', pad=pad, unk=unk, bos=bos, lower=True, ngram=3)
            if 'bert' in args.feat:
                from transformers import AutoTokenizer
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = NGramField('bert',
                                  pad=t.pad_token,
                                  unk=t.unk_token,
                                  bos=t.bos_token or t.cls_token)
                BERT.vocab = t.get_vocab()

        WORD = RawField('words')
        # TEXT = RawField('texts')
        HEAD = RawField('heads')
        ARC = BoolChartField('arcs', use_vocab=False, fn=LatentCharCoNLL.get_arcs)
        REL = ChartField('rels', unk=nul, fn=LatentCharCoNLL.get_rels)
        # SPAN_MASK = BoolChartField('span_mask', use_vocab=False, fn=LatentCharCoNLL.get_span_constraint) if args.span_mask else None
        SPAN_MASK = BoolChartField('span_mask', use_vocab=False, fn=LatentCharCoNLL.get_span_constraint)
        COMBINE_MASK = BoolChartField('combine_mask', use_vocab=False, fn=LatentCharCoNLL.get_combine_mask) if args.combine_mask else None
        # COMBINE_MASK = BoolChartField('combine_mask', use_vocab=False, fn=LatentCharCoNLL.get_combine_mask)
        transform = LatentCharCoNLL(FORM=(CHAR, BICHAR, TRICHAR, BERT), LEMMA=WORD, HEAD=HEAD, PHEAD=(ARC, REL, SPAN_MASK, COMBINE_MASK))

        train = Dataset(transform, args.train)
        if args.encoder == 'lstm':
            dev = Dataset(transform, args.dev)
            test = Dataset(transform, args.test)
            datasets = [train, dev, test]
            char_embed = CHAR.tailor_embed(datasets, args.char_embed) if args.char_embed else None
            CHAR.build(train, args.min_freq, char_embed, norm=lambda x: x / torch.std(x))
            if BICHAR is not None:
                bichar_embed = BICHAR.tailor_embed(datasets, args.bichar_embed) if args.bichar_embed else None
                BICHAR.build(train, min_freq=3, embed=bichar_embed, norm=lambda x: x / torch.std(x))
            if TRICHAR is not None:
                # FIXME: trichar 数量和 Yan的对不上
                trichar_embed = TRICHAR.tailor_embed(datasets, args.trichar_embed) if args.trichar_embed else None
                TRICHAR.build(train, min_freq=5, embed=trichar_embed, norm=lambda x: x / torch.std(x))
        REL.build(train)

        args.update({
            'n_chars': len(CHAR.vocab) if args.encoder != 'lstm' else CHAR.vocab.n_init,
            'n_rels': len(REL.vocab),
            'n_bichars': len(BICHAR.vocab) if BICHAR is not None else None,
            'n_trichars': len(TRICHAR.vocab) if TRICHAR is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': CHAR.pad_index,
            'unk_index': CHAR.unk_index,
            'bos_index': CHAR.bos_index,
            'nul_index': REL.vocab[nul],
            'intra_word_rel_indexes': [index for rel, index in REL.vocab.items() if rel == nul or rel.startswith('c-')],
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained({'pretrained_char_embed': CHAR.embed if hasattr(CHAR, 'embed') else None,
                                                   'pretrained_bichar_embed': BICHAR.embed if hasattr(BICHAR, 'embed') else None,
                                                   'pretrained_trichar_embed': TRICHAR.embed if hasattr(TRICHAR, 'embed') else None}).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)


class MixedCharCRFDependencyParser(LatentCharCRFDependencyParser):

    NAME = 'mixed-char-crf-dependency'
    MODEL = LatentCharCRFDependencyModel
    TRANSFORM = MixedCharCoNLL

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _train(self, loader):
        self.model.train()

        bar = progress_bar(loader)

        for i, batch in enumerate(bar, 1):
            chars, *feats, words, heads, deps, arcs, rels, span_mask, combine_mask = batch
            intra_mask, inter_mask = span_mask[:, 0, ...], span_mask[:, 1, ...]
            c_span_mask = intra_mask | inter_mask
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(chars, feats)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, c_span_mask=c_span_mask if self.args.span_mask else None, combine_mask=combine_mask if self.args.combine_mask else None)
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

        if not hasattr(self.args, 'intra_word_rel_indexes'):
            setattr(self.args, 'intra_word_rel_indexes', [self.args.nul_index])

        total_loss, metric = 0, CharAttachmentMetric()
        # struct_metric = WordStructMetric()
        total_conflict = 0

        preds = {'arcs': [], 'rels': []}

        for batch in loader:
            chars, *feats, words, heads, deps, arcs, rels, span_mask, combine_mask = batch
            intra_mask, inter_mask = span_mask[:, 0, ...], span_mask[:, 1, ...]
            c_span_mask = intra_mask | inter_mask
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc, s_rel = self.model(chars, feats)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, c_span_mask=c_span_mask if self.args.span_mask else None, combine_mask=combine_mask if self.args.combine_mask else None)

            texts = [[bos] + [char for word in seq for char in word] for seq in words]
            # print(f"texts:\n{texts[:2]}")
            if self.args.use_gold_seg:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, intra_word_rel_indexes=self.args.intra_word_rel_indexes, c_span_mask=c_span_mask, combine_mask=combine_mask, intra_mask=intra_mask)
            elif self.args.constraint_decoding:
                arc_preds, rel_preds = self.model.constraint_decode(s_arc, s_rel, mask, intra_word_rel_indexes=self.args.intra_word_rel_indexes)
            else:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, intra_word_rel_indexes=self.args.intra_word_rel_indexes)
            arc_preds = [seq.tolist() for seq in arc_preds[mask].split(lens)]
            # print(f"arc_preds:\n{arc_preds[:2]}")
            rel_preds = [seq.tolist() for seq in rel_preds[mask].split(lens)]
            # print(f"rel_preds:\n{rel_preds[:2]}")
            # recover char-level predictions to word-level predictions
            preds = [self.TRANSFORM.recover_words(arc_pred, rel_pred, self.args.intra_word_rel_indexes) for arc_pred, rel_pred in zip(arc_preds, rel_preds)]
            # total_conflict += sum([1 if CharCoNLL.check_conflict(arc_pred, rel_pred, self.args.intra_word_rel_indexes) else 0 for arc_pred, rel_pred in zip(arc_preds, rel_preds)])
            # struct_metric([[((begin, end), [arc_pred[i-1] for i in range(begin, end)]) for (begin, end), heads, rel in pred] for pred, arc_pred in zip(preds, arc_preds)])
            # print(f"preds:\n{preds[:2]}")
            # pred_words = [[f"{''.join(text[b:e])} <-({self.REL.vocab[rel]}) {''.join(text[head[0]:head[1]])}" for (b, e), head, rel in pred] for pred, text in zip(preds, texts)]
            # print(f"pred_words:\n{pred_words[:2]}")
            # remove punctuations
            # preds = [[((b, e), head, rel) for (b, e), head, rel in pred if not ispunct(''.join(text[b:e]))] for pred, text in zip(preds, texts)]
            preds = [[((b, e), head, rel, ispunct(''.join(text[b:e]))) for (b, e), head, rel in pred] for pred, text in zip(preds, texts)]

            golds = [self.TRANSFORM.build_relations(ws, hs, self.REL.vocab[rs]) for ws, hs, rs in zip(words, heads, deps)]
            # print(f"golds:\n{golds[:2]}")
            # gold_words = [[f"{''.join(text[b:e])} <-({self.REL.vocab[rel]}) {''.join(text[head[0]:head[1]])}" for (b, e), head, rel in gold] for gold, text in zip(golds, texts)]
            # print(f"gold_words:\n{gold_words[:2]}")
            # remove punctuations
            golds = [[((b, e), head, rel, ispunct(''.join(text[b:e]))) for (b, e), head, rel in gold] for gold, text in zip(golds, texts)]

            total_loss += loss.item()
            metric(preds, golds)

        total_loss /= len(loader)
        # logger.info(f"total conflict sentence: {total_conflict}")
        # logger.info(f"word structures: {struct_metric}")

        return total_loss, metric

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
            parser.model.load_pretrained(parser.args.embed_dict).to(args.device)
            return parser

        logger.info("Building the fields")
        BICHAR, TRICHAR, BERT = None, None, None
        if args.encoder != 'lstm':
            from transformers import AutoTokenizer
            t = AutoTokenizer.from_pretrained(args.bert)
            CHAR = NGramField('chars',
                              pad=t.pad_token,
                              unk=t.unk_token,
                              bos=t.bos_token or t.cls_token)
            CHAR.vocab = t.get_vocab()
        else:
            CHAR = NGramField('chars', pad=pad, unk=unk, bos=bos, lower=True, ngram=1)
            if 'bichar' in args.feat:
                BICHAR = NGramField('bichars', pad=pad, unk=unk, bos=bos, lower=True, ngram=2)
            if 'trichar' in args.feat:
                TRICHAR = NGramField('trichars', pad=pad, unk=unk, bos=bos, lower=True, ngram=3)
            if 'bert' in args.feat:
                from transformers import AutoTokenizer
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = NGramField('bert',
                                  pad=t.pad_token,
                                  unk=t.unk_token,
                                  bos=t.bos_token or t.cls_token)
                BERT.vocab = t.get_vocab()

        WORD = RawField('words')
        # TEXT = RawField('texts')
        HEAD = RawField('heads')
        DEP = RawField('deps')
        ARC = BoolChartField('arcs', use_vocab=False, fn=MixedCharCoNLL.get_arcs)
        REL = ChartField('rels', unk=nul, fn=MixedCharCoNLL.get_rels if not args.use_intra_rels else MixedCharCoNLL.get_intra_rels)
        # SPAN_MASK = BoolChartField('span_mask', use_vocab=False, fn=LatentCharCoNLL.get_span_constraint) if args.span_mask else None
        SPAN_MASK = BoolChartField('span_mask', use_vocab=False, fn=MixedCharCoNLL.get_span_constraint)
        # COMBINE_MASK = BoolChartField('combine_mask', use_vocab=False, fn=LatentCharCoNLL.get_combine_mask) if args.combine_mask else None
        COMBINE_MASK = BoolChartField('combine_mask', use_vocab=False, fn=MixedCharCoNLL.get_combine_mask)
        transform = MixedCharCoNLL(FORM=(CHAR, BICHAR, TRICHAR, BERT), LEMMA=WORD, HEAD=HEAD, DEPREL=DEP, PHEAD=(ARC, REL, SPAN_MASK, COMBINE_MASK))

        train = Dataset(transform, args.train)
        if args.encoder == 'lstm':
            dev = Dataset(transform, args.dev)
            test = Dataset(transform, args.test)
            datasets = [train, dev, test]
            char_embed = CHAR.tailor_embed(datasets, args.char_embed) if args.char_embed else None
            CHAR.build(train, args.min_freq, char_embed, norm=lambda x: x / torch.std(x))
            if BICHAR is not None:
                bichar_embed = BICHAR.tailor_embed(datasets, args.bichar_embed) if args.bichar_embed else None
                BICHAR.build(train, min_freq=3, embed=bichar_embed, norm=lambda x: x / torch.std(x))
            if TRICHAR is not None:
                # FIXME: trichar 数量和 Yan的对不上
                trichar_embed = TRICHAR.tailor_embed(datasets, args.trichar_embed) if args.trichar_embed else None
                TRICHAR.build(train, min_freq=5, embed=trichar_embed, norm=lambda x: x / torch.std(x))
        REL.build(train)

        print('rels', REL.vocab.stoi)

        args.update({
            'n_chars': len(CHAR.vocab) if args.encoder != 'lstm' else CHAR.vocab.n_init,
            'n_rels': len(REL.vocab),
            'n_bichars': len(BICHAR.vocab) if BICHAR is not None else None,
            'n_trichars': len(TRICHAR.vocab) if TRICHAR is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': CHAR.pad_index,
            'unk_index': CHAR.unk_index,
            'bos_index': CHAR.bos_index,
            'nul_index': REL.vocab[nul],
            'intra_word_rel_indexes': [index for rel, index in REL.vocab.items() if rel == nul or rel.startswith('c-')],
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained({'pretrained_char_embed': CHAR.embed if hasattr(CHAR, 'embed') else None,
                                                   'pretrained_bichar_embed': BICHAR.embed if hasattr(BICHAR, 'embed') else None,
                                                   'pretrained_trichar_embed': TRICHAR.embed if hasattr(TRICHAR, 'embed') else None}).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)


class ClusterCharCRFDependencyParser(MixedCharCRFDependencyParser):

    NAME = 'cluster-char-crf-dependency'
    MODEL = ClusterCharCRFDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            parser.model.load_pretrained(parser.args.embed_dict).to(args.device)
            return parser

        logger.info("Building the fields")
        BICHAR, TRICHAR, BERT = None, None, None
        if args.encoder != 'lstm':
            from transformers import AutoTokenizer
            t = AutoTokenizer.from_pretrained(args.bert)
            CHAR = NGramField('chars',
                              pad=t.pad_token,
                              unk=t.unk_token,
                              bos=t.bos_token or t.cls_token)
            CHAR.vocab = t.get_vocab()
        else:
            CHAR = NGramField('chars', pad=pad, unk=unk, bos=bos, lower=True, ngram=1)
            if 'bichar' in args.feat:
                BICHAR = NGramField('bichars', pad=pad, unk=unk, bos=bos, lower=True, ngram=2)
            if 'trichar' in args.feat:
                TRICHAR = NGramField('trichars', pad=pad, unk=unk, bos=bos, lower=True, ngram=3)
            if 'bert' in args.feat:
                from transformers import AutoTokenizer
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = NGramField('bert',
                                  pad=t.pad_token,
                                  unk=t.unk_token,
                                  bos=t.bos_token or t.cls_token)
                BERT.vocab = t.get_vocab()

        WORD = RawField('words')
        # TEXT = RawField('texts')
        HEAD = RawField('heads')
        DEP = RawField('deps')
        ARC = BoolChartField('arcs', use_vocab=False, fn=MixedCharCoNLL.get_arcs)
        REL = ChartField('rels', unk=nul, fn=MixedCharCoNLL.get_rels if not args.use_intra_rels else MixedCharCoNLL.get_intra_rels)
        # SPAN_MASK = BoolChartField('span_mask', use_vocab=False, fn=LatentCharCoNLL.get_span_constraint) if args.span_mask else None
        SPAN_MASK = BoolChartField('span_mask', use_vocab=False, fn=MixedCharCoNLL.get_span_constraint)
        # COMBINE_MASK = BoolChartField('combine_mask', use_vocab=False, fn=LatentCharCoNLL.get_combine_mask) if args.combine_mask else None
        COMBINE_MASK = BoolChartField('combine_mask', use_vocab=False, fn=MixedCharCoNLL.get_combine_mask)
        transform = MixedCharCoNLL(FORM=(CHAR, BICHAR, TRICHAR, BERT), LEMMA=WORD, HEAD=HEAD, DEPREL=DEP, PHEAD=(ARC, REL, SPAN_MASK, COMBINE_MASK))

        train = Dataset(transform, args.train)
        if args.encoder == 'lstm':
            dev = Dataset(transform, args.dev)
            test = Dataset(transform, args.test)
            datasets = [train, dev, test]
            char_embed = CHAR.tailor_embed(datasets, args.char_embed) if args.char_embed else None
            CHAR.build(train, args.min_freq, char_embed, norm=lambda x: x / torch.std(x))
            if BICHAR is not None:
                bichar_embed = BICHAR.tailor_embed(datasets, args.bichar_embed) if args.bichar_embed else None
                BICHAR.build(train, min_freq=3, embed=bichar_embed, norm=lambda x: x / torch.std(x))
            if TRICHAR is not None:
                # FIXME: trichar 数量和 Yan的对不上
                trichar_embed = TRICHAR.tailor_embed(datasets, args.trichar_embed) if args.trichar_embed else None
                TRICHAR.build(train, min_freq=5, embed=trichar_embed, norm=lambda x: x / torch.std(x))
        REL.build(train)

        print('rels', REL.vocab.stoi)
        rel2cluster = get_rel2cluster(REL.vocab.stoi, clusters, rel_cluster_map)
        assert len(rel2cluster) == len(REL.vocab), f'rel number mismatch ! {len(rel2cluster)} != {len(REL.vocab)}'

        args.update({
            'n_chars': len(CHAR.vocab) if args.encoder != 'lstm' else CHAR.vocab.n_init,
            'n_rels': len(REL.vocab),
            'n_bichars': len(BICHAR.vocab) if BICHAR is not None else None,
            'n_trichars': len(TRICHAR.vocab) if TRICHAR is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': CHAR.pad_index,
            'unk_index': CHAR.unk_index,
            'bos_index': CHAR.bos_index,
            'nul_index': REL.vocab[nul],
            'intra_word_rel_indexes': [index for rel, index in REL.vocab.items() if rel == nul or rel.startswith('c-')],
            'rel2cluster': rel2cluster,
            'n_clusters': len(clusters),
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained({'pretrained_char_embed': CHAR.embed if hasattr(CHAR, 'embed') else None,
                                                   'pretrained_bichar_embed': BICHAR.embed if hasattr(BICHAR, 'embed') else None,
                                                   'pretrained_trichar_embed': TRICHAR.embed if hasattr(TRICHAR, 'embed') else None}).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)


class Coarse2FineCharCRFDependencyParser(LatentCharCRFDependencyParser):

    NAME = 'c2f-char-crf-dependency'
    MODEL = Coarse2FineCharCRFDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _train(self, loader):
        self.model.train()

        bar = progress_bar(loader)

        for i, batch in enumerate(bar, 1):
            chars, *feats, words, heads, arcs, rels, span_mask = batch
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            # [batch_size, seq_len, seq_len]
            chart_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            # [batch_size, seq_len, seq_len, 2]
            arcs = torch.stack((rels.eq(self.args.nul_index), rels.ne(self.args.nul_index) & rels.ge(0)), dim=-1)
            intra_mask, inter_mask = span_mask[:, 0, ...], span_mask[:, 1, ...]
            # intra/inter complete span
            c_span_mask = torch.stack((intra_mask, inter_mask), dim=-1) if self.args.c_span_mask else None
            # intra/inter incomplete span
            i_span_mask = torch.stack((intra_mask, ~intra_mask & chart_mask), dim=-1) if self.args.i_span_mask else None
            s_arc, s_rel = self.model(chars, feats)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, c_span_mask=c_span_mask, i_span_mask=i_span_mask)
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

        if not hasattr(self.args, 'intra_word_rel_indexes'):
            setattr(self.args, 'intra_word_rel_indexes', [self.args.nul_index])

        total_loss, metric = 0, CharAttachmentMetric()

        for batch in loader:
            chars, *feats, words, heads, arcs, rels, span_mask = batch
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            # [batch_size, seq_len, seq_len]
            chart_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            # [batch_size, seq_len, seq_len, 2]
            arcs = torch.stack((rels.eq(self.args.nul_index), rels.ne(self.args.nul_index) & rels.ge(0)), dim=-1)
            intra_mask, inter_mask = span_mask[:, 0, ...], span_mask[:, 1, ...]
            # intra/inter complete span
            c_span_mask = torch.stack((intra_mask, inter_mask), dim=-1) if self.args.c_span_mask else None
            # intra/inter incomplete span
            i_span_mask = torch.stack((intra_mask, ~intra_mask & chart_mask), dim=-1) if self.args.i_span_mask else None
            s_arc, s_rel = self.model(chars, feats)
            # loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, c_span_mask=c_span_mask, i_span_mask=i_span_mask)

            if self.args.use_gold_tree:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, target=arcs, c_span_mask=c_span_mask, i_span_mask=i_span_mask)
            elif self.args.use_gold_seg:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, c_span_mask=c_span_mask, i_span_mask=i_span_mask)
            elif self.args.subword_parsing:
                sents = CharCoNLL.word2span(words)
                arc_preds, rel_preds = self.model.subword_decode(s_arc, s_rel, mask, sents, c_span_mask=c_span_mask, i_span_mask=i_span_mask)
            else:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            texts = [[unk] + [char for word in seq for char in word] for seq in words]
            if self.args.subword_parsing:
                preds = [LatentCharCoNLL.build_char_level_relations(ws, hs, rs) for ws, hs, rs in zip(words, arc_preds, rel_preds)]
                # remove punctuations
                preds = [[((b, e), head, rel, ispunct(''.join(text[b:e]))) for (b, e), head, rel in pred] for pred, text in zip(preds, texts)]
            else:
                arc_preds = [seq.tolist() for seq in arc_preds[mask].split(lens)]
                rel_preds = [seq.tolist() for seq in rel_preds[mask].split(lens)]
                # recover char-level predictions to word-level predictions
                preds = [CharCoNLL.recover_words(arc_pred, rel_pred, self.args.intra_word_rel_indexes) for arc_pred, rel_pred in zip(arc_preds, rel_preds)]
                # pred_words = [[f"{''.join(text[b:e])} <-({self.REL.vocab[rel]}) {''.join(text[head[0]:head[1]])}" for (b, e), head, rel in pred] for pred, text in zip(preds, texts)]
                # remove punctuations
                preds = [[((b, e), head, rel, ispunct(''.join(text[b:e]))) for (b, e), head, rel in pred] for pred, text in zip(preds, texts)]

            golds = [LatentCharCoNLL.build_relations(ws, hs, rs) for ws, hs, rs in zip(words, heads, rels.tolist())]
            # gold_words = [[f"{''.join(text[b:e])} <-({self.REL.vocab[rel]}) {''.join(text[head[0]:head[1]])}" for (b, e), head, rel in gold] for gold, text in zip(golds, texts)]
            # remove punctuations
            golds = [[((b, e), head, rel, ispunct(''.join(text[b:e]))) for (b, e), head, rel in gold] for gold, text in zip(golds, texts)]

            # total_loss += loss.item()
            metric(preds, golds)
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _test(self, loader):
        self.model.eval()

        if not hasattr(self.args, 'intra_word_rel_indexes'):
            setattr(self.args, 'intra_word_rel_indexes', [self.args.nul_index])

        results = {'id': [], 'word': [], 'head': [], 'dep': [], 'intra_word_struct': [], 'intra_word_labels': []}

        for batch in loader:
            chars, *feats, words, heads, arcs, rels, span_mask = batch
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            # [batch_size, seq_len, seq_len]
            chart_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            # [batch_size, seq_len, seq_len, 2]
            arcs = torch.stack((rels.eq(self.args.nul_index), rels.ne(self.args.nul_index) & rels.ge(0)), dim=-1)
            intra_mask, inter_mask = span_mask[:, 0, ...], span_mask[:, 1, ...]
            # intra/inter complete span
            c_span_mask = torch.stack((intra_mask, inter_mask), dim=-1) if self.args.c_span_mask else None
            # intra/inter incomplete span
            i_span_mask = torch.stack((intra_mask, ~intra_mask & chart_mask), dim=-1) if self.args.i_span_mask else None
            s_arc, s_rel = self.model(chars, feats)

            if self.args.use_gold_tree:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, target=arcs, c_span_mask=c_span_mask, i_span_mask=i_span_mask)
                # print(arc_preds.shape, rel_preds.shape)
                # print(arc_preds[0])
                marginals = self.model.crf.marginalize(s_arc, mask)
                constraint_marginals = self.model.crf.marginalize(s_arc, mask, target=arcs, c_span_mask=c_span_mask, i_span_mask=i_span_mask)
            elif self.args.use_gold_seg:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, c_span_mask=c_span_mask, i_span_mask=i_span_mask)
                marginals = self.model.crf.marginalize(s_arc, mask)
                constraint_marginals = self.model.crf.marginalize(s_arc, mask, c_span_mask=c_span_mask, i_span_mask=i_span_mask)
            else:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            arc_preds = [seq.tolist() for seq in arc_preds[mask].split(lens)]
            rel_preds = [seq.tolist() for seq in rel_preds[mask].split(lens)]
            [CharCoNLL.check_root_as_head(arc_pred, rel_pred, self.args.intra_word_rel_indexes) for arc_pred, rel_pred in zip(arc_preds, rel_preds)]

            # recover char-level predictions to word-level predictions
            preds = [CharCoNLL.recover_words(arc_pred, rel_pred, self.args.intra_word_rel_indexes) for arc_pred, rel_pred in zip(arc_preds, rel_preds)]
            texts = [[bos] + [char for word in seq for char in word] for seq in words]
            for word, text, marginal, constraint_marginal, s in zip(words, texts, marginals, constraint_marginals, s_arc):
                n_char = len(text)
                if n_char > 5 and n_char <= 10:
                    if all(len(w) < 3 for w in word):
                        continue
                    print(word)
                    print(text)
                    print(marginal[:n_char, :n_char, 0])
                    print(marginal[:n_char, :n_char, 1])
                    print(constraint_marginal[:n_char, :n_char, 0])
                    print(constraint_marginal[:n_char, :n_char, 1])
                    # s_intra = s[:n_char, :n_char, 0]
                    # print(s_intra)
                    # s_intra = s_intra.fill_diagonal_(-1e10)
                    # print(s_intra.softmax(-1))

            for text, arc_pred, rel_pred, pred in zip(texts, arc_preds, rel_preds, preds):
                word_boundaries, head_boundaries, dependency_relations = list(zip(*pred))
                # add $root
                word_boundaries = [(0, 1)] + list(word_boundaries)
                results['id'].append([f"{i+1}" for i in range(len(word_boundaries)-1)])
                # get all word
                results['word'].append([''.join(text[begin:end]) for begin, end in word_boundaries[1:]])
                # get all head
                results['head'].append([word_boundaries.index(boundary) for boundary in head_boundaries])
                # get all dep relation
                results['dep'].append([self.REL.vocab[rel] for rel in dependency_relations])
                # get all intra word structure
                results['intra_word_struct'].append(['_'.join([str(arc_pred[i]-begin+1) if arc_pred[i] >= begin and arc_pred[i] < end else '0' for i in range(begin-1, end-1)]) for begin, end in word_boundaries[1:]])
                # get all intra word labels
                label_seq = [[self.REL.vocab[rel_pred[i]] if arc_pred[i] >= begin and arc_pred[i] < end else ROOT for i in range(begin-1, end-1)] for begin, end in word_boundaries[1:]]
                results['intra_word_labels'].append(['_'.join(label[2:] if label.startswith('c-') else label for label in labels) for labels in label_seq])
        return results

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
            parser.model.load_pretrained(parser.args.embed_dict).to(args.device)
            return parser

        logger.info("Building the fields")
        BICHAR, TRICHAR, BERT = None, None, None
        if args.encoder != 'lstm':
            from transformers import AutoTokenizer
            t = AutoTokenizer.from_pretrained(args.bert)
            CHAR = NGramField('chars',
                              pad=t.pad_token,
                              unk=t.unk_token,
                              bos=t.bos_token or t.cls_token)
            CHAR.vocab = t.get_vocab()
        else:
            CHAR = NGramField('chars', pad=pad, unk=unk, bos=bos, lower=True, ngram=1)
            if 'bichar' in args.feat:
                BICHAR = NGramField('bichars', pad=pad, unk=unk, bos=bos, lower=True, ngram=2)
            if 'trichar' in args.feat:
                TRICHAR = NGramField('trichars', pad=pad, unk=unk, bos=bos, lower=True, ngram=3)
            if 'bert' in args.feat:
                from transformers import AutoTokenizer
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = NGramField('bert',
                                  pad=t.pad_token,
                                  unk=t.unk_token,
                                  bos=t.bos_token or t.cls_token)
                BERT.vocab = t.get_vocab()

        WORD = RawField('words')
        # TEXT = RawField('texts')
        HEAD = RawField('heads')
        ARC = BoolChartField('arcs', use_vocab=False, fn=LatentCharCoNLL.get_arcs)
        REL = ChartField('rels', unk=nul, fn=LatentCharCoNLL.get_rels)
        SPAN_MASK = BoolChartField('span_mask', use_vocab=False, fn=LatentCharCoNLL.get_span_constraint)
        transform = Coarse2FineCharCoNLL(FORM=(CHAR, BICHAR, TRICHAR, BERT), LEMMA=WORD, HEAD=HEAD, PHEAD=(ARC, REL, SPAN_MASK))

        train = Dataset(transform, args.train)
        if args.encoder == 'lstm':
            dev = Dataset(transform, args.dev)
            test = Dataset(transform, args.test)
            datasets = [train, dev, test]
            char_embed = CHAR.tailor_embed(datasets, args.char_embed) if args.char_embed else None
            CHAR.build(train, args.min_freq, char_embed, norm=lambda x: x / torch.std(x))
            if BICHAR is not None:
                bichar_embed = BICHAR.tailor_embed(datasets, args.bichar_embed) if args.bichar_embed else None
                BICHAR.build(train, min_freq=3, embed=bichar_embed, norm=lambda x: x / torch.std(x))
            if TRICHAR is not None:
                # FIXME: trichar 数量和 Yan的对不上
                trichar_embed = TRICHAR.tailor_embed(datasets, args.trichar_embed) if args.trichar_embed else None
                TRICHAR.build(train, min_freq=5, embed=trichar_embed, norm=lambda x: x / torch.std(x))
        REL.build(train)

        args.update({
            'n_chars': len(CHAR.vocab) if args.encoder != 'lstm' else CHAR.vocab.n_init,
            'n_rels': len(REL.vocab),
            'n_bichars': len(BICHAR.vocab) if BICHAR is not None else None,
            'n_trichars': len(TRICHAR.vocab) if TRICHAR is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': CHAR.pad_index,
            'unk_index': CHAR.unk_index,
            'bos_index': CHAR.bos_index,
            'nul_index': REL.vocab[nul],
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained({'pretrained_char_embed': CHAR.embed if hasattr(CHAR, 'embed') else None,
                                                   'pretrained_bichar_embed': BICHAR.embed if hasattr(BICHAR, 'embed') else None,
                                                   'pretrained_trichar_embed': TRICHAR.embed if hasattr(TRICHAR, 'embed') else None}).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)


class SplitCoarse2FineCharCRFDependencyParser(Coarse2FineCharCRFDependencyParser):

    NAME = 'split-c2f-char-crf-dependency'
    MODEL = SplitCoarse2FineCharCRFDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _train(self, loader):
        self.model.train()

        bar = progress_bar(loader)

        for i, batch in enumerate(bar, 1):
            chars, *feats, words, heads, arcs, rels, span_mask = batch
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            # [batch_size, seq_len, seq_len]
            chart_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            # [batch_size, seq_len, seq_len, 2]
            arcs = torch.stack((rels.eq(self.args.nul_index), rels.ne(self.args.nul_index) & rels.ge(0)), dim=-1)
            intra_mask, inter_mask = span_mask[:, 0, ...], span_mask[:, 1, ...]
            # intra/inter complete span
            c_span_mask = torch.stack((intra_mask, inter_mask), dim=-1) if self.args.c_span_mask else None
            # intra/inter incomplete span
            i_span_mask = torch.stack((intra_mask, ~intra_mask & chart_mask), dim=-1) if self.args.i_span_mask else None
            s_arc, s_rel, orth_loss = self.model(chars, feats)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, c_span_mask=c_span_mask, i_span_mask=i_span_mask)
            print('loss:', loss, orth_loss)
            loss += self.args.orth_loss_weight * orth_loss
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

        if not hasattr(self.args, 'intra_word_rel_indexes'):
            setattr(self.args, 'intra_word_rel_indexes', [self.args.nul_index])

        total_loss, metric = 0, CharAttachmentMetric()

        for batch in loader:
            chars, *feats, words, heads, arcs, rels, span_mask = batch
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            # [batch_size, seq_len, seq_len]
            chart_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            # [batch_size, seq_len, seq_len, 2]
            arcs = torch.stack((rels.eq(self.args.nul_index), rels.ne(self.args.nul_index) & rels.ge(0)), dim=-1)
            intra_mask, inter_mask = span_mask[:, 0, ...], span_mask[:, 1, ...]
            # intra/inter complete span
            c_span_mask = torch.stack((intra_mask, inter_mask), dim=-1) if self.args.c_span_mask else None
            # intra/inter incomplete span
            i_span_mask = torch.stack((intra_mask, ~intra_mask & chart_mask), dim=-1) if self.args.i_span_mask else None
            s_arc, s_rel, orth_loss = self.model(chars, feats)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, c_span_mask=c_span_mask, i_span_mask=i_span_mask)
            loss += self.args.orth_loss_weight * orth_loss
            texts = [[unk] + [char for word in seq for char in word] for seq in words]
            if self.args.use_gold_tree:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, target=arcs, c_span_mask=c_span_mask, i_span_mask=i_span_mask)
            elif self.args.use_gold_seg:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, c_span_mask=c_span_mask, i_span_mask=i_span_mask)
            else:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            arc_preds = [seq.tolist() for seq in arc_preds[mask].split(lens)]
            rel_preds = [seq.tolist() for seq in rel_preds[mask].split(lens)]
            # recover char-level predictions to word-level predictions
            preds = [CharCoNLL.recover_words(arc_pred, rel_pred, self.args.intra_word_rel_indexes) for arc_pred, rel_pred in zip(arc_preds, rel_preds)]
            # pred_words = [[f"{''.join(text[b:e])} <-({self.REL.vocab[rel]}) {''.join(text[head[0]:head[1]])}" for (b, e), head, rel in pred] for pred, text in zip(preds, texts)]
            # remove punctuations
            # preds = [[((b, e), head, rel) for (b, e), head, rel in pred if not ispunct(''.join(text[b:e]))] for pred, text in zip(preds, texts)]
            preds = [[((b, e), head, rel, ispunct(''.join(text[b:e]))) for (b, e), head, rel in pred] for pred, text in zip(preds, texts)]

            golds = [LatentCharCoNLL.build_relations(ws, hs, rs) for ws, hs, rs in zip(words, heads, rels.tolist())]
            # gold_words = [[f"{''.join(text[b:e])} <-({self.REL.vocab[rel]}) {''.join(text[head[0]:head[1]])}" for (b, e), head, rel in gold] for gold, text in zip(golds, texts)]
            # remove punctuations
            golds = [[((b, e), head, rel, ispunct(''.join(text[b:e]))) for (b, e), head, rel in gold] for gold, text in zip(golds, texts)]

            total_loss += loss.item()
            metric(preds, golds)
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _test(self, loader):
        self.model.eval()

        if not hasattr(self.args, 'intra_word_rel_indexes'):
            setattr(self.args, 'intra_word_rel_indexes', [self.args.nul_index])

        results = {'id': [], 'word': [], 'head': [], 'dep': [], 'intra_word_struct': [], 'intra_word_labels': []}

        for batch in loader:
            chars, *feats, words, heads, arcs, rels, span_mask = batch
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            # [batch_size, seq_len, seq_len]
            chart_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            # [batch_size, seq_len, seq_len, 2]
            arcs = torch.stack((rels.eq(self.args.nul_index), rels.ne(self.args.nul_index) & rels.ge(0)), dim=-1)
            intra_mask, inter_mask = span_mask[:, 0, ...], span_mask[:, 1, ...]
            # intra/inter complete span
            c_span_mask = torch.stack((intra_mask, inter_mask), dim=-1) if self.args.c_span_mask else None
            # intra/inter incomplete span
            i_span_mask = torch.stack((intra_mask, ~intra_mask & chart_mask), dim=-1) if self.args.i_span_mask else None
            s_arc, s_rel, _ = self.model(chars, feats)

            if self.args.use_gold_tree:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, target=arcs, c_span_mask=c_span_mask, i_span_mask=i_span_mask)
            elif self.args.use_gold_seg:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask, c_span_mask=c_span_mask, i_span_mask=i_span_mask)
            else:
                arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            arc_preds = [seq.tolist() for seq in arc_preds[mask].split(lens)]
            rel_preds = [seq.tolist() for seq in rel_preds[mask].split(lens)]
            # [CharCoNLL.check_root_as_head(arc_pred, rel_pred, self.args.intra_word_rel_indexes) for arc_pred, rel_pred in zip(arc_preds, rel_preds)]

            # recover char-level predictions to word-level predictions
            preds = [CharCoNLL.recover_words(arc_pred, rel_pred, self.args.intra_word_rel_indexes) for arc_pred, rel_pred in zip(arc_preds, rel_preds)]
            texts = [[bos] + [char for word in seq for char in word] for seq in words]

            for text, arc_pred, rel_pred, pred in zip(texts, arc_preds, rel_preds, preds):
                word_boundaries, head_boundaries, dependency_relations = list(zip(*pred))
                # add $root
                word_boundaries = [(0, 1)] + list(word_boundaries)
                results['id'].append([f"{i+1}" for i in range(len(word_boundaries)-1)])
                # get all word
                results['word'].append([''.join(text[begin:end]) for begin, end in word_boundaries[1:]])
                # get all head
                results['head'].append([word_boundaries.index(boundary) for boundary in head_boundaries])
                # get all dep relation
                results['dep'].append([self.REL.vocab[rel] for rel in dependency_relations])
                # get all intra word structure
                results['intra_word_struct'].append(['_'.join([str(arc_pred[i]-begin+1) if arc_pred[i] >= begin and arc_pred[i] < end else '0' for i in range(begin-1, end-1)]) for begin, end in word_boundaries[1:]])
                # get all intra word labels
                label_seq = [[self.REL.vocab[rel_pred[i]] if arc_pred[i] >= begin and arc_pred[i] < end else ROOT for i in range(begin-1, end-1)] for begin, end in word_boundaries[1:]]
                results['intra_word_labels'].append(['_'.join(label[2:] if label.startswith('c-') else label for label in labels) for labels in label_seq])
        return results

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        if not hasattr(self.args, 'intra_word_rel_indexes'):
            setattr(self.args, 'intra_word_rel_indexes', [self.args.nul_index])

        results = {'id': [], 'word': [], 'head': [], 'dep': [], 'intra_word_struct': [], 'intra_word_labels': []}

        for batch in loader:
            chars, *feats, words = batch
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc, s_rel, _ = self.model(chars, feats)

            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            arc_preds = [seq.tolist() for seq in arc_preds[mask].split(lens)]
            rel_preds = [seq.tolist() for seq in rel_preds[mask].split(lens)]
            # recover char-level predictions to word-level predictions
            preds = [CharCoNLL.recover_words(arc_pred, rel_pred, self.args.intra_word_rel_indexes) for arc_pred, rel_pred in zip(arc_preds, rel_preds)]
            texts = [[bos] + [char for word in seq for char in word] for seq in words]

            for text, arc_pred, rel_pred, pred in zip(texts, arc_preds, rel_preds, preds):
                word_boundaries, head_boundaries, dependency_relations = list(zip(*pred))
                # add $root
                word_boundaries = [(0, 1)] + list(word_boundaries)
                results['id'].append([f"{i+1}" for i in range(len(word_boundaries)-1)])
                # get all word
                results['word'].append([''.join(text[begin:end]) for begin, end in word_boundaries[1:]])
                # get all head
                results['head'].append([word_boundaries.index(boundary) for boundary in head_boundaries])
                # get all dep relation
                results['dep'].append([self.REL.vocab[rel] for rel in dependency_relations])
                # get all intra word structure
                results['intra_word_struct'].append(['_'.join([str(arc_pred[i]-begin+1) if arc_pred[i] >= begin and arc_pred[i] < end else '0' for i in range(begin-1, end-1)]) for begin, end in word_boundaries[1:]])
                # get all intra word labels
                label_seq = [[self.REL.vocab[rel_pred[i]] if arc_pred[i] >= begin and arc_pred[i] < end else ROOT for i in range(begin-1, end-1)] for begin, end in word_boundaries[1:]]
                results['intra_word_labels'].append(['_'.join(label[2:] if label.startswith('c-') else label for label in labels) for labels in label_seq])

        return results


class MixedCoarse2FineCharCRFDependencyParser(Coarse2FineCharCRFDependencyParser):

    NAME = 'mixed-c2f-char-crf-dependency'
    MODEL = Coarse2FineCharCRFDependencyModel
    TRANSFORM = MixedCoarse2FineCharCoNLL

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
            parser.model.load_pretrained(parser.args.embed_dict).to(args.device)
            return parser

        logger.info("Building the fields")
        BICHAR, TRICHAR, BERT = None, None, None
        if args.encoder != 'lstm':
            from transformers import AutoTokenizer
            t = AutoTokenizer.from_pretrained(args.bert)
            CHAR = NGramField('chars',
                              pad=t.pad_token,
                              unk=t.unk_token,
                              bos=t.bos_token or t.cls_token)
            CHAR.vocab = t.get_vocab()
        else:
            CHAR = NGramField('chars', pad=pad, unk=unk, bos=bos, lower=True, ngram=1)
            if 'bichar' in args.feat:
                BICHAR = NGramField('bichars', pad=pad, unk=unk, bos=bos, lower=True, ngram=2)
            if 'trichar' in args.feat:
                TRICHAR = NGramField('trichars', pad=pad, unk=unk, bos=bos, lower=True, ngram=3)
            if 'bert' in args.feat:
                from transformers import AutoTokenizer
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = NGramField('bert',
                                  pad=t.pad_token,
                                  unk=t.unk_token,
                                  bos=t.bos_token or t.cls_token)
                BERT.vocab = t.get_vocab()

        WORD = RawField('words')
        # TEXT = RawField('texts')
        HEAD = RawField('heads')
        ARC = BoolChartField('arcs', use_vocab=False, fn=MixedCoarse2FineCharCoNLL.get_arcs)
        REL = ChartField('rels', unk=nul, fn=MixedCoarse2FineCharCoNLL.get_rels)
        SPAN_MASK = BoolChartField('span_mask', use_vocab=False, fn=MixedCoarse2FineCharCoNLL.get_span_constraint)
        transform = MixedCoarse2FineCharCoNLL(FORM=(CHAR, BICHAR, TRICHAR, BERT), LEMMA=WORD, HEAD=HEAD, PHEAD=(ARC, REL, SPAN_MASK))

        train = Dataset(transform, args.train)
        if args.encoder == 'lstm':
            dev = Dataset(transform, args.dev)
            test = Dataset(transform, args.test)
            datasets = [train, dev, test]
            char_embed = CHAR.tailor_embed(datasets, args.char_embed) if args.char_embed else None
            CHAR.build(train, args.min_freq, char_embed, norm=lambda x: x / torch.std(x))
            if BICHAR is not None:
                bichar_embed = BICHAR.tailor_embed(datasets, args.bichar_embed) if args.bichar_embed else None
                BICHAR.build(train, min_freq=3, embed=bichar_embed, norm=lambda x: x / torch.std(x))
            if TRICHAR is not None:
                # FIXME: trichar 数量和 Yan的对不上
                trichar_embed = TRICHAR.tailor_embed(datasets, args.trichar_embed) if args.trichar_embed else None
                TRICHAR.build(train, min_freq=5, embed=trichar_embed, norm=lambda x: x / torch.std(x))
        REL.build(train)

        args.update({
            'n_chars': len(CHAR.vocab) if args.encoder != 'lstm' else CHAR.vocab.n_init,
            'n_rels': len(REL.vocab),
            'n_bichars': len(BICHAR.vocab) if BICHAR is not None else None,
            'n_trichars': len(TRICHAR.vocab) if TRICHAR is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': CHAR.pad_index,
            'unk_index': CHAR.unk_index,
            'bos_index': CHAR.bos_index,
            'nul_index': REL.vocab[nul],
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained({'pretrained_char_embed': CHAR.embed if hasattr(CHAR, 'embed') else None,
                                                   'pretrained_bichar_embed': BICHAR.embed if hasattr(BICHAR, 'embed') else None,
                                                   'pretrained_trichar_embed': TRICHAR.embed if hasattr(TRICHAR, 'embed') else None}).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)


class MixedSplitCoarse2FineCharCRFDependencyParser(SplitCoarse2FineCharCRFDependencyParser):

    NAME = 'mixed-split-c2f-char-crf-dependency'
    MODEL = SplitCoarse2FineCharCRFDependencyModel
    TRANSFORM = MixedCoarse2FineCharCoNLL

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
            parser.model.load_pretrained(parser.args.embed_dict).to(args.device)
            return parser

        logger.info("Building the fields")
        BICHAR, TRICHAR, BERT = None, None, None
        if args.encoder != 'lstm':
            from transformers import AutoTokenizer
            t = AutoTokenizer.from_pretrained(args.bert)
            CHAR = NGramField('chars',
                              pad=t.pad_token,
                              unk=t.unk_token,
                              bos=t.bos_token or t.cls_token)
            CHAR.vocab = t.get_vocab()
        else:
            CHAR = NGramField('chars', pad=pad, unk=unk, bos=bos, lower=True, ngram=1)
            if 'bichar' in args.feat:
                BICHAR = NGramField('bichars', pad=pad, unk=unk, bos=bos, lower=True, ngram=2)
            if 'trichar' in args.feat:
                TRICHAR = NGramField('trichars', pad=pad, unk=unk, bos=bos, lower=True, ngram=3)
            if 'bert' in args.feat:
                from transformers import AutoTokenizer
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = NGramField('bert',
                                  pad=t.pad_token,
                                  unk=t.unk_token,
                                  bos=t.bos_token or t.cls_token)
                BERT.vocab = t.get_vocab()

        WORD = RawField('words')
        # TEXT = RawField('texts')
        HEAD = RawField('heads')
        ARC = BoolChartField('arcs', use_vocab=False, fn=MixedCoarse2FineCharCoNLL.get_arcs)
        REL = ChartField('rels', unk=nul, fn=MixedCoarse2FineCharCoNLL.get_rels)
        SPAN_MASK = BoolChartField('span_mask', use_vocab=False, fn=MixedCoarse2FineCharCoNLL.get_span_constraint)
        transform = MixedCoarse2FineCharCoNLL(FORM=(CHAR, BICHAR, TRICHAR, BERT), LEMMA=WORD, HEAD=HEAD, PHEAD=(ARC, REL, SPAN_MASK))

        train = Dataset(transform, args.train)
        if args.encoder == 'lstm':
            dev = Dataset(transform, args.dev)
            test = Dataset(transform, args.test)
            datasets = [train, dev, test]
            char_embed = CHAR.tailor_embed(datasets, args.char_embed) if args.char_embed else None
            CHAR.build(train, args.min_freq, char_embed, norm=lambda x: x / torch.std(x))
            if BICHAR is not None:
                bichar_embed = BICHAR.tailor_embed(datasets, args.bichar_embed) if args.bichar_embed else None
                BICHAR.build(train, min_freq=3, embed=bichar_embed, norm=lambda x: x / torch.std(x))
            if TRICHAR is not None:
                # FIXME: trichar 数量和 Yan的对不上
                trichar_embed = TRICHAR.tailor_embed(datasets, args.trichar_embed) if args.trichar_embed else None
                TRICHAR.build(train, min_freq=5, embed=trichar_embed, norm=lambda x: x / torch.std(x))
        REL.build(train)

        args.update({
            'n_chars': len(CHAR.vocab) if args.encoder != 'lstm' else CHAR.vocab.n_init,
            'n_rels': len(REL.vocab),
            'n_bichars': len(BICHAR.vocab) if BICHAR is not None else None,
            'n_trichars': len(TRICHAR.vocab) if TRICHAR is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': CHAR.pad_index,
            'unk_index': CHAR.unk_index,
            'bos_index': CHAR.bos_index,
            'nul_index': REL.vocab[nul],
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained({'pretrained_char_embed': CHAR.embed if hasattr(CHAR, 'embed') else None,
                                                   'pretrained_bichar_embed': BICHAR.embed if hasattr(BICHAR, 'embed') else None,
                                                   'pretrained_trichar_embed': TRICHAR.embed if hasattr(TRICHAR, 'embed') else None}).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)

