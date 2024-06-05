# -*- coding: utf-8 -*-

from collections import Counter
from datetime import datetime


class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return 0.


class DateMetric(Metric):

    def __call__(self):
        self.time = datetime.now().timestamp()

    @property
    def score(self):
        return self.time


class LossMetric(Metric):

    def __init__(self, eps=1e-32):
        super().__init__()

        self.eps = eps
        self.loss = 1e32

    def __call__(self, loss):
        self.loss = loss if isinstance(loss, float) else loss.item()

    def __repr__(self):
        return f"Loss: {self.loss:.4f}"

    @property
    def score(self):
        return 1 / (self.loss + self.eps)


class SegF1Metric(Metric):

    def __init__(self, eps=1e-8):
        super(SegF1Metric, self).__init__()

        self.tp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        """[summary]

        Args:
            preds (): List[List[tuple(i, j)]]
            golds ([type]): [description]
        """
        for pred, gold in zip(preds, golds):
            tp = list((Counter(pred) & Counter(gold)).elements())
            self.tp += len(tp)
            self.pred += len(pred)
            self.gold += len(gold)

    def __repr__(self):
        return f"P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"

    @property
    def score(self):
        return self.f

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)


class AttachmentMetric(Metric):

    def __init__(self, eps=1e-12):
        super().__init__()

        self.eps = eps

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def __repr__(self):
        s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"UAS: {self.uas:6.2%} LAS: {self.las:6.2%}"
        return s

    def __call__(self, arc_preds, rel_preds, arc_golds, rel_golds, mask):
        lens = mask.sum(1)
        arc_mask = arc_preds.eq(arc_golds) & mask
        rel_mask = rel_preds.eq(rel_golds) & arc_mask
        arc_mask_seq, rel_mask_seq = arc_mask[mask], rel_mask[mask]

        self.n += len(mask)
        self.n_ucm += arc_mask.sum(1).eq(lens).sum().item()
        self.n_lcm += rel_mask.sum(1).eq(lens).sum().item()

        self.total += len(arc_mask_seq)
        self.correct_arcs += arc_mask_seq.sum().item()
        self.correct_rels += rel_mask_seq.sum().item()
        return self

    @property
    def score(self):
        return self.las

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.correct_rels / (self.total + self.eps)


class UnlabeledAttachmentMetric(AttachmentMetric):

    def __init__(self, eps=1e-12):
        super().__init__(eps)
        self.n_correct_root = 0.0

    def __repr__(self):
        s = f"UCM: {self.ucm:6.2%} "
        s += f"UAS: {self.uas:6.2%} "
        s += f"Root: {self.root_acc:6.2%}"
        return s

    def __call__(self, preds, golds):
        pred_heads = [head for word, head in preds]
        gold_heads = [head for word, head in golds]
        pred_root = (idx for idx, arc_pred in enumerate(pred_heads, 1) if arc_pred == 0)
        gold_root = (idx for idx, arc_gold in enumerate(gold_heads, 1) if arc_gold == 0)
        mask = [1 if arc_pred == arc_gold else 0 for arc_pred, arc_gold in zip(pred_heads, gold_heads)]

        self.n += 1
        self.n_ucm += 1 if all(mask) else 0
        # if all(mask):
        #     print(preds, golds)
        self.n_correct_root += 1 if next(pred_root, None) == next(gold_root, None) else 0

        self.total += len(mask)
        self.correct_arcs += sum(mask)
        return self

    @property
    def root_acc(self):
        return self.n_correct_root / (self.n + self.eps)

    @property
    def score(self):
        return self.uas


class CharAttachmentMetric(AttachmentMetric):

    def __init__(self, eps=1e-12):
        super().__init__()

        self.pred = 0.0
        self.gold = 0.0

        self.correct_segs = 0.0
        self.pred_seg = 0.0
        self.gold_seg = 0.0

    def __repr__(self):
        s = f"\nSegP: {self.seg_p:6.2%} SegR: {self.seg_r:6.2%} SegF: {self.seg_f:6.2%}"
        s += '\n' + super().__repr__()
        s += f"\nUP: {self.unlabel_p:6.2%} UR: {self.unlabel_r:6.2%} UF: {self.unlabel_f:6.2%}"
        s += f"\nLP: {self.label_p:6.2%} LR: {self.label_r:6.2%} LF: {self.label_f:6.2%} "
        return s

    def __call__(self, preds, golds):
        assert len(preds) == len(golds), f"len(preds)={len(preds)} len(golds)={len(golds)}"
        for pred, gold in zip(preds, golds):
            seg_pred = Counter([word for word, head, rel, punct in pred])
            seg_gold = Counter([word for word, head, rel, punct in gold])
            correct_segs = list((seg_pred & seg_gold).elements())
            pred = [(word, head, rel) for word, head, rel, punct in pred if not punct]
            gold = [(word, head, rel) for word, head, rel, punct in gold if not punct]
            upred = Counter([(word, head) for word, head, rel in pred])
            ugold = Counter([(word, head) for word, head, rel in gold])
            # upred = Counter([(word, head[-1]) for word, head, rel in pred])
            # ugold = Counter([(word, head[-1]) for word, head, rel in gold])
            correct_arcs = list((upred & ugold).elements())
            lpred = Counter([(word, head, rel) for word, head, rel in pred])
            lgold = Counter([(word, head, rel) for word, head, rel in gold])
            # lpred = Counter([(word, head[-1], rel) for word, head, rel in pred])
            # lgold = Counter([(word, head[-1], rel) for word, head, rel in gold])
            correct_rels = list((lpred & lgold).elements())

            self.n += 1
            self.n_ucm += len(correct_arcs) == len(pred) == len(gold)
            self.n_lcm += len(correct_rels) == len(pred) == len(gold)
            self.correct_arcs += len(correct_arcs)
            self.correct_rels += len(correct_rels)
            self.total += len(gold)
            self.pred += len(pred)
            self.gold += len(gold)
            self.correct_segs += len(correct_segs)
            self.pred_seg += len(seg_pred)
            self.gold_seg += len(seg_gold)

    @property
    def score(self):
        return self.label_f

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.correct_rels / (self.total + self.eps)

    @property
    def unlabel_p(self):
        return self.correct_arcs / (self.pred + self.eps)

    @property
    def unlabel_r(self):
        return self.correct_arcs / (self.gold + self.eps)

    @property
    def unlabel_f(self):
        return 2 * self.correct_arcs / (self.pred + self.gold + self.eps)

    @property
    def label_p(self):
        return self.correct_rels / (self.pred + self.eps)

    @property
    def label_r(self):
        return self.correct_rels / (self.gold + self.eps)

    @property
    def label_f(self):
        return 2 * self.correct_rels / (self.pred + self.gold + self.eps)

    @property
    def seg_p(self):
        return self.correct_segs / (self.pred_seg + self.eps)

    @property
    def seg_r(self):
        return self.correct_segs / (self.gold_seg + self.eps)

    @property
    def seg_f(self):
        return 2 * self.correct_segs / (self.pred_seg + self.gold_seg + self.eps)


class DetailedCharAttachmentMetric(CharAttachmentMetric):

    def __init__(self, eps=1e-12):
        super().__init__()

        self.seg_wrong = 0.0
        self.head_wrong = 0.0
        self.all_right = 0.0

    def __call__(self, preds, golds):
        super().__call__(preds, golds)
        for pred, gold in zip(preds, golds):
            seg_gold = [word for word, head, rel, punct in gold] + [(0, 1)]
            pred = [(word, head) for word, head, rel, punct in pred if not punct]
            gold = [(word, head) for word, head, rel, punct in gold if not punct]
            for word, head in pred:
                if word not in seg_gold:
                    self.seg_wrong += 1
                elif head not in seg_gold:
                    self.seg_wrong += 1
                elif (word, head) not in gold:
                    self.head_wrong += 1
                elif (word, head) in gold:
                    self.all_right += 1
                else:
                    print(word, head)
                    raise ValueError('something wrong')

    def __repr__(self):
        s = super().__repr__()
        s += f"\nPred Size: {self.pred} Gold Size: {self.gold}"
        s += f"\nSeg Wrong: {self.seg_wrong_p:6.2%} Head Wrong: {self.head_wrong_p:6.2%} All Right: {self.all_right_p:6.2%}"
        return s

    @property
    def seg_wrong_p(self):
        return self.seg_wrong / (self.pred + self.eps)

    @property
    def head_wrong_p(self):
        return self.head_wrong / (self.pred + self.eps)

    @property
    def all_right_p(self):
        return self.all_right / (self.pred + self.eps)


class MetricByLength(object):

    def __init__(self, metric) -> None:
        self.lens = [(1, 2), (2, 3), (3, 4), (4, 1e3)]
        self.metrics = [metric() for _ in self.lens]

    def __call__(self, preds, golds):
        for (l, r), metric in zip(self.lens, self.metrics):
            sub_preds = [[p for p in pred if l <= p[0][1] - p[0][0] < r] for pred in preds]
            sub_golds = [[g for g in gold if l <= g[0][1] - g[0][0] < r] for gold in golds]
            metric(sub_preds, sub_golds)

    def __repr__(self) -> str:
        s = ''
        for (l, r), metric in zip(self.lens, self.metrics):
            s += f"\n\nword length: [{l} - {r}) !"
            s += repr(metric)

        return s


class WordStructMetric(object):

    def __init__(self) -> None:
        self.word_struct = Counter()
        self.leftward = 0
        self.rightward = 0

    def __call__(self, preds, chars):
        words = set()
        for pred, sent in zip(preds, chars):
            for (begin, end), heads in pred:
                word = ''.join(sent[begin:end])
                # check leftward or rightward
                for modifier, head in zip(range(begin, end), heads):
                    if modifier < head:
                        self.leftward += 1
                    elif modifier > head:
                        self.rightward += 1
                    else:
                        print(begin, end, heads)
                        print(modifier, head)
                        raise ValueError('head should not be equal to modifier')
                # count word structure
                length = end - begin
                relative_heads = tuple(head - begin + 1 if head >= begin and head < end else 0 for head in heads)
                if len([i for i in heads if i == 0]) > 1:
                    continue
                if word in words:
                    continue
                else:
                    words.add(word)
                self.word_struct[(length, relative_heads)] += 1

    def __repr__(self) -> str:
        s = f"{self.leftward} leftward, {self.rightward} rightward"
        s += '\n'
        for (length, heads), size in self.word_struct.most_common():
            s += f"{length} {'_'.join(str(head) for head in heads)} {size}\n"

        return s


class SpanMetric(Metric):

    def __init__(self, eps=1e-12):
        super().__init__()

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.utp = 0.0
        self.ltp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):
            upred = Counter([(i, j) for i, j, label in pred])
            ugold = Counter([(i, j) for i, j, label in gold])
            utp = list((upred & ugold).elements())
            lpred = Counter(pred)
            lgold = Counter(gold)
            ltp = list((lpred & lgold).elements())
            self.n += 1
            self.n_ucm += len(utp) == len(pred) == len(gold)
            self.n_lcm += len(ltp) == len(pred) == len(gold)
            self.utp += len(utp)
            self.ltp += len(ltp)
            self.pred += len(pred)
            self.gold += len(gold)
        return self

    def __repr__(self):
        s = f"\nUCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"\nUP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} "
        s += f"\nLP: {self.lp:6.2%} LR: {self.lr:6.2%} LF: {self.lf:6.2%}"

        return s

    @property
    def score(self):
        return self.lf

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def lp(self):
        return self.ltp / (self.pred + self.eps)

    @property
    def lr(self):
        return self.ltp / (self.gold + self.eps)

    @property
    def lf(self):
        return 2 * self.ltp / (self.pred + self.gold + self.eps)


class SpanSRLMetric(SpanMetric):

    def __init__(self, eps=1e-12):
        super().__init__()

        self.official_lp = 0.0
        self.official_lr = 0.0
        self.official_lf = 0.0

        self.total_correct_predicate = 0.0
        self.total_pred_predicate = 0.0
        self.total_gold_predicate = 0.0

        self.total_correct_argument = 0.0
        self.total_pred_argument = 0.0
        self.total_gold_argument = 0.0

        self.total_full_covered_prop = 0.0
        self.total_not_covered_prop = 0.0
        self.total_partial_covered_prop = 0.0

        # divide propositions by width (word size)
        self.total_correct_with_1 = 0.0
        self.total_pred_width_1 = 0.0
        self.total_gold_width_1 = 0.0

        self.total_correct_with_2 = 0.0
        self.total_pred_width_2 = 0.0
        self.total_gold_width_2 = 0.0

        # 3 ~ 6
        self.total_correct_with_3 = 0.0
        self.total_pred_width_3 = 0.0
        self.total_gold_width_3 = 0.0

        # > 6
        self.total_correct_with_6 = 0.0
        self.total_pred_width_6 = 0.0
        self.total_gold_width_6 = 0.0

    def __call__(self, preds, golds):
        assert len(preds) == len(golds)
        for pred, gold in zip(preds, golds):
            upred = Counter([(v, b, e) for v, b, e, label in pred])
            ugold = Counter([(v, b, e) for v, b, e, label in gold])
            utp = list((upred & ugold).elements())
            lpred = Counter(pred)
            lgold = Counter(gold)
            ltp = list((lpred & lgold).elements())
            self.n += 1
            self.n_ucm += len(utp) == len(pred) == len(gold)
            self.n_lcm += len(ltp) == len(pred) == len(gold)
            self.utp += len(utp)
            self.ltp += len(ltp)
            self.pred += len(pred)
            self.gold += len(gold)

            # predicate accuracy
            pred_predicate = Counter([v for v, b, e, label in pred])
            gold_predicate = Counter([v for v, b, e, label in gold])
            self.total_correct_predicate += len(pred_predicate & gold_predicate)
            self.total_pred_predicate += len(pred_predicate)
            self.total_gold_predicate += len(gold_predicate)

            # argument
            pred_argument = Counter([(b, e) for v, b, e, label in pred if not b == e]) 
            gold_argument = Counter([(b, e) for v, b, e, label in gold if not b == e])
            self.total_correct_argument += len(pred_argument & gold_argument)
            self.total_pred_argument += len(pred_argument)
            self.total_gold_argument += len(gold_argument)

            # argument cover rate
            for pred_prop in upred:
                flag = False
                for gold_prop in ugold:
                    if pred_prop[0] == gold_prop[0]:
                        if pred_prop[1] == gold_prop[1] and pred_prop[2] == gold_prop[2]:
                            self.total_full_covered_prop += 1
                            flag = True
                        elif pred_prop[2] >= gold_prop[1] and pred_prop[1] <= gold_prop[2]:
                            self.total_partial_covered_prop += 1
                            flag = True
                    if flag:
                        break
                if not flag:
                    self.total_not_covered_prop += 1

            # width=1
            pred_width_1 = Counter([(v, b, e) for v, b, e, label in pred if e - b == 0])
            gold_width_1 = Counter([(v, b, e) for v, b, e, label in gold if e - b == 0])
            self.total_correct_with_1 += len(pred_width_1 & gold_width_1)
            self.total_pred_width_1 += len(pred_width_1)
            self.total_gold_width_1 += len(gold_width_1)
            # width=2
            pred_width_2 = Counter([(v, b, e, label) for v, b, e, label in pred if e - b == 1])
            gold_width_2 = Counter([(v, b, e, label) for v, b, e, label in gold if e - b == 1])
            self.total_correct_with_2 += len(pred_width_2 & gold_width_2)
            self.total_pred_width_2 += len(pred_width_2)
            self.total_gold_width_2 += len(gold_width_2)
            # 3 < width < 6
            pred_width_3 = Counter([(v, b, e, label) for v, b, e, label in pred if e - b >= 2 and e - b <= 5])
            gold_width_3 = Counter([(v, b, e, label) for v, b, e, label in gold if e - b >= 2 and e - b <= 5])
            self.total_correct_with_3 += len(pred_width_3 & gold_width_3)
            self.total_pred_width_3 += len(pred_width_3)
            self.total_gold_width_3 += len(gold_width_3)
            # width > 6
            pred_width_6 = Counter([(v, b, e, label) for v, b, e, label in pred if e - b > 5])
            gold_width_6 = Counter([(v, b, e, label) for v, b, e, label in gold if e - b > 5])
            self.total_correct_with_6 += len(pred_width_6 & gold_width_6)
            self.total_pred_width_6 += len(pred_width_6)
            self.total_gold_width_6 += len(gold_width_6)

        return self

    def __repr__(self):
        s = super().__repr__()
        s += "\n== Official results =="
        s += f"\nLP: {self.official_lp:6.2%} LR: {self.official_lr:6.2%} LF: {self.official_lf:6.2%}"
        s += f"\n{self.official_lp*100:6.2f} {self.official_lr*100:6.2f} {self.official_lf*100:6.2f}"
        s += f"\n== Predicate == P: {self.prd_p:6.2%} R: {self.prd_r:6.2%} F: {self.prd_f:6.2%}"
        s += f"\n== Argument == P: {self.arg_p:6.2%} R: {self.arg_r:6.2%} F: {self.arg_f:6.2%} Correct: {self.total_correct_argument} Pred: {self.total_pred_argument} Gold: {self.total_gold_argument}"
        s += f"\n== Proposition Cover Rate == \
        \n partial covered: {self.total_partial_covered_prop} {self.gold} {self.total_partial_covered_prop/(self.gold+self.eps):6.2%} \
        \n full covered {self.total_full_covered_prop} {self.gold} {self.total_full_covered_prop/(self.gold+self.eps):6.2%} \
        \n not covered {self.total_not_covered_prop} {self.gold} {self.total_not_covered_prop/(self.gold+self.eps):6.2%}"
        s += f"\n== Gold Size Proportion == {self.total_gold_width_1/self.gold:6.2%} {self.total_gold_width_2/self.gold:6.2%} {self.total_gold_width_3/self.gold:6.2%} {self.total_gold_width_6/self.gold:6.2%}"
        if not self.pred:
            self.pred = self.eps
        s += f"\n== Pred Size Proportion == {self.total_pred_width_1/self.pred:6.2%} {self.total_pred_width_2/self.pred:6.2%} {self.total_pred_width_3/self.pred:6.2%} {self.total_pred_width_6/self.pred:6.2%}"
        s += f"\n== Gold Size == {self.total_gold_width_1} {self.total_gold_width_2} {self.total_gold_width_3} {self.total_gold_width_6}"
        s += f"\n== Pred Size == {self.total_pred_width_1} {self.total_pred_width_2} {self.total_pred_width_3} {self.total_pred_width_6}"
        s += f"\n== Corr Size == {self.total_correct_with_1} {self.total_correct_with_2} {self.total_correct_with_3} {self.total_correct_with_6}"
        s += f"\n== Width = 1 == P: {self.width_1_p:6.2%} R: {self.width_1_r:6.2%} F: {self.width_1_f:6.2%} \
               \n== Width = 2 == P: {self.width_2_p:6.2%} R: {self.width_2_r:6.2%} F: {self.width_2_f:6.2%} \
               \n== Width < 6 == P: {self.width_3_p:6.2%} R: {self.width_3_r:6.2%} F: {self.width_3_f:6.2%} \
               \n== Width > 6 == P: {self.width_6_p:6.2%} R: {self.width_6_r:6.2%} F: {self.width_6_f:6.2%}"

        return s

    def set_official(self, p, r, f):
        self.official_lp = p
        self.official_lr = r
        self.official_lf = f

    @property
    def score(self):
        return self.official_lf

    @property
    def prd_p(self):
        return self.total_correct_predicate / (self.total_pred_predicate + self.eps)

    @property
    def prd_r(self):
        return self.total_correct_predicate / (self.total_gold_predicate + self.eps)

    @property
    def prd_f(self):
        # return 2 * self.prd_p * self.prd_r / (self.prd_p + self.prd_r + self.eps)
        return 2 * self.total_correct_predicate / (self.total_pred_predicate + self.total_gold_predicate + self.eps)

    @property
    def arg_p(self):
        return self.total_correct_argument / (self.total_pred_argument + self.eps)

    @property
    def arg_r(self):
        return self.total_correct_argument / (self.total_gold_argument + self.eps)

    @property
    def arg_f(self):
        # return 2 * self.arg_p * self.arg_r / (self.arg_p + self.arg_r + self.eps)
        return 2 * self.total_correct_argument / (self.total_pred_argument + self.total_gold_argument + self.eps)

    @property
    def width_1_p(self):
        return self.total_correct_with_1 / (self.total_pred_width_1 + self.eps)

    @property
    def width_1_r(self):
        return self.total_correct_with_1 / (self.total_gold_width_1 + self.eps)

    @property
    def width_1_f(self):
        return 2 * self.total_correct_with_1 / (self.total_pred_width_1 + self.total_gold_width_1 + self.eps)

    @property
    def width_2_p(self):
        return self.total_correct_with_2 / (self.total_pred_width_2 + self.eps)

    @property
    def width_2_r(self):
        return self.total_correct_with_2 / (self.total_gold_width_2 + self.eps)

    @property
    def width_2_f(self):
        return 2 * self.total_correct_with_2 / (self.total_pred_width_2 + self.total_gold_width_2 + self.eps)

    @property
    def width_3_p(self):
        return self.total_correct_with_3 / (self.total_pred_width_3 + self.eps)

    @property
    def width_3_r(self):
        return self.total_correct_with_3 / (self.total_gold_width_3 + self.eps)

    @property
    def width_3_f(self):
        return 2 * self.total_correct_with_3 / (self.total_pred_width_3 + self.total_gold_width_3 + self.eps)

    @property
    def width_6_p(self):
        return self.total_correct_with_6 / (self.total_pred_width_6 + self.eps)

    @property
    def width_6_r(self):
        return self.total_correct_with_6 / (self.total_gold_width_6 + self.eps)

    @property
    def width_6_f(self):
        return 2 * self.total_correct_with_6 / (self.total_pred_width_6 + self.total_gold_width_6 + self.eps)


class WordBasedJointConstAndSpanSRLMetric(SpanSRLMetric):

    def __init__(self, eps=1e-12):
        super().__init__(eps)

        self.const_utp = 0.0
        self.const_ltp = 0.0
        self.const_pred = 0.0
        self.const_gold = 0.0

        self.other_const_lf = None

        self.total_pred_constituent = 0.0
        self.total_gold_constituent = 0.0
        # pred srl and pred const
        self.pred_pred_consistent = 0.0
        self.gold_gold_consistent = 0.0
        # pred srl and gold const
        self.pred_gold_consistent = 0.0
        # gold srl and pred const
        self.gold_pred_consistent = 0.0

    def __call__(self, preds, golds, const_preds, const_golds):
        super().__call__(preds, golds)

        for pred, gold in zip(const_preds, const_golds):
            upred = Counter([(i, j) for i, j, label in pred])
            ugold = Counter([(i, j) for i, j, label in gold])
            utp = list((upred & ugold).elements())
            lpred = Counter(pred)
            lgold = Counter(gold)
            ltp = list((lpred & lgold).elements())
            self.n += 1
            self.n_ucm += len(utp) == len(pred) == len(gold)
            self.n_lcm += len(ltp) == len(pred) == len(gold)
            self.const_utp += len(utp)
            self.const_ltp += len(ltp)
            self.const_pred += len(pred)
            self.const_gold += len(gold)

            # const_pred = Counter([(b+1, e) for b, e, label in pred])
            # const_gold = Counter([(b+1, e) for b, e, label in gold])
            # self.total_pred_constituent += len(const_pred)
            # self.total_gold_constituent += len(const_gold)

        # for srl_pred, srl_gold, const_pred, const_gold, length in zip(preds, golds, const_preds, const_golds, lens):
        #     print('-----------------------------')
        #     print('length:', length)
        #     print(f"srl_pred: {srl_pred}")
        #     print(f"srl_gold: {srl_gold}")
        #     print(f"const_pred: {const_pred}")
        #     print(f"const_gold: {const_gold}")
        #     srl_pred = Counter([(b, e) for v, b, e, label in srl_pred if not b == e])
        #     srl_gold = Counter([(b, e) for v, b, e, label in srl_gold if not b == e])
        #     const_pred = Counter([(b+1, e) for b, e, label in const_pred])
        #     const_gold = Counter([(b+1, e) for b, e, label in const_gold])
        #     print(f"srl_pred: {srl_pred}")
        #     print(f"srl_gold: {srl_gold}")
        #     print(f"const_pred: {const_pred}")
        #     print(f"const_gold: {const_gold}")
        #     self.total_pred_constituent += len(const_pred)
        #     self.total_gold_constituent += len(const_gold)
        #     self.pred_pred_consistent += len(srl_pred & const_pred)
        #     self.gold_gold_consistent += len(srl_gold & const_gold)
        #     self.pred_gold_consistent += len(srl_pred & const_gold)
        #     self.gold_pred_consistent += len(srl_gold & const_pred)

        return self

    def __repr__(self):
        s = f"\n== Const results ==\nUCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"\nUP: {self.const_up:6.2%} UR: {self.const_ur:6.2%} UF: {self.const_uf:6.2%} "
        s += f"\nLP: {self.const_lp:6.2%} LR: {self.const_lr:6.2%} LF: {self.const_lf:6.2%}"
        s += super().__repr__()
        # s += f"\n == Pred SRL and Pred Const Consistency == {self.pred_pred_consistent} / {self.total_pred_argument} = {self.pred_pred_consistent/self.total_pred_argument} {self.total_pred_constituent} "
        # s += f"\n == Gold SRL and Gold Const Consistency == {self.gold_gold_consistent} / {self.total_gold_argument} = {self.gold_gold_consistent/self.total_gold_argument} {self.total_gold_constituent} "
        # s += f"\n == Pred SRL and Gold Const Consistency == {self.pred_gold_consistent} / {self.total_pred_argument} = {self.pred_gold_consistent/self.total_pred_argument} {self.total_gold_constituent} "
        # s += f"\n == Gold SRL and Pred Const Consistency == {self.gold_pred_consistent} / {self.total_gold_argument} = {self.gold_pred_consistent/self.total_gold_argument} {self.total_pred_constituent} "

        return s

    @property
    def score(self):
        if self.other_const_lf is not None:
            return self.lf + self.other_const_lf
        return self.lf + self.const_lf

    def update(self, other):
        self.other_const_lf = other.const_lf

    @property
    def const_up(self):
        return self.const_utp / (self.const_pred + self.eps)

    @property
    def const_ur(self):
        return self.const_utp / (self.const_gold + self.eps)

    @property
    def const_uf(self):
        return 2 * self.const_utp / (self.const_pred + self.const_gold + self.eps)

    @property
    def const_lp(self):
        return self.const_ltp / (self.const_pred + self.eps)

    @property
    def const_lr(self):
        return self.const_ltp / (self.const_gold + self.eps)

    @property
    def const_lf(self):
        return 2 * self.const_ltp / (self.const_pred + self.const_gold + self.eps)


class ChartMetric(Metric):

    def __init__(self, eps=1e-12):
        super(ChartMetric, self).__init__()

        self.tp = 0.0
        self.utp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        pred_mask = preds.ge(0)
        gold_mask = golds.ge(0)
        span_mask = pred_mask & gold_mask
        self.pred += pred_mask.sum().item()
        self.gold += gold_mask.sum().item()
        self.tp += (preds.eq(golds) & span_mask).sum().item()
        self.utp += span_mask.sum().item()
        return self

    def __repr__(self):
        return f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"

    @property
    def score(self):
        return self.f

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)
