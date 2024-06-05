# -*- coding: utf-8 -*-

import subprocess


def change_BES(source_file, tgt_file, task):
    '''
    for BES
    '''
    sent_idx = 0
    sum_conf1_count = 0
    sum_conf2_count = 0
    if(task == '05'):
        word_idx_to_write = 2
    else:
        word_idx_to_write = 1
    with open(source_file, 'r') as f:
        lines = [line.strip() for line in f]
    sentences = []
    start, i = 0, 0
    for line in lines:
        if not line:
            sentences.append(lines[start:i])
            start = i + 1
        i += 1

    new_sentence_lsts = []
    for sentence in sentences:
        sent_idx += 1
        sentence_lst = []
        for line in sentence:
            sentence_lst.append(line.split('\t'))
        # sentence_lst:[line_lst,...,] line_lst:[num, word, lemma, _, pos, _, _, _, relas, _]

        # firstly find all predicates 
        num_words = len(sentence_lst)
        prd_map = {}  # 33:1, 44:2
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                continue
            relas = line_lst[8].split('|')
            for rela in relas:
                head, rel = rela.split(':')
                if (head == '0'):
                    prd_map[i] = len(prd_map) + 1
                    break

        arc_values = []
        # [[[a0],[a0]],]
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                arc_value = [[] for j in range(len(prd_map))]
                arc_values.append(arc_value)
            else:
                relas = line_lst[8].split('|')
                arc_value = [[] for j in range(len(prd_map))]
                for rela in relas:
                    head, rel = rela.split(':')
                    head_idx = int(head)
                    if (head_idx in prd_map):
                        # 这个步骤保证是srl结构，去掉0，和那些没有被预测为谓词的，边（这样应该好点，因为谓词预测准确率应该蛮高）
                        arc_value[prd_map[head_idx] - 1].append(rel)
                        # 应该只有一个，一个词根一个谓词只能有一个关系
                arc_values.append(arc_value)

        re_prd_map = {}  # 1:33, 2:44
        for key, value in prd_map.items():
            re_prd_map[value] = key

        new_columns = []
        column_1 = []
        for i, line_lst in enumerate(sentence_lst, 1):
            if (i in prd_map):
                column_1.append(line_lst[word_idx_to_write])
            else:
                column_1.append('-')
        new_columns.append(column_1)

        for key, value in re_prd_map.items():
            this_prd_arc = [
                word_arc_lsts[key - 1] for word_arc_lsts in arc_values
            ]
            # [[rel], [rel], [],...]
            this_prd_idx = value  # start from 1
            this_column, con1, con2 = produce_column_BES(this_prd_arc, this_prd_idx)
            sum_conf1_count += con1
            sum_conf2_count += con2
            new_columns.append(this_column)
        
        new_sentence_lst = []
        num_column = len(new_columns)
        for i in range(num_words):
            new_line_lst = []
            for j in range(num_column):
                new_line_lst.append(new_columns[j][i])
            new_sentence_lst.append(new_line_lst)
        new_sentence_lsts.append(new_sentence_lst)
    print('conflict I-:'+str(sum_conf1_count))
    print('conflict label:'+str(sum_conf2_count))
    with open(tgt_file, 'w') as f:
        for new_sentence_lst in new_sentence_lsts:
            for line_lst in new_sentence_lst:
                f.write(' '.join(line_lst) + '\n')
            f.write('\n')

def change_BE(source_file, tgt_file, task):
    '''
    for BE
    '''
    sum_conf1_count = 0
    sum_conf2_count = 0
    if(task == '05'):
        word_idx_to_write = 2
    else:
        word_idx_to_write = 1
    # change simple crosstag conllu to target type
    with open(source_file, 'r') as f:
        lines = [line.strip() for line in f]
    sentences = []
    start, i = 0, 0
    for line in lines:
        if not line:
            sentences.append(lines[start:i])
            start = i + 1
        i += 1

    new_sentence_lsts = []
    for sentence in sentences:
        sentence_lst = []
        for line in sentence:
            sentence_lst.append(line.split('\t'))
        # sentence_lst:[line_lst,...,] line_lst:[num, word, lemma, _, pos, _, _, _, relas, _]

        # firstly find all predicates 
        num_words = len(sentence_lst)
        prd_map = {}  # 33:1, 44:2
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                continue
            relas = line_lst[8].split('|')
            for rela in relas:
                head, rel = rela.split(':')
                if (head == '0'):
                    prd_map[i] = len(prd_map) + 1
                    break

        arc_values = []
        # [[[a0],[a0]],]
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                arc_value = [[] for j in range(len(prd_map))]
                arc_values.append(arc_value)
            else:
                relas = line_lst[8].split('|')
                arc_value = [[] for j in range(len(prd_map))]
                for rela in relas:
                    head, rel = rela.split(':')
                    head_idx = int(head)
                    if (head_idx in prd_map):
                        # 这个步骤保证是srl结构，去掉0，和那些没有被预测为谓词的，边（这样应该好点，因为谓词预测准确率应该蛮高）
                        arc_value[prd_map[head_idx] - 1].append(rel)
                        # 应该只有一个，一个词根一个谓词只能有一个关系
                arc_values.append(arc_value)

        re_prd_map = {}  # 1:33, 2:44
        for key, value in prd_map.items():
            re_prd_map[value] = key

        new_columns = []
        column_1 = []
        for i, line_lst in enumerate(sentence_lst, 1):
            if (i in prd_map):
                column_1.append(line_lst[word_idx_to_write])
            else:
                column_1.append('-')
        new_columns.append(column_1)

        for key, value in re_prd_map.items():
            this_prd_arc = [
                word_arc_lsts[key - 1] for word_arc_lsts in arc_values
            ]
            # [[rel], [rel], [],...]
            this_prd_idx = value  # start from 1
            this_column, con1, con2 = produce_column_BE(this_prd_arc, this_prd_idx)
            sum_conf1_count += con1
            sum_conf2_count += con2
            new_columns.append(this_column)

        new_sentence_lst = []
        num_column = len(new_columns)
        for i in range(num_words):
            new_line_lst = []
            for j in range(num_column):
                new_line_lst.append(new_columns[j][i])
            new_sentence_lst.append(new_line_lst)
        new_sentence_lsts.append(new_sentence_lst)
    print('conflict I-:'+str(sum_conf1_count))
    print('conflict label:'+str(sum_conf2_count))
    with open(tgt_file, 'w') as f:
        for new_sentence_lst in new_sentence_lsts:
            for line_lst in new_sentence_lst:
                f.write(' '.join(line_lst) + '\n')
            f.write('\n')

def change_BII(source_file, tgt_file, task):
    """
    for bii
    """
    sum_false_count = 0
    if(task == '05'):
        word_idx_to_write = 2
    else:
        word_idx_to_write = 1
    with open(source_file, 'r') as f:
        lines = [line.strip() for line in f]
    sentences = []
    start, i = 0, 0
    for line in lines:
        if not line:
            sentences.append(lines[start:i])
            start = i + 1
        i += 1

    new_sentence_lsts = []
    for sentence in sentences:
        sentence_lst = []
        for line in sentence:
            sentence_lst.append(line.split('\t'))
        # sentence_lst:[line_lst,...,] line_lst:[num, word, lemma, _, pos, _, _, _, relas, _]

        # firstly find all predicates 
        num_words = len(sentence_lst)
        prd_map = {}  # 33:1, 44:2
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                continue
            relas = line_lst[8].split('|')
            for rela in relas:
                head, rel = rela.split(':')
                if (head == '0'):
                    prd_map[i] = len(prd_map) + 1
                    break

        arc_values = []
        # [[[a0],[a0]],]
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                arc_value = [[] for j in range(len(prd_map))]
                arc_values.append(arc_value)
            else:
                relas = line_lst[8].split('|')
                arc_value = [[] for j in range(len(prd_map))]
                for rela in relas:
                    head, rel = rela.split(':')
                    head_idx = int(head)
                    if (head_idx in prd_map):
                        # 这个步骤保证是srl结构，去掉0，和那些没有被预测为谓词的，边（这样应该好点，因为谓词预测准确率应该蛮高）
                        arc_value[prd_map[head_idx] - 1].append(rel)
                        # 应该只有一个，一个词根一个谓词只能有一个关系
                arc_values.append(arc_value)

        re_prd_map = {}  # 1:33, 2:44
        for key, value in prd_map.items():
            re_prd_map[value] = key

        new_columns = []
        column_1 = []
        for i, line_lst in enumerate(sentence_lst, 1):
            if (i in prd_map):
                column_1.append(line_lst[word_idx_to_write])
            else:
                column_1.append('-')
        new_columns.append(column_1)

        for key, value in re_prd_map.items():
            this_prd_arc = [
                word_arc_lsts[key - 1] for word_arc_lsts in arc_values
            ]
            # [[rel], [rel], [],...]
            this_prd_idx = value  # start from 1
            this_column, count = produce_column_BII(this_prd_arc, this_prd_idx)
            sum_false_count += count
            new_columns.append(this_column)

        new_sentence_lst = []
        num_column = len(new_columns)
        for i in range(num_words):
            new_line_lst = []
            for j in range(num_column):
                new_line_lst.append(new_columns[j][i])
            new_sentence_lst.append(new_line_lst)
        new_sentence_lsts.append(new_sentence_lst)
    print('conflict I-:'+str(sum_false_count))
    with open(tgt_file, 'w') as f:
        for new_sentence_lst in new_sentence_lsts:
            for line_lst in new_sentence_lst:
                f.write(' '.join(line_lst) + '\n')
            f.write('\n')

def change_BIES(source_file, tgt_file, task):
    '''
    for BIES
    '''
    sum_false_count = 0
    if(task == '05'):
        word_idx_to_write = 2
    else:
        word_idx_to_write = 1
    with open(source_file, 'r') as f:
        lines = [line.strip() for line in f]
    sentences = []
    start, i = 0, 0
    for line in lines:
        if not line:
            sentences.append(lines[start:i])
            start = i + 1
        i += 1
    
    new_sentence_lsts = []
    for sentence in sentences:
        sentence_lst = []
        for line in sentence:
            sentence_lst.append(line.split('\t'))
        # sentence_lst:[line_lst,...,] line_lst:[num, word, lemma, _, pos, _, _, _, relas, _]

        # firstly find all predicates 
        num_words = len(sentence_lst)
        prd_map = {}  # 33:1, 44:2
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                continue
            relas = line_lst[8].split('|')
            for rela in relas:
                head, rel = rela.split(':')
                if (head == '0'):
                    prd_map[i] = len(prd_map) + 1
                    break
        
        arc_values = []
        # [[[a0],[a0]],]
        for i, line_lst in enumerate(sentence_lst, 1):
            if (line_lst[8] == '_'):
                arc_value = [[] for j in range(len(prd_map))]
                arc_values.append(arc_value)
            else:
                relas = line_lst[8].split('|')
                arc_value = [[] for j in range(len(prd_map))]
                for rela in relas:
                    head, rel = rela.split(':')
                    head_idx = int(head)
                    if (head_idx in prd_map):
                        # 这个步骤保证是srl结构，去掉0，和那些没有被预测为谓词的，边（这样应该好点，因为谓词预测准确率应该蛮高）
                        arc_value[prd_map[head_idx] - 1].append(rel)
                        # 应该只有一个，一个词根一个谓词只能有一个关系
                arc_values.append(arc_value)
        
        re_prd_map = {}  # 1:33, 2:44
        for key, value in prd_map.items():
            re_prd_map[value] = key

        new_columns = []
        column_1 = []
        for i, line_lst in enumerate(sentence_lst, 1):
            if (i in prd_map):
                column_1.append(line_lst[word_idx_to_write])
            else:
                column_1.append('-')
        new_columns.append(column_1)

        for key, value in re_prd_map.items():
            this_prd_arc = [
                word_arc_lsts[key - 1] for word_arc_lsts in arc_values
            ]
            # [[rel], [rel], [],...]
            this_prd_idx = value  # start from 1
            this_column, count = produce_column_BIES(this_prd_arc, this_prd_idx)
            sum_false_count += count
            new_columns.append(this_column)
        
        new_sentence_lst = []
        num_column = len(new_columns)
        for i in range(num_words):
            new_line_lst = []
            for j in range(num_column):
                new_line_lst.append(new_columns[j][i])
            new_sentence_lst.append(new_line_lst)
        new_sentence_lsts.append(new_sentence_lst)
    print('conflict I-:'+str(sum_false_count))
    with open(tgt_file, 'w') as f:
        for new_sentence_lst in new_sentence_lsts:
            for line_lst in new_sentence_lst:
                f.write(' '.join(line_lst) + '\n')
            f.write('\n')

def produce_column_BIES(relas, prd_idx):
    count = 0
    column = ['*'] * len(relas)
    column[prd_idx-1] = '(V*)'
    args = []
    i = 0
    while (i < len(relas)):
        rel = relas[i]
        if ((i + 1) == prd_idx):
            # column.append('(V*)')
            i += 1
        elif(rel == ['[prd]']):
            # column.append('*')
            i += 1
        elif (len(rel) == 0):
            # column.append('*')
            i += 1
        else:
            s_rel = rel[0]
            position_tag = s_rel[0]
            label = s_rel[2:]
            if position_tag in ('I', 'E'):
                # column.append('*')   # 直接把冲突的I删掉
                i += 1
                count += 1
            elif position_tag == 'S':
                # column.append('(' + label + '*' + ')')
                args.append([i, i, label])
                i += 1
            else:
                span_start = i
                i += 1
                if i>=len(relas):
                    # column.append('(' + label + '*' + ')')
                    i += 1
                elif relas[i][0].startswith('B-'):
                    count += 1
                    continue
                elif relas[i][0].startswith('E-'):
                    args.append([span_start, i, label])
                    i += 1
                elif relas[i][0].startswith('S-'):
                    args.append([i, i, relas[i][0][2:]])
                    count += 1
                    i += 1
                else:
                    # relas[i][0].startswith('I-')
                    while i < len(relas) and len(relas[i]) > 0 and relas[i][0].startswith('I-'):
                        i += 1
                    if i < len(relas):
                        args.append([span_start, i, label])
                        i += 1
    
    for st, ed, role in args:
        length = ed-st+1
        if length == 1:
            column[st] = '(' + role + '*' + ')'
        else:
            column[st] = '(' + role + '*'
            column[ed] = '*' + ')'
    
    return column, count

def produce_column_BII(relas, prd_idx):
    # 暂时是直接按照预测的B、I进行划分,然后选最多的label作为label
    count = 0
    # column = []
    column = ['*'] * len(relas)
    column[prd_idx-1] = '(V*)'
    args = []
    i = 0
    while (i < len(relas)):
        rel = relas[i]
        if ((i + 1) == prd_idx):
            # column.append('(V*)')
            i += 1
        elif(rel == ['[prd]']):
            # column.append('*')
            i += 1
        elif (len(rel) == 0):
            # column.append('*')
            i += 1
        else:
            s_rel = rel[0]
            position_tag = s_rel[0]
            label = s_rel[2:]
            # if (position_tag in ('B', 'I')):
                # 这里把I也考虑进来，防止第一个是I（I之前没有B，那么这个I当成B）
            if(position_tag == 'I'):
                # pdb.set_trace()
                # column.append('*')   # 直接把冲突的I删掉
                i += 1
                count += 1
                # pdb.set_trace()
            else:
                span_start = i
                i += 1
                labels = {}
                labels[label] = 1
                while (i < len(relas) and len(relas[i]) > 0):
                    if (relas[i][0][0] == 'I'):
                        labels[relas[i][0][2:]] = labels.get(
                            relas[i][0][2:], 0) + 1
                        i += 1
                    else:
                        # relas[i][0][0] == 'B' 直接把i指向下一个B
                        break
                length = i - span_start
                max_label = label
                max_num = 0
                for key, value in labels.items():
                    if (value > max_num):
                        max_num = value
                        max_label = key
                if (length == 1):
                    # column.append('(' + max_label + '*' + ')')
                    args.append([span_start, span_start, max_label])
                else:
                    # column.append('(' + max_label + '*')
                    # column += ['*'] * (length - 2)
                    # column.append('*' + ')')
                    args.append([span_start, i-1, max_label])
    for st, ed, role in args:
        length = ed-st+1
        if length == 1:
            column[st] = '(' + role + '*' + ')'
        else:
            column[st] = '(' + role + '*'
            column[ed] = '*' + ')'
    return column, count

def produce_column_BE(relas, prd_idx):
    # used for BE
    # 暂时是直接按照预测的B、I进行划分
    count = 0
    count2 = 0
    column = []
    # span_start = -1
    i = 0
    while (i < len(relas)):
        rel = relas[i]
        # print(i)
        # print(relas)
        if ((i + 1) == prd_idx):
            # 其实谓词不影响
            column.append('(V*)')
            i += 1
        elif(rel == ['[prd]']):
            column.append('*')
            i += 1
        elif(rel == ['Other']):
            column.append('*')
            i += 1
        elif (len(rel) == 0):
            column.append('*')
            i += 1
        else:
            s_rel = rel[0]
            position_tag = s_rel[0]
            label = s_rel[2:]  # label直接按第一个边界的label
            if(position_tag == 'I'):
                column.append('*')   # 直接把冲突的I删掉
                i += 1
                count += 1
            else:
            # if(position_tag in ('B', 'I')):
                span_start = i
                span_end = -1
                i += 1
                # labels = {}
                # labels[label] = 1
                while (i < len(relas)):
                    if (len(relas[i]) == 0 or relas[i] == ['Other']):
                        i += 1
                        continue
                    else:
                        # relas[i][0][0] == 'B' or 'I'
                        if (relas[i][0][0] == 'B'):
                            break
                        else:
                            span_end = i
                            label2 = relas[i][0][2:]  # 以后面那个作为label
                            i += 1
                            break
                if (span_end != -1):
                    if (label == label2):
                        length = span_end - span_start + 1
                        column.append('(' + label + '*')
                        column += ['*'] * (length - 2)
                        column.append('*' + ')')
                    else:
                        length = span_end - span_start + 1
                        column += ['*'] * length
                        count2 += 1
                else:
                    column.append('(' + label + '*' + ')')
                    column += ['*'] * (i - 1 - span_start)
    return column, count, count2

def produce_column_BES(relas, prd_idx):
    # used for BES
    flag = 0
    count = 0
    count2 = 0
    column = ['*'] * len(relas)
    column[prd_idx-1] = '(V*)'
    args = []
    i = 0
    while (i < len(relas)):
        rel = relas[i]
        if ((i + 1) == prd_idx):
            # 其实谓词不影响
            # column.append('(V*)')
            i += 1
        elif(rel == ['[prd]']):
            # column.append('*')
            i += 1
        elif (len(rel) == 0):
            # column.append('*')
            i += 1
        else:
            s_rel = rel[0]
            position_tag = s_rel[0]
            label = s_rel[2:]  # label直接按第一个边界的label
            if(position_tag == 'E'):
                column.append('*')   # 直接把冲突的I删掉
                i += 1
                count += 1
            elif position_tag == 'S':
                args.append([i, i, label])
                i += 1
            else:
                span_start = i
                i += 1
                if i>=len(relas):
                    # column.append('(' + label + '*' + ')')
                    i += 1
                elif len(relas[i]) == 0:
                    while i < len(relas) and len(relas[i]) == 0:
                        i += 1
                    if i < len(relas):
                        if relas[i][0].startswith('E-'):
                            new_label = relas[i][0][2:]
                            if label != new_label:
                                count2 += 1
                            else:
                                args.append([span_start, i, label])
                        else:
                            count += 1
                        i += 1
                elif relas[i][0].startswith('B-'):
                    count += 1
                    continue
                elif relas[i][0].startswith('E-'):
                    new_label = relas[i][0][2:]
                    args.append([span_start, i, label])
                    if label != new_label:
                        count2 += 1
                    i += 1
                else:
                    # relas[i][0].startswith('S-')
                    new_label = relas[i][0][2:]
                    args.append([i, i, new_label])
                    i += 1
                    count += 1

    for st, ed, role in args:
        length = ed-st+1
        if length == 1:
            column[st] = '(' + role + '*' + ')'
        else:
            column[st] = '(' + role + '*'
            column[ed] = '*' + ')'
    
    return column, count, count2


def get_srl_results(gold_path, pred_path, file_seed, task, schema):
    _SRL_CONLL_EVAL_SCRIPT = 'data/conll05-original-style/eval.sh'
    tgt_temp_file = 'tgt_temp_file' + file_seed
    if schema == 'BE':
        change_BE(pred_path, tgt_temp_file, task)
    elif schema == 'BII':
        change_BII(pred_path, tgt_temp_file, task)
    elif schema == 'BIES':
        change_BIES(pred_path, tgt_temp_file, task)
    elif schema == "BES":
        change_BES(pred_path, tgt_temp_file, task)
    else:
        raise NotImplementedError
    child = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, gold_path, tgt_temp_file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    eval_info = child.communicate()[0]
    print(eval_info)
    child2 = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, tgt_temp_file, gold_path), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    eval_info2 = child2.communicate()[0]
    print(eval_info2)
    # os.remove(tgt_temp_file)
    conll_recall = float(str(eval_info).strip().split("Overall")[1].strip().split('\\n')[0].split()[4])
    conll_precision = float(str(eval_info2).strip().split("Overall")[1].strip().split('\\n')[0].split()[4])
    conll_f1 = 2 * conll_recall * conll_precision / (conll_recall + conll_precision + 1e-12)
    lisa_f1 = float(str(eval_info).strip().split("Overall")[1].strip().split('\\n')[0].split()[5])

    return conll_recall, conll_precision, conll_f1, lisa_f1