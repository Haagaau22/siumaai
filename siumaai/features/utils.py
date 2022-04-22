

def compare_words_tokens(tokens, words, word_ids):
    previous_word_id = None
    new_words = []
    for index, word_id in enumerate(word_ids):
        if word_id is not None and previous_word_id == word_id:
            new_words[-1] += tokens[index][2:]
        elif word_id is not None and previous_word_id != word_id:
            new_words.append(tokens[index])
        previous_word_id = word_id

    print(words)
    print(new_words)
    for index, new_word in enumerate(new_words):
        if words[index].lower() != new_word:
            return False
    else:
        return True



def add_pad_for_labels(word_ids, labels, pad_id, token_type_ids, label_all_tokens=False, required_token_type_id=0, return_criterion_mask=False):

    new_labels = []
    criterion_mask = []
    previous_word_id = None
    for index, word_id in enumerate(word_ids):
        if word_id is None or token_type_ids[index] != required_token_type_id:
            new_labels.append(pad_id)
            criterion_mask.append(0)
        elif word_id !=  previous_word_id:
            new_labels.append(labels[word_id])
            criterion_mask.append(1)
        else:
            new_labels.append(labels[-1] if label_all_tokens is True else pad_id)
            criterion_mask.append(0)
        previous_word_id = word_id

    if return_criterion_mask is False:
        return new_labels
    return new_labels, criterion_mask



def remove_pad_for_labels(word_ids, labels, token_type_ids, required_token_type_id=0):
    new_labels = []

    previous_word_id = None
    for index, word_id in enumerate(word_ids):
        if word_id is None or token_type_ids[index] != required_token_type_id or word_id ==  previous_word_id:
            continue
        else:
            new_labels.append(labels[index])
        previous_word_id = word_id
    return new_labels


def add_pad_for_2d_labels(word_ids, labels, pad_id, token_type_ids, label_all_tokens=False, required_token_type_id=0, return_criterion_mask=False):

    new_labels = [[pad_id for _ in word_ids] for _ in word_ids]
    criterion_mask = [[0 for _ in word_ids] for _ in word_ids]
    x_previous_word_id = None
    for x_index, x_word_id in enumerate(word_ids):
        if token_type_ids[x_index] != required_token_type_id or x_word_id is None or (label_all_tokens is False and x_previous_word_id == x_word_id):
            continue

        y_previous_word_id = None
        for y_index, y_word_id in enumerate(word_ids):
            if token_type_ids[y_index] != required_token_type_id or y_word_id is None or (label_all_tokens is False and y_previous_word_id == y_word_id):
                continue

            if y_previous_word_id != y_word_id:
                criterion_mask[x_index][y_index] = 1
            new_labels[x_index][y_index] = labels[x_word_id][y_word_id]
            y_previous_word_id = y_word_id

        x_previous_word_id = x_word_id

    if return_criterion_mask is False:
        return new_labels
    return new_labels, criterion_mask


def remove_pad_for_2d_labels(word_ids, labels, token_type_ids, required_token_type_id=0):

    new_labels = []

    x_previous_word_id = None
    for x_index, x_word_id in enumerate(word_ids):
        if token_type_ids[x_index] != required_token_type_id or x_word_id is None or x_previous_word_id == x_word_id:
            continue

        _new_labels = []

        y_previous_word_id = None
        for y_index, y_word_id in enumerate(word_ids):
            if token_type_ids[y_index] != required_token_type_id or y_word_id is None or y_previous_word_id == y_word_id:
                continue

            _new_labels.append(labels[x_index][y_index])
            y_previous_word_id = y_word_id

        new_labels.append(_new_labels)
        x_previous_word_id = x_word_id

    return new_labels


def add_pad_for_3d_labels(
        word_ids, 
        labels, 
        pad_id, 
        token_type_ids, 
        label_all_tokens=False, 
        required_token_type_id=0, 
        return_criterion_mask=False):
    new_labels = []
    criterion_mask = []
    for _labels in labels:
        _new_labels, _criterion_mask = add_pad_for_2d_labels(
                word_ids, 
                _labels, 
                pad_id, 
                token_type_ids, 
                label_all_tokens=label_all_tokens, 
                required_token_type_id=required_token_type_id, 
                return_criterion_mask=return_criterion_mask)
        new_labels.append(_new_labels)
        criterion_mask.append(_criterion_mask)

    if return_criterion_mask is False:
        return new_labels
    return new_labels, criterion_mask


def remove_pad_for_3d_labels(word_ids, labels, token_type_ids, required_token_type_id=0):

    return [
        remove_pad_for_2d_labels(word_ids, _labels, token_type_ids, required_token_type_id=required_token_type_id)
        for _labels in labels
    ]

    # for _labels in labels:
    #     import ipdb; ipdb.set_trace()
    #     _new_labels = remove_pad_for_2d_labels(word_ids, _labels, token_type_ids, required_token_type_id=required_token_type_id)
    #     print(_new_labels)

