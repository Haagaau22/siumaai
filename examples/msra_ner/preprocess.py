

def load_raw_data(filepath_list):

    label_set = set()
    data_list = []
    index = 0
    for filepath in filepath_list:
        with open(filepath, encoding='utf8')as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                last_label = 'o'
                sub_text = ''
                text = ''
                entities = []
                for item in line.split(' '):
                    word, label = item.split('/')
                    label_set.add(label)
                    if sub_text != '' and label != last_label:
                        entities.append({
                            'start_idx': len(text)-len(sub_text),
                            'end_idx': len(text),
                            'entity': sub_text,
                            'label': last_label
                            })
                        sub_text = ''

                    if label != 'o':
                        sub_text += word
                        last_label = label

                    text += word
                if sub_text != '':
                    entities.append({
                        'start_idx': len(text)-len(sub_text),
                        'end_idx': len(text),
                        'entity': sub_text,
                        'label': last_label
                        })

                data_list.append({
                    'id': index,
                    'text': text,
                    'entities': entities
                    })
                index += 1
                
    return label_set, data_list


if __name__ == '__main__':

    import os
    import json

    label_set, data_list = load_raw_data([
        'msra/raw/train1.txt',
        'msra/raw/testright1.txt'
        ])
    label_set.remove('o')
    label_set.add('O')
    print(label_set)
    print(len(data_list))

    ner_root_path = 'msra/ner'

    if not os.path.isdir(ner_root_path):
        os.mkdir(ner_root_path)

    with open(os.path.join(ner_root_path, 'data.json'), 'w', encoding='utf8')as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)

    with open(os.path.join(ner_root_path, 'labels.txt'), 'w', encoding='utf8')as f:
        for label in label_set:
            f.write(f'{label}\n')

    import numpy as np
    len_list = []
    for data in data_list:
        len_list.append(len(data['text']))
    len_list = np.array(len_list)
    print(np.percentile(len_list, 80))
    print(np.percentile(len_list, 90))
    print(np.percentile(len_list, 95))
