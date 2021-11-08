def count_key_in_dict(dictionary, search_key, decision_fn=None):
    count = 0
    if type(dictionary) == dict:
        for k, v in dictionary.items():
            if decision_fn is None:
                count += k == search_key
            else:
                count += decision_fn(k, search_key)
            count += count_key_in_dict(v, search_key)
    return count


def extract_values_for_key_in_dict(dictionary, search_key, decision_fn=None):
    values = []
    if type(dictionary) == dict:
        for k, v in dictionary.items():
            if decision_fn is None:
                if k == search_key:
                    if type(v) == list:
                        values += v
                    else:
                        values += [v]
            elif decision_fn(k, search_key):
                if type(v) == list:
                    values += v
                else:
                    values += [v]
            values += extract_values_for_key_in_dict(v, search_key)
    return values
