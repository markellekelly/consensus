import numpy as np

def get_consensus(arr):
    y = list(arr)
    most = max(list(map(y.count, y)))
    modes = list(set(filter(lambda x: y.count(x) == most, y)))
    if len(modes) > 1:
        return None, modes
    return modes[0], modes


def print_Sigma(df, n, q=None):
    corr_str = ""
    for i in range(1, n+1):
        corr_str += str(i) + "----\t\t"
    print(corr_str)
    for i in range(1,n+1):
        row = ""
        for j in range(1,n+1):
            if q:
                corr = np.quantile(df['Sigma[{},{}]'.format(i,j)], q=q)
            else:
                corr = np.mean(df['Sigma[{},{}]'.format(i,j)])
            row += str(round(corr,3)) + "\t\t"
        print(row)

def create_data(n_items):
    y = [np.random.choice([0,1]) for _ in range(1000)]
    x1 = y
    x3 = y
    #x5 = y
    x2 = [np.random.choice([0,1]) for _ in range(1000)]
    #x4 = [np.random.choice([0,1]) for _ in range(500)]

    m1 = []
    acc = 0
    for i in range(1000):
        m_prob = np.random.uniform(0.55, 1)
        model_pred = np.random.binomial(n=1, p=m_prob)
        if y[i] == 0:
            model_pred = abs(1 - model_pred)
        if model_pred == y[i]:
            acc += 1
        model_conf = [0, 0]
        model_conf[model_pred] = m_prob
        model_conf[abs(1 - model_pred)] = 1 - m_prob
        m1.append(model_conf)
    data_dict = {
        'Y_M': [[m1[i]] for i in range(n_items)],
        'Y_H': [[x1[i]+1, x2[i]+1, x3[i]+1] for i in range(n_items)],
        'n_humans': 3,
        'n_models': 1,
        'K':2,
        'n_items':n_items
    }
    Y_M_test = [[m1[i]] for i in range(800, 1000)]
    Y_H_test = [[x1[i]+1, x2[i]+1, x3[i]+1] for i in range(800, 1000)]
    return data_dict, Y_M_test, Y_H_test