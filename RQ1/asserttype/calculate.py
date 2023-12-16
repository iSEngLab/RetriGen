import pandas as pd
import argparse


def get_assertion_type(args):
    assertion_types = ['Equals', 'True', 'That', 'NotNull', 'False', 'Null', 'ArrayEquals', 'Same']
    assertion_types_with_assert = ['assert' + i for i in assertion_types]
    df = pd.read_csv(args.data_file)
    source = df['source']
    target = df['target']
    f1 = open(args.result_file, 'r', encoding="utf-8")
    preds1 = f1.read().splitlines()
    match1 = []
    for i in range(len(preds1)):
        if preds1[i] == "match:":
            match1.append(int(preds1[i + 1]))
    # # 构建一个二维列表来统计对应的单元格
    table_data = [[[0, 0] for _ in range(9)] for _ in range(1)]

    for i in range(len(source)):
        s = source[i]
        t = target[i]
        m1 = match1[i]
        row = 8
        n1 = t.split(" ")[0]
        n2 = t.split(" ")[6]
        if n1 in assertion_types_with_assert:
            row = assertion_types_with_assert.index(n1)
        elif n2 in assertion_types_with_assert:
            row = assertion_types_with_assert.index(n2)
        m = [m1]
        for j in range(1):
            table_data[j][row][0] += 1
            table_data[j][row][1] += m[j]
    for i in range(1):
        model_accu = 0
        for j in range(9):
            accu = table_data[i][j][1] / table_data[i][j][0]
            num = table_data[i][j][1]
            model_accu += num
            table_data[i][j] = '{} ({}%)'.format(num, round(accu * 100, 2))
        print(model_accu)
        print(model_accu / len(source))
    # 将统计结果转换为DataFrame
    columns_labels = assertion_types
    columns_labels.append('Other')
    index_labels = ['CodeT5_old']
    df = pd.DataFrame(table_data, index=index_labels, columns=columns_labels)
    print(df)
    df.to_csv(args.output_file, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, help="dataset file")
    parser.add_argument("--result_file", type=str, help="result file")
    parser.add_argument("--output_file", type=str, help="assert type output result file")
    args = parser.parse_args()

    get_assertion_type(args)
