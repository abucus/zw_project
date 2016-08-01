import pandas as pd
from os.path import join

from heuristic_parallel import HeuristicParallelModel
from model import original_model
import matplotlib.pyplot as plt


def chart(path, out_dir):
    d = pd.read_csv(path)

    plt.style.use('ggplot')
    size = (8, 8)

    original_cost = d[d['Method'] == 'Original']['Time Cost (seconds)']

    heuristic_cost = d[d['Method'] == 'Heuristic']['Time Cost (seconds)']

    original_objvalue = d[d['Method'] == 'Original']['Obj Value']

    heuristic_objvalue = d[d['Method'] == 'Heuristic']['Obj Value']

    figure = plt.figure(figsize=size)
    original_cost_line, = plt.plot(d['Project Num'].unique(), original_cost.tolist(), 'ro-', label='Original Time Cost')
    heuristic_cost_line, = plt.plot(d['Project Num'].unique(), heuristic_cost.tolist(), 'bo-',
                                    label='Heuristic Time Cost')
    plt.xlabel('Project Size')
    plt.ylabel('Time Cost (seconds)')
    plt.legend(handles=[original_cost_line, heuristic_cost_line])
    figure.savefig(join(out_dir, 'Time_Cost_Comparison.png'))

    figure = plt.figure(figsize=size)
    original_obj_line, = plt.plot(d['Project Num'].unique(), original_objvalue.tolist(), 'ro-',
                                  label='Original Objective Value')
    heuristic_obj_line, = plt.plot(d['Project Num'].unique(), heuristic_objvalue.tolist(), 'bo-',
                                   label='Heuristic Objective Value')
    plt.xlabel('Project Size')
    plt.ylabel('Objective Value')
    plt.legend(handles=[original_obj_line, heuristic_obj_line])
    figure.savefig(join(out_dir, 'Obj_Value_Comparison.png'))


if __name__ == "__main__":
    baes_input_path = './Inputs/P=%d/'
    base_output_path = './Output/P=%d/'
    result = pd.DataFrame(columns=['Project Num', 'Method', 'Time Cost (seconds)', 'Obj Value'])
    result_idx = 0

    for i in [10, 15, 20, 25, 30, 35, 40, 45]:
        input_path = baes_input_path % i
        output_path = base_output_path % i

        (objValue, time_cost) = original_model(input_path, output_path)
        result.loc[result_idx] = [i, 'Original', time_cost, objValue]
        result_idx += 1

        heuristic_model = HeuristicParallelModel(input_path, output_path)
        (objValue, time_cost) = heuristic_model.optimize()
        result.loc[result_idx] = [i, 'Heuristic', time_cost, objValue]
        result_idx += 1

        result.to_csv('./Output/experiment_result.csv', index=False)

    chart('./Output/experiment_result.csv', './Output/')