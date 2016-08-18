import pandas as pd
from os.path import join

from heuristic_parallel import HeuristicParallelModel
from model import original_model
import matplotlib.pyplot as plt


def chart(path, out_dir):
    d = pd.read_csv(path)

    plt.style.use('ggplot')
    size = (8, 8)

    figure = plt.figure(figsize=size)
    original_cost_line, = plt.plot(d['Project Num'].unique(), d['Original Time Cost (seconds)'], 'ro-',
                                   label='Original Time Cost')
    heuristic_cost_line, = plt.plot(d['Project Num'].unique(), d['Heuristic Time Cost (seconds)'], 'bo-',
                                    label='Heuristic Time Cost')
    plt.xlabel('Project Size')
    plt.ylabel('Time Cost (seconds)')
    plt.legend(handles=[original_cost_line, heuristic_cost_line])
    figure.savefig(join(out_dir, 'Time_Cost_Comparison.png'))

    figure = plt.figure(figsize=size)
    original_obj_line, = plt.plot(d['Project Num'].unique(), d['Original Obj Value'], 'ro-',
                                  label='Original Objective Value')
    heuristic_obj_line, = plt.plot(d['Project Num'].unique(), d['Heuristic Obj Value'], 'bo-',
                                   label='Heuristic Objective Value')
    plt.xlabel('Project Size')
    plt.ylabel('Objective Value')
    plt.legend(handles=[original_obj_line, heuristic_obj_line])
    figure.savefig(join(out_dir, 'Obj_Value_Comparison.png'))


if __name__ == "__main__":
    baes_input_path = './Inputs/P=%d/'
    base_output_path = './Output/P=%d/'
    result = pd.DataFrame(
        columns=['Project Num', 'Original Time Cost (seconds)', 'Original Obj Value', 'Heuristic Time Cost (seconds)',
                 'Heuristic Obj Value'])
    result_idx = 0

    for i in [10, 15, 20, 25, 30, 35, 40, 45][:2]:
        input_path = baes_input_path % i
        output_path = base_output_path % i

        (original_objValue, original_time_cost, m) = original_model(input_path, output_path)

        heuristic_model = HeuristicParallelModel(input_path, output_path)
        (heuristic_objValue, heuristic_time_cost) = heuristic_model.optimize()

        result.loc[result_idx] = [i, original_time_cost, original_objValue, heuristic_time_cost, heuristic_objValue]
        result_idx += 1

        result.to_csv('./Output/experiment_result.csv', index=False)

    chart('./Output/experiment_result.csv', './Output/')
