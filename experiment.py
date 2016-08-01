from heuristic import HeuristicModel
from heuristic_parallel import HeuristicParallelModel
from model import original_model
import pandas as pd

if __name__ == "__main__":
    baes_input_path = './Inputs/P=%d/'
    base_output_path = './Output/P=%d/'
    result = pd.DataFrame(columns=['Project Num', 'Method', 'Time Cost (seconds)', 'Obj Value'])
    result_idx = 0

    for i in [10, 15, 20, 25, 30, 35, 40, 45][:1]:
        input_path = baes_input_path % i
        output_path = base_output_path % i

        (objValue, time_cost) = original_model(input_path, output_path)
        result.loc[result_idx] = [i, 'Original', time_cost, objValue]
        result_idx += 1

        heuristic_model = HeuristicParallelModel(input_path, output_path)
        (objValue, time_cost) = heuristic_model.optimize()
        result.loc[result_idx] = [i, 'Original', time_cost, objValue]
        result_idx += 1

        result.to_csv('./Output/experiment_result.csv', index=False)
