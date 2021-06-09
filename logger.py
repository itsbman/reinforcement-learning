import pandas as pd
import os
import time


class Logger:
    def __init__(self, dir_name='111'):
        # model_name = config['brain']
        # if model_name == 'policy_net':
        #     model_name += f"-{config['brain_config']['model']}"

        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        # self.writer = SummaryWriter('log')
        self.log_dict = dict()

        self.output_file = dir_name + f'/run_stats-{time.time()}.csv'

    def write_log(self):
        max_n = max([len(data) for data in self.log_dict.values()])
        for key in self.log_dict:
            self.log_dict[key] += [''] * (max_n - len(self.log_dict[key]))
        df = pd.DataFrame(self.log_dict)
        if not os.path.isfile(self.output_file):
            df.to_csv(self.output_file, mode='a', index=False)
        else:
            df.to_csv(self.output_file, mode='a', index=False, header=False)
        self.log_dict.clear()

    def store(self, **kwargs):
        for k, v in kwargs.items():
            if not(k in self.log_dict.keys()):
                self.log_dict[k] = []
            self.log_dict[k].append(v)